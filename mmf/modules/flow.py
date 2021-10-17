import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from typing import Dict, Tuple

class InvertibleMultiHeadFlow(nn.Module):
    @staticmethod
    def _get_heads(in_features):
        units = [32, 16, 8]
        for unit in units:
            if in_features % unit == 0:
                return in_features // unit
        assert in_features < 8, 'features={}'.format(in_features)
        return 1

    def __init__(self, in_features, heads=None, type='A'):
        super(InvertibleMultiHeadFlow, self).__init__()
        self.in_features = in_features
        if heads is None:
            heads = InvertibleMultiHeadFlow._get_heads(in_features)
        self.heads = heads
        self.type = type
        assert in_features % heads == 0, 'features ({}) should be divided by heads ({})'.format(in_features, heads)
        assert type in ['A', 'B'], 'type should belong to [A, B]'
        self.weight = Parameter(torch.Tensor(in_features // heads, in_features // heads))
        self.register_buffer('weight_inv', self.weight.data.clone())
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        self.sync()

    def sync(self):
        self.weight_inv.copy_(self.weight.data.inverse())

    def forward(self, input, mask, fwd):
        """
        Args:
            input: Tensor
                input tensor [batch, N1, N2, ..., Nl, in_features]
            mask: Tensor
                mask tensor [batch, N1, N2, ...,Nl]
        Returns: out: Tensor , logdet: Tensor
            out: [batch, N1, N2, ..., in_features], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`
        """
        if fwd:
            size = input.size()
            dim = input.dim()
            # [batch, N1, N2, ..., heads, in_features/ heads]
            if self.type == 'A':
                out = input.view(*size[:-1], self.heads, self.in_features // self.heads)
            else:
                out = input.view(*size[:-1], self.in_features // self.heads, self.heads).transpose(-2, -1)

            out = F.linear(out, self.weight)
            if self.type == 'B':
                out = out.transpose(-2, -1).contiguous()
            out = out.view(*size)

            return out

        else:
            size = input.size()
            dim = input.dim()
            # [batch, N1, N2, ..., heads, in_features/ heads]
            if self.type == 'A':
                out = input.view(*size[:-1], self.heads, self.in_features // self.heads)
            else:
                out = input.view(*size[:-1], self.in_features // self.heads, self.heads).transpose(-2, -1)

            out = F.linear(out, self.weight_inv)
            if self.type == 'B':
                out = out.transpose(-2, -1).contiguous()
            out = out.view(*size)

            return out


class ActNormFlow(nn.Module):
    def __init__(self, in_features):
        super(ActNormFlow, self).__init__()
        self.in_features = in_features
        self.log_scale = Parameter(torch.Tensor(in_features))
        self.bias = Parameter(torch.Tensor(in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.log_scale, mean=0, std=0.05)
        nn.init.constant_(self.bias, 0.)

    def forward(self, input, mask, fwd):
        """
        Args:
            input: Tensor
                input tensor [batch, N1, N2, ..., Nl, in_features]
            mask: Tensor
                mask tensor [batch, N1, N2, ...,Nl]
        Returns: out: Tensor , logdet: Tensor
            out: [batch, N1, N2, ..., in_features], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`
        """
        if fwd:
            dim = input.dim()
            out = input * self.log_scale.exp() + self.bias
            out = out * mask.unsqueeze(dim - 1)
            return out
        else:
            dim = input.dim()
            out = (input - self.bias) * mask.unsqueeze(dim - 1)
            out = out.div(self.log_scale.exp() + 1e-8)
            return out



class LinearWeightNorm(nn.Module):
    """
    Linear with weight normalization
    """
    def __init__(self, in_features, out_features, bias=True):
        super(LinearWeightNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.05)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)
        self.linear = nn.utils.weight_norm(self.linear)

    def forward(self, input):
        return self.linear(input)


class NICEBlock(nn.Module):
    def __init__(self, in_features, out_features, hidden_features):
        super(NICEBlock, self).__init__()
        self.layer1 = LinearWeightNorm(in_features, hidden_features, bias=True)
        self.activation = nn.ELU(inplace=True)
        self.layer2 = LinearWeightNorm(hidden_features, out_features, bias=True)

    def forward(self, x, mask):
        out = self.activation(self.layer1(x))
        out = self.layer2(out)
        return out

class Affine():
    @staticmethod
    def fwd(z: torch.Tensor, mask: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_scale = params.chunk(2, dim=2)
        scale = log_scale.add_(2.0).sigmoid_()
        z = (scale * z + mu).mul(mask.unsqueeze(2))
        return z

    @staticmethod
    def bwd(z: torch.Tensor, mask: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_scale = params.chunk(2, dim=2)
        scale = log_scale.add_(2.0).sigmoid_()
        z = (z - mu).div(scale + 1e-12).mul(mask.unsqueeze(2))
        return z



class AffineCoupling(nn.Module):
    """
    NICE Flow
    """
    def __init__(self, features, hidden_features=None, split_dim=2, split_type='continuous', order='up', factor=2,
                 transform='affine', type='conv', kernel=3, rnn_mode='LSTM', heads=1, dropout=0.0, pos_enc='add'):
        super(AffineCoupling, self).__init__()
        self.features = features
        self.split_dim = split_dim
        self.split_type = split_type

        self.up = order == 'up'
        out_features = features // factor
        in_features = features - out_features
        self.z1_channels = in_features if self.up else out_features

        self.transform = Affine
        out_features = out_features * 2

        self.net = NICEBlock(in_features, out_features, hidden_features)

    def split(self, z, mask):
        split_dim = self.split_dim
        split_type = self.split_type
        dim = z.size(split_dim)
        return z.split([self.z1_channels, dim - self.z1_channels], dim=split_dim), mask

    def unsplit(self, z1, z2):
        split_dim = self.split_dim
        split_type = self.split_type
        return torch.cat([z1, z2], dim=split_dim)

    def calc_params(self, z: torch.Tensor, mask: torch.Tensor):
        params = self.net(z, mask)
        return params

    def forward(self, input, mask, fwd):
        if fwd:
            # [batch, length, in_channels]
            (z1, z2), mask = self.split(input, mask)
            # [batch, length, features]
            z, zp = (z1, z2) if self.up else (z2, z1)

            params = self.calc_params(z, mask)
            zp = self.transform.fwd(zp, mask, params)

            z1, z2 = (z, zp) if self.up else (zp, z)
            return self.unsplit(z1, z2)
        else:
            # [batch, length, in_channels]
            (z1, z2), mask = self.split(input, mask)
            # [batch, length, features]
            z, zp = (z1, z2) if self.up else (z2, z1)

            params = self.calc_params(z, mask)
            zp = self.transform.bwd(zp, mask, params)

            z1, z2 = (z, zp) if self.up else (zp, z)
            return self.unsplit(z1, z2)


class FlowLayer(nn.Module):
    def __init__(self, features=2048, hidden_features=2048, heads=32):
        super(FlowLayer, self).__init__()
        self.actnorm1 = ActNormFlow(features)
        self.linear1 = InvertibleMultiHeadFlow(features, heads, type='A')
        self.unit1_1 = AffineCoupling(features, hidden_features, order='up')
        self.unit1_2 = AffineCoupling(features, hidden_features, order='down')
        #self.actnorm2 = ActNormFlow(features)
        #self.linear2 = InvertibleMultiHeadFlow(features, heads, type='B')
        #self.unit2_1 = AffineCoupling(features, hidden_features, order='up')
        #self.unit2_2 = AffineCoupling(features, hidden_features, order='down')

    def forward(self, input, mask, fwd):
        if fwd:
            x = self.actnorm1(input, mask, fwd)
            x = self.linear1(x, mask, fwd)
            x = self.unit1_1(x, mask, fwd)
            x = self.unit1_2(x, mask, fwd)

            #x = self.actnorm2(x, mask, fwd)
            #x = self.linear2(x, mask, fwd)
            #x = self.unit2_1(x, mask, fwd)
            #x = self.unit2_2(x, mask, fwd)
            return x
        else:
            #x = self.unit2_2(input, mask, fwd)
            #x = self.unit2_1(x, mask, fwd)
            #x = self.linear2(x, mask, fwd)
            #x = self.actnorm2(x, mask, fwd)
             
            x = self.unit1_2(input, mask, fwd)
            x = self.unit1_1(x, mask, fwd)
            x = self.linear1(x, mask, fwd)
            x = self.actnorm1(x, mask, fwd)
            return x
             


