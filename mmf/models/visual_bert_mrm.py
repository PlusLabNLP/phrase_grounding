# Copyright (c) Facebook, Inc. and its affiliates.
# Initial version was taken from https://github.com/uclanlp/visualbert
# which was cleaned up and adapted for MMF.

import os
from copy import deepcopy
import math

import torch
from mmf.common.registry import registry
from mmf.models import BaseModel
from mmf.modules.embeddings import BertVisioLinguisticEmbeddings, ImageBertVisioLinguisticEmbeddings
from mmf.utils.configuration import get_mmf_cache_dir
from mmf.utils.modeling import get_optimizer_parameters_for_bert
from mmf.utils.transform import (
    transform_to_batch_sequence,
    transform_to_batch_sequence_dim,
)
from omegaconf import OmegaConf
from torch import nn
from transformers.modeling_bert import (
    BertConfig,
    BertEncoder,
    BertForPreTraining,
    BertLayer,
    BertPooler,
    BertPredictionHeadTransform,
    BertPreTrainedModel,
)


class VisualBERTMRMBase(BertPreTrainedModel):
    def __init__(
        self,
        config,
        visual_embedding_dim=512,
        embedding_strategy="plain",
        bypass_transformer=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        super().__init__(config)
        self.config = config

        config.visual_embedding_dim = visual_embedding_dim
        config.embedding_strategy = embedding_strategy
        config.bypass_transformer = bypass_transformer
        config.output_attentions = output_attentions
        config.output_hidden_states = output_hidden_states

        self.embeddings = ImageBertVisioLinguisticEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.bypass_transformer = config.bypass_transformer

        if self.bypass_transformer:
            self.additional_layer = BertLayer(config)

        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states
        self.fixed_head_masks = [None for _ in range(len(self.encoder.layer))]
        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        visual_embeddings=None,
        position_embeddings_visual=None,
        visual_embeddings_type=None,
        image_text_alignment=None,
        image_info=None,
        image_embeddings=None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of
        # causal attention used in OpenAI GPT, we just need to prepare the
        # broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output, vis_embeddings = self.embeddings(
            input_ids,
            token_type_ids,
            visual_embeddings=visual_embeddings,
            position_embeddings_visual=position_embeddings_visual,
            visual_embeddings_type=visual_embeddings_type,
            image_text_alignment=image_text_alignment,
            image_info=image_info,
            image_embeddings=image_embeddings,
        )

        if self.bypass_transformer and visual_embeddings is not None:
            assert (
                not self.output_hidden_states
            )  # Don't support this for the bypass model
            text_length = input_ids.size(1)
            text_embedding_output = embedding_output[:, :text_length, :]
            visual_part = embedding_output[:, text_length:, :]

            text_extended_attention_mask = extended_attention_mask[
                :, :, :text_length, :text_length
            ]

            encoded_layers = self.encoder(
                text_embedding_output,
                text_extended_attention_mask,
                self.fixed_head_masks,
            )
            sequence_output = encoded_layers[0]
            new_input = torch.cat((sequence_output, visual_part), dim=1)
            final_sequence_output = self.additional_layer(
                new_input, extended_attention_mask
            )
            pooled_output = self.pooler(final_sequence_output)
            return final_sequence_output, pooled_output, vis_embeddings

        else:
            encoded_layers = self.encoder(
                embedding_output, extended_attention_mask, self.fixed_head_masks
            )
            sequence_output = encoded_layers[0]
            pooled_output = self.pooler(sequence_output)
            attn_data_list = []

            if self.output_attentions:
                attn_data_list = encoded_layers[1:]

            return sequence_output, pooled_output, attn_data_list, vis_embeddings


class VisualBERTMRMForPretraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states

        # If bert_model_name is not specified, you will need to specify
        # all of the required parameters for BERTConfig and a pretrained
        # model won't be loaded
        self.bert_model_name = getattr(self.config, "bert_model_name", None)
        self.bert_config = BertConfig.from_dict(
            OmegaConf.to_container(self.config, resolve=True)
        )
        if self.bert_model_name is None:
            self.bert = VisualBERTMRMBase(
                self.bert_config,
                visual_embedding_dim=self.config.visual_embedding_dim,
                embedding_strategy=self.config.embedding_strategy,
                bypass_transformer=self.config.bypass_transformer,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states,
            )
        else:
            self.bert = VisualBERTMRMBase.from_pretrained(
                self.config.bert_model_name,
                config=self.bert_config,
                cache_dir=os.path.join(
                    get_mmf_cache_dir(), "distributed_{}".format(-1)
                ),
                visual_embedding_dim=self.config.visual_embedding_dim,
                embedding_strategy=self.config.embedding_strategy,
                bypass_transformer=self.config.bypass_transformer,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states,
            )

        self.vocab_size = self.bert.config.vocab_size

        # TODO: Once omegaconf fixes int keys issue, bring this back
        # See https://github.com/omry/omegaconf/issues/149
        # with omegaconf.open_dict(self.config):
        #     # Add bert config such as hidden_state to our main config
        #     self.config.update(self.bert.config.to_dict())
        if self.bert_model_name is None:
            bert_masked_lm = BertForPreTraining(self.bert.config)
        else:
            bert_masked_lm = BertForPreTraining.from_pretrained(
                self.config.bert_model_name,
                cache_dir=os.path.join(
                    get_mmf_cache_dir(), "distributed_{}".format(-1)
                ),
            )
        self.cls = deepcopy(bert_masked_lm.cls)
        self.cls_vis1 = BertPredictionHeadTransform(self.bert.config)
        self.cls_vis2 = nn.Linear(self.bert.config.hidden_size, self.config.visual_embedding_dim)

        self.input_vis = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.loss_fct_vis = nn.CrossEntropyLoss(ignore_index=-1) #nn.SmoothL1Loss()
        self.init_weights()

    def init_weights(self):
        if self.config.random_initialize is False:
            if self.bert_model_name is None:
                # No pretrained model, init weights
                self.bert.init_weights()
                self.cls.apply(self.bert._init_weights)

            self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them
            instead.
        """
        self.bert._tie_or_clone_weights(
            self.cls.predictions.decoder, self.bert.embeddings.word_embeddings
        )

    def _tie_or_clone_weights_transpose(self, output_embeddings, input_embeddings):
        output_embeddings.weight = nn.Parameter(input_embeddings.weight.transpose(0, 1).clone())

    def forward(
        self,
        input_ids,
        input_mask,
        attention_mask=None,
        token_type_ids=None,
        visual_embeddings=None,
        position_embeddings_visual=None,
        visual_embeddings_type=None,
        image_text_alignment=None,
        masked_lm_labels=None,
        image_labels=None,
        image_info=None,
        image_mask=None,
        image_text_label=None,
        images=None,
        regress_mask=None,
        cls_prob_mask=None,
        is_correct=None,
        targets=None,
        image_id=None,
        start_ind=None,
        end_ind=None,
    ):
        if cls_prob_mask is not None:
            image_mask = cls_prob_mask.bool()
        image_embeddings=None

        masked_visual_embeddings = visual_embeddings.clone()
        masked_visual_embeddings[image_labels==0] = 0
        sequence_output, pooled_output, attention_weights, vis_embeddings = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            masked_visual_embeddings,
            position_embeddings_visual,
            visual_embeddings_type,
            image_text_alignment,
            image_info=image_info,
            image_embeddings=image_embeddings,
        )

        output_dict = {}




        if self.output_attentions:
            output_dict["attention_weights"] = attention_weights

        if self.output_hidden_states:
            output_dict["sequence_output"] = sequence_output
            output_dict["pooled_output"] = pooled_output

        text_length = input_ids.size(1)
        text_hidden = sequence_output[:, :text_length, :]

        prediction_scores, seq_relationship_score = self.cls(
            text_hidden, pooled_output
        )

        seq_relationship_score = seq_relationship_score[is_correct != -1]
        if len(seq_relationship_score) > 0:
            seq_label = is_correct[is_correct != -1]
            global_loss = self.loss_fct(
                seq_relationship_score.contiguous().view(-1, 2),
                seq_label.contiguous().view(-1),
            )
            output_dict['global_loss'] = global_loss
            

        if masked_lm_labels is not None:
            if len(masked_lm_labels) > 0:
                output_dict["logits"] = prediction_scores
                masked_lm_loss = self.loss_fct(
                    prediction_scores.contiguous().view(-1, self.vocab_size),
                    masked_lm_labels.contiguous().view(-1),
                )
                output_dict["masked_lm_loss"] = masked_lm_loss #'''


        vis_hidden = sequence_output[:, text_length:, :]


        
        if True: #sim_loss
            vis_hidden_t = vis_hidden[is_correct==-1]
            if len(vis_hidden_t) > 0:
                text_hidden_t = text_hidden[is_correct==-1]
                sim = torch.matmul(vis_hidden_t, text_hidden_t.transpose(-1, -2))/math.sqrt(768)
                input_ids_t = input_ids[is_correct==-1]
                mask_text = 1-((input_ids_t==0)+(input_ids_t==101)+(input_ids_t==102)).float()
                sim1 = nn.Softmax(dim=-1)(sim)
                sim2 = nn.Softmax(dim=-2)(sim)
                sim = (torch.matmul(sim1, sim2.transpose(-1, -2)))
                sim_loss = torch.einsum('bii->b', sim) #-torch.trace(sim)
                
                len_denom = torch.sum(mask_text, -1)
                len_denom = torch.max(torch.min(len_denom, torch.ones_like(len_denom)*64), torch.ones_like(len_denom))
                sim_loss = -0.1*torch.mean(torch.log(sim_loss/len_denom+1))
                output_dict["sim_loss"] = sim_loss

        if True: #olp
            masklabel2 = ((image_labels!=-2))
            masklabel2 = masklabel2.contiguous().view(-1)
            vis_text_scores = self.cls.predictions(vis_hidden)
            vis_text_scores = vis_text_scores.view(-1, self.vocab_size)
            image_text_label = image_text_label.view(-1, self.vocab_size)
            vis_text_scores = vis_text_scores[masklabel2]
            image_text_label = image_text_label[masklabel2]
            if len(vis_text_scores) > 0:
                vis_text_scores = -torch.nn.functional.log_softmax(vis_text_scores, dim=-1)
                vis_text_scores = vis_text_scores*image_text_label
                masked_region_loss2 = vis_text_scores.sum() / masklabel2.sum() #MASK this'''
            else:
                masked_region_loss2 = None
        else:
            masked_region_loss2 = None

        if True: #mrm
            vis_hidden = self.cls_vis1(vis_hidden)
            batch_size = vis_hidden.size(0)
            image_length = vis_hidden.size(1)
            all_cand = True
            vis_scores = self.cls_vis2(vis_hidden)
            if all_cand: 
                image_cls_labels = torch.arange(image_length*batch_size, device=vis_hidden.device)
            else:
                image_cls_labels = torch.arange(image_length, device=vis_hidden.device).repeat(batch_size)
            select_image_labels = ((image_labels!=-1)&(image_labels!=-2)).contiguous().view(-1)
            image_cls_labels = image_cls_labels[select_image_labels]
            #import math
            if len(image_cls_labels) > 0:
                if all_cand: 
                    expand_visual_embeddings = visual_embeddings.view(batch_size*image_length, -1)
                    vis_scores = vis_scores / torch.norm(vis_scores, dim=-1, keepdim=True) #.detach()
                    expand_visual_embeddings = expand_visual_embeddings / torch.norm(expand_visual_embeddings, dim=-1, keepdim=True) #.detach()
                    masked_vis_scores1 = torch.matmul(vis_scores, expand_visual_embeddings.transpose(-1, -2)) # /math.sqrt(768)
                    masked_vis_scores = masked_vis_scores1.view(batch_size*image_length, -1)
                    masked_vis_scores = masked_vis_scores[select_image_labels]
                else:
                    vis_scores = vis_scores / torch.norm(vis_scores, dim=-1, keepdim=True).detach()
                    visual_embeddings = visual_embeddings / torch.norm(visual_embeddings, dim=-1, keepdim=True).detach()
                    masked_vis_scores1 = torch.matmul(vis_scores, visual_embeddings.transpose(-1, -2)) #/ math.sqrt(768)
                    masked_vis_scores = masked_vis_scores1.view(-1, image_length)
                    masked_vis_scores = masked_vis_scores[select_image_labels]

                masked_region_loss = self.loss_fct_vis(masked_vis_scores, image_cls_labels) #'''
            else:
                masked_region_loss = None
        else:
            masked_region_loss = None
        
        if masked_region_loss is not None:
            if masked_region_loss2 is not None:
                output_dict["masked_region_loss"] = masked_region_loss2 + masked_region_loss
            else:
                output_dict["masked_region_loss"] = masked_region_loss
        elif masked_region_loss2 is not None:
            output_dict["masked_region_loss"] = masked_region_loss2
        return output_dict


class VisualBERTMRMForClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states
        self.pooler_strategy = self.config.get("pooler_strategy", "default")

        # If bert_model_name is not specified, you will need to specify
        # all of the required parameters for BERTConfig and a pretrained
        # model won't be loaded
        self.bert_model_name = getattr(self.config, "bert_model_name", None)
        self.bert_config = BertConfig.from_dict(
            OmegaConf.to_container(self.config, resolve=True)
        )
        if self.bert_model_name is None:
            self.bert = VisualBERTMRMBase(
                self.bert_config,
                visual_embedding_dim=self.config.visual_embedding_dim,
                embedding_strategy=self.config.embedding_strategy,
                bypass_transformer=self.config.bypass_transformer,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states,
            )
        else:
            self.bert = VisualBERTMRMBase.from_pretrained(
                self.config.bert_model_name,
                config=self.bert_config,
                cache_dir=os.path.join(
                    get_mmf_cache_dir(), "distributed_{}".format(-1)
                ),
                visual_embedding_dim=self.config.visual_embedding_dim,
                embedding_strategy=self.config.embedding_strategy,
                bypass_transformer=self.config.bypass_transformer,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states,
            )

        self.training_head_type = self.config.training_head_type
        self.num_labels = self.config.num_labels
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        if not 'retrieval' in self.config.training_head_type:
            if self.config.training_head_type == "nlvr2":
                self.bert.config.hidden_size *= 2
            elif 'refcoco2' in self.config.training_head_type:
                self.config.num_labels = 1
                self.bert.config.hidden_size *= 2 
            elif 'flickr' in self.config.training_head_type or 'refcoco' in self.config.training_head_type:
                self.config.num_labels = 1 
                self.bert.config.hidden_size *= 2
        
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.config.num_labels),
        )

        self.init_weights()

    def init_weights(self):
        if self.config.random_initialize is False:
            if self.bert_model_name is None:
                # No pretrained model, init weights
                self.bert.init_weights()

            # Classifier needs to be initialized always as it is task specific
            self.classifier.apply(self.bert._init_weights)

    def forward(
        self,
        input_ids,
        input_mask,
        attention_mask=None,
        token_type_ids=None,
        visual_embeddings=None,
        position_embeddings_visual=None,
        visual_embeddings_type=None,
        image_text_alignment=None,
        masked_lm_labels=None,
        image_labels=None,
        image_info=None,
        image_mask=None,
        image_text_label=None,
        images=None,
        regress_mask=None,
        cls_prob_mask=None,
        is_correct=None,
        targets=None,
        image_id=None,
        start_ind=None,
        end_ind=None,
    ):
        #image_info=None
        sequence_output, pooled_output, attention_weights, vis_embeddings = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            visual_embeddings,
            position_embeddings_visual,
            visual_embeddings_type,
            image_text_alignment,
            image_info=image_info,
        )

        if self.training_head_type == "nlvr2":
            # 2B * H => B * 2H
            b, h = pooled_output.size()
            pooled_output = torch.cat(
                [pooled_output[: b // 2], pooled_output[b // 2 :]], dim=1
            )

        output_dict = {}
        if self.output_attentions:
            output_dict["attention_weights"] = attention_weights

        if self.output_hidden_states:
            output_dict["sequence_output"] = sequence_output
            output_dict["pooled_output"] = pooled_output

        if self.pooler_strategy == "vqa":
            # In VQA2 pooling strategy, we use representation from second last token
            index_to_gather = input_mask.sum(1) - 2
            pooled_output = torch.gather(
                sequence_output,
                1,
                index_to_gather.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(index_to_gather.size(0), 1, sequence_output.size(-1)),
            )
        elif self.training_head_type == 'refcoco2':
            text_length = input_ids.size(1)
            sequence_output_v = sequence_output[:, text_length:, :]
            sequence_output_t = sequence_output[:, :text_length, :]
            avg_text = []
            avg_text2 = []
            for i, (sid, eid) in enumerate(zip(start_ind, end_ind)):
                seq_t = (sequence_output_t[i, sid+1:eid+1, :]) #textlen, 768

                seq_t2 = torch.max(seq_t, 0)[0]
                seq_t2 = seq_t2.unsqueeze(0)
                avg_text2.append(seq_t2.squeeze(0))#'''

            avg_text2 = torch.stack(avg_text2) #[bs, 768]
            avg_text2 = avg_text2.unsqueeze(1).repeat(1, sequence_output_v.size(1), 1)
            bs = sequence_output_v.size(0)
            vis_len = sequence_output_v.size(1)
            sequence_v = torch.cat([sequence_output_v, avg_text2], -1).view(sequence_output_v.size(0)*sequence_output_v.size(1), -1)
            sequence_v = self.classifier(sequence_v)
            sequence_v = sequence_v.view(bs, vis_len, -1)
            sim = nn.Softmax(dim=-2)(sequence_v)
            output_dict['scores'] = sim
            return output_dict
        elif self.training_head_type == 'refcoco':
            text_length = input_ids.size(1)
            sequence_output_v = sequence_output[:, text_length:, :]
            sequence_output_t = sequence_output[:, :text_length, :]
            avg_text = []
            avg_text2 = []
            for i, (sid, eid) in enumerate(zip(start_ind, end_ind)):
                seq_t = (sequence_output_t[i, sid+1:eid+1, :]) #textlen, 768

                seq_tt = torch.matmul(sequence_output_v[i], seq_t.transpose(-1, -2)) /math.sqrt(768) #vislen,textslen
                seq_t2 = torch.matmul(seq_tt, seq_t) #vis_len, 768
                avg_text.append(seq_t2)


            avg_text = torch.stack(avg_text) #[bs, 768]
            bs = sequence_output_v.size(0)
            vis_len = sequence_output_v.size(1)
            sequence_v = torch.cat([sequence_output_v, avg_text], -1).view(sequence_output_v.size(0)*sequence_output_v.size(1), -1)
            sequence_v = self.classifier(sequence_v)
            sequence_v = sequence_v.view(bs, vis_len, -1)


            sequence_v = sequence_v + (1-cls_prob_mask.float())*-10000000.0
            sim = nn.Softmax(dim=-2)(sequence_v)
            output_dict['scores'] = sim
            return output_dict

        elif self.training_head_type == 'flickr30k':
            text_length = input_ids.size(1)
            sequence_output_v = sequence_output[:, text_length:, :]
            sequence_output_t = sequence_output[:, :text_length, :]
            avg_text = []
            for i, (sid, eid) in enumerate(zip(start_ind, end_ind)):
                seq_t = (sequence_output_t[i, sid+1:eid+1, :])
                seq_t2 = torch.mean(seq_t, 0).unsqueeze(0)
                for _ in range(0):
                    att = torch.matmul(seq_t2, sequence_output_v[i].transpose(-1, -2)) /math.sqrt(768)
                    att = nn.Softmax(dim=-1)(att)
                    seq_v = torch.matmul(att, sequence_output_v[i])
                    att = torch.matmul(seq_t, seq_v.transpose(-1, -2)) /math.sqrt(768)
                    att = nn.Softmax(dim=0)(att)
                    seq_t2=(torch.sum(att*seq_t, 0)).unsqueeze(0)
                if sid >= len(sequence_output_t[i]):
                    print(sid)
                avg_text.append(seq_t2.squeeze(0))

            avg_text = torch.stack(avg_text) #[bs, 768]

            avg_text = avg_text.unsqueeze(1).repeat(1, sequence_output_v.size(1), 1)
            bs = sequence_output_v.size(0)
            vis_len = sequence_output_v.size(1)
            sequence_v = torch.cat([sequence_output_v, avg_text], -1).view(sequence_output_v.size(0)*sequence_output_v.size(1), -1)
            sequence_v = self.classifier(sequence_v)
            sequence_v = sequence_v.view(bs, vis_len, -1)
            sim = nn.Softmax(dim=-2)(sequence_v)
            output_dict['scores'] = sim
            return output_dict
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.contiguous().view(-1, self.num_labels)
        output_dict["scores"] = reshaped_logits
        return output_dict


@registry.register_model("visual_bert_mrm")
class VisualBERTMRM(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vocab_map = torch.load('/nas/multi-modal/ziyidou/mmf/object_vocab_bertbaseuncased.pt')
        #self.vocab_map[0] = 0

    @classmethod
    def config_path(cls):
        return "configs/models/visual_bert_mrm/pretrain.yaml"

    def build(self):
        if self.config.training_head_type == "pretraining":
            self.model = VisualBERTMRMForPretraining(self.config)
        elif self.config.training_head_type == "vis":
            self.model = VisualBERTMRMForVisualization(self.config)
        else:
            self.model = VisualBERTMRMForClassification(self.config)

        if self.config.special_visual_initialize:
            self.model.bert.embeddings.initialize_visual_from_pretrained()

        if getattr(self.config, "freeze_base", False):
            for p in self.model.bert.parameters():
                p.requires_grad = False

    def flatten(self, sample_list, to_be_flattened=None, to_be_flattened_dim=None):
        if to_be_flattened is None:
            to_be_flattened = {}
        if to_be_flattened_dim is None:
            to_be_flattened_dim = {}
        for key in to_be_flattened:
            # Make sure these keys are present or otherwise set these keys to None
            sample_list[key] = getattr(sample_list, key, None)
            sample_list[key] = transform_to_batch_sequence(sample_list[key])
        for key in to_be_flattened_dim:
            sample_list[key] = getattr(sample_list, key, None)
            sample_list[key] = transform_to_batch_sequence_dim(sample_list[key])

        if sample_list.visual_embeddings_type is None:
            if sample_list.image_mask is not None:
                sample_list.visual_embeddings_type = torch.zeros_like(
                    sample_list.image_mask
                )

        if sample_list.image_mask is not None:
            attention_mask = torch.cat(
                (sample_list.input_mask, sample_list.image_mask), dim=-1
            )
        else:
            attention_mask = sample_list.input_mask

        sample_list.attention_mask = attention_mask

        return sample_list

    def get_optimizer_parameters(self, config):
        return get_optimizer_parameters_for_bert(self.model, config)

    def flatten_for_bert(self, sample_list, **kwargs):
        to_be_flattened = [
            "input_ids",
            "token_type_ids",
            "input_mask",
            "image_mask",
            "masked_lm_labels",
            "position_embeddings_visual",
            "visual_embeddings_type",
            "image_labels",
        ]
        to_be_flattened_dim = ["image_text_alignment", "visual_embeddings", "image_text_labels"]

        # We want to convert everything into: batch x sequence_length x (dim).
        flattened = self.flatten(sample_list, to_be_flattened, to_be_flattened_dim)
        return flattened

    def update_sample_list_based_on_head(self, sample_list):
        bert_input_ids = sample_list.input_ids
        bert_input_mask = sample_list.input_mask
        bert_input_type_ids = sample_list.segment_ids

        if self.config.training_head_type == "nlvr2":
            bert_input_ids = torch.cat([bert_input_ids, bert_input_ids])
            bert_input_mask = torch.cat([bert_input_mask, bert_input_mask])
            bert_input_type_ids = torch.cat([bert_input_type_ids, bert_input_type_ids])

            # image input
            img0 = getattr(sample_list, "img0", {})
            image_info = getattr(img0, "image_info_0", {})
            image_dim_variable_0 = getattr(image_info, "max_features", None)
            image_feat_variable_0 = getattr(img0, "image_feature_0", None)

            img1 = getattr(sample_list, "img1", {})
            image_info = getattr(img1, "image_info_0", {})
            image_dim_variable_1 = getattr(image_info, "max_features", None)
            image_feat_variable_1 = getattr(img1, "image_feature_0", None)

            image_feat_variable = torch.cat(
                [image_feat_variable_0, image_feat_variable_1]
            )
            image_dim_variable = torch.cat([image_dim_variable_0, image_dim_variable_1])
            image_info = torch.cat([image_info_0['bbox'], image_info_1['bbox']])
        else:
            image_info = getattr(sample_list, "image_info_0", {})
            image_dim_variable = getattr(image_info, "max_features", None)
            image_feat_variable = getattr(sample_list, "image_feature_0", None)
            image_info = image_info['bbox']

        image_labels = getattr(sample_list, "image_labels", None)
        if image_labels is not None:
            image_labels = (image_labels[:, : image_feat_variable.shape[1]])
        

        sample_list.visual_embeddings = image_feat_variable
        sample_list.image_dim = image_dim_variable
        sample_list.input_ids = bert_input_ids
        sample_list.input_mask = bert_input_mask
        sample_list.token_type_ids = bert_input_type_ids
        sample_list.image_labels = image_labels
        sample_list.image_info = image_info
        return sample_list

    def add_custom_params(self, sample_list):
        visual_embeddings = getattr(sample_list, "visual_embeddings", None)
        image_dim = getattr(sample_list, "image_dim", None)
        # pretraining labels
        sample_list.masked_lm_labels = getattr(sample_list, "lm_label_ids", None)
        # image_feat_variable = batch x ( num_choice x ) image_feature_length x dim
        # Prepare Mask
        if visual_embeddings is not None and image_dim is not None:
            image_mask = torch.arange(
                visual_embeddings.size(-2), device=visual_embeddings.device
            ).expand(*visual_embeddings.size()[:-1])
            if len(image_dim.size()) < len(image_mask.size()):
                image_dim = image_dim.unsqueeze(-1)
                assert len(image_dim.size()) == len(image_mask.size())
            image_mask = image_mask < image_dim
            sample_list.image_mask = image_mask.long()
        else:
            sample_list.image_mask = None

        sample_list.position_embeddings_visual = None

        return sample_list

    # Backward compatibility for code from original VisualBERT
    @classmethod
    def format_state_key(cls, key):
        return (
            key.replace("bert.bert", "model.bert")
            .replace("bert.cls", "model.cls")
            .replace("bert.classifier", "model.classifier")
        )

    def forward(self, sample_list):
        sample_list = self.update_sample_list_based_on_head(sample_list)
        sample_list = self.add_custom_params(sample_list)
        sample_list = self.flatten_for_bert(sample_list)
        if getattr(sample_list, "cls_prob", None) is not None:
            sample_list.image_text_label = torch.matmul(sample_list.cls_prob, self.vocab_map.to(sample_list.cls_prob.device))
        else:
            sample_list.image_text_label = None
        images =  getattr(sample_list, "images", None)
        image =  getattr(sample_list, "image", None)
        regress_mask =  getattr(sample_list, "regress_mask", None)
        cls_prob_mask = getattr(sample_list, 'cls_prob_mask', None)
        is_correct = getattr(sample_list, 'is_correct', None)
        targets = getattr(sample_list, 'targets', None)
        image_id = getattr(sample_list, 'image_id', None)
        start_ind =  getattr(sample_list, "start_ind", None)
        end_ind =  getattr(sample_list, "end_ind", None)

        output_dict = self.model(
            sample_list.input_ids,
            sample_list.input_mask,
            sample_list.attention_mask,
            sample_list.token_type_ids,
            sample_list.visual_embeddings,
            sample_list.position_embeddings_visual,
            sample_list.visual_embeddings_type,
            sample_list.image_text_alignment,
            sample_list.masked_lm_labels,
            sample_list.image_labels,
            sample_list.image_info,
            sample_list.image_mask,
            sample_list.image_text_label,
            (images, image),
            regress_mask,
            cls_prob_mask,
            is_correct,
            targets,
            image_id,
            start_ind,
            end_ind,
        )

        if "pretraining" in self.config.training_head_type:
            loss_key = "{}/{}".format(
                sample_list.dataset_name, sample_list.dataset_type
            )
            output_dict["losses"] = {}
            if 'masked_lm_loss' in output_dict:
                output_dict["losses"][loss_key + "/masked_lm_loss"] = output_dict.pop(
                    "masked_lm_loss"
                )
            if 'masked_region_loss' in output_dict:
                output_dict["losses"][loss_key + "/masked_region_loss"] = output_dict.pop(
                    "masked_region_loss"
                )
            if 'global_loss' in output_dict:
                output_dict["losses"][loss_key + "/global_loss"] = output_dict.pop(
                    "global_loss"
                )
            if 'sim_loss' in output_dict:
                output_dict["losses"][loss_key + "/sim_loss"] = output_dict.pop(
                    'sim_loss'
                ) 

        return output_dict
