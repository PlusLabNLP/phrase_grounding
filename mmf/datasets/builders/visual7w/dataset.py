# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from mmf.common.sample import Sample
from mmf.datasets.mmf_dataset import MMFDataset
from torchvision.ops import box_iou


def iou(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_area = (
        (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
    ).view(1, K)

    anchors_area = (
        (anchors[:, 2] - anchors[:, 0] + 1) * (anchors[:, 3] - anchors[:, 1] + 1)
    ).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (
        torch.min(boxes[:, :, 2], query_boxes[:, :, 2])
        - torch.max(boxes[:, :, 0], query_boxes[:, :, 0])
        + 1
    )
    iw[iw < 0] = 0

    ih = (
        torch.min(boxes[:, :, 3], query_boxes[:, :, 3])
        - torch.max(boxes[:, :, 1], query_boxes[:, :, 1])
        + 1
    )
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps


class Visual7WDataset(MMFDataset):
    def __init__(self, config, *args, dataset_name="visual7w", **kwargs):
        super().__init__(dataset_name, config, *args, **kwargs)

    def __getitem__(self, idx):

        sample_info = self.annotation_db[idx]
        features_data = self.features_db[idx]
        multiple_choice_idx = torch.tensor(sample_info["mc_idx"])

        # image_info_0 contains roi features computed on a given image.
        boxes = torch.from_numpy(features_data["image_info_0"]["boxes"])
        features_data["image_info_0"]['image_width'] = features_data["image_info_0"]["image_w"]
        features_data["image_info_0"]['image_height'] = features_data["image_info_0"]["image_h"]
        features_data["image_info_0"]['bbox'] = features_data["image_info_0"]["boxes"]
        features_data["image_info_1"]['image_width'] = features_data["image_info_1"]["image_w"]
        features_data["image_info_1"]['image_height'] = features_data["image_info_1"]["image_h"]
        features_data["image_info_1"]['bbox'] = features_data["image_info_1"]["boxes"]
        features = features_data["image_feature_0"]

        num_boxes = features.size()[0]
        global_feature = torch.sum(features, 0) / num_boxes
        global_feature = global_feature.unsqueeze(0)
        image_w = features_data["image_info_0"]["image_width"]
        image_h = features_data["image_info_0"]["image_height"]

        global_bounding_box = torch.tensor(
            [[0, 0, image_w, image_h]], dtype=torch.float
        )

        features = torch.cat((global_feature, features), 0)
        boxes = torch.cat((global_bounding_box, boxes), 0)
        num_boxes = num_boxes + 1

        # image_info_1 contains roi features computed on ground truth boxes the
        # expression was pointing towards. Final input is contanenation of the two
        # and the classifier votes for each of the four options using independent logits.
        gt_num_boxes = features_data["image_info_1"]["num_boxes"]
        gt_boxes = torch.from_numpy(
            features_data["image_info_1"]["boxes"][:gt_num_boxes, :]
        )
        gt_features = features_data["image_feature_1"][:gt_num_boxes, :]

        #gt_num_boxes = gt_num_boxes-1
        #gt_boxes = gt_boxes[1:]
        #gt_features = gt_features[1:]

        '''print(1)
        print(boxes.size())
        print(gt_boxes.size())
        print(gt_boxes[0])
        print(multiple_choice_idx)
        1/0'''

        all_boxes = torch.cat((boxes, gt_boxes), 0)
        all_features = torch.cat((features, gt_features), 0)
        total_num_boxes = min(
            int(num_boxes + int(gt_num_boxes)), self.config.max_region_num
        )

        ref_box = sample_info["refBox"]
        # given the mix boxes, and ref_box, calculate the overlap.
        #targets = box_iou(
        targets = iou(
            torch.tensor(all_boxes[:, :4]).float(), torch.tensor([ref_box]).float()
        )
        targets[targets < 0.5] = 0
        #targets[targets >= 0.5] = 1
        total_features = self.config.max_features + 1
        targets = targets[total_features:]
        targets = targets[multiple_choice_idx].squeeze()
        all_boxes_pad = torch.zeros((self.config.max_region_num, 4))
        if hasattr(self, "transformer_bbox_processor"):
            features_data["image_info_0"] = self.transformer_bbox_processor(
                features_data["image_info_0"]
            )
            features_data["image_info_1"] = self.transformer_bbox_processor(
                features_data["image_info_1"]
            )
            global_bounding_box = torch.tensor(
                [[0, 0, 1, 1, 1]], dtype=torch.float
            )
            boxes = (features_data["image_info_0"]["bbox"])
            gt_boxes = (
                features_data["image_info_1"]["bbox"][:gt_num_boxes, :]
            )
            all_boxes = torch.cat((global_bounding_box, boxes, gt_boxes), 0)
            all_boxes_pad = torch.zeros((self.config.max_region_num, 5))
        all_features_pad = torch.zeros(
            (self.config.max_region_num, 2048), dtype=torch.float32
        )

        all_boxes_pad[:total_num_boxes] = all_boxes[:total_num_boxes]
        all_features_pad[:total_num_boxes] = all_features[:total_num_boxes]

        text_processor_argument = {"text": sample_info["caption"]}
        processed_question = self.text_processor(text_processor_argument)

        current_sample = Sample()
        current_sample.image_feature_0 = all_features_pad
        current_sample.update(processed_question)
        current_sample.image_info_0 = {}
        current_sample.image_info_0["image_width"] = features_data["image_info_0"][
            "image_width"
        ]
        current_sample.image_info_0["image_height"] = features_data["image_info_0"][
            "image_height"
        ]
        current_sample.image_info_0["bbox"] = all_boxes_pad
        current_sample.image_info_0["max_features"] = torch.tensor(total_num_boxes)
        current_sample.image_info_0["max_gt_features"] = gt_num_boxes
        current_sample.image_info_0["max_image_features"] = num_boxes
        current_sample.image_info_0["multiple_choice_idx"] = multiple_choice_idx
        current_sample.targets = targets #.long().contiguous().view(-1)

        return current_sample
