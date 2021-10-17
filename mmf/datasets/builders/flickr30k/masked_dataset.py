# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import numpy as np

from mmf.common.sample import Sample
from mmf.datasets.mmf_dataset import MMFDataset

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

class MaskedFlickr30kDataset(MMFDataset):
    def __init__(self, config, *args, dataset_name="masked_flickr30k", **kwargs):
        super().__init__(dataset_name, config, *args, **kwargs)

        # +1 because old codebase adds a global feature while reading the features
        # from the lmdb. If we set config.max_features to + 1 value already in yaml,
        # this codebase adds zero padding. To work around that, setting #features to
        # max_region_num - 1, so this code doesnt add zero padding and then add global
        # feature and increment _max_region_num by one like below.
        self._max_region_num = self.config.max_features #+ 1

    def __getitem__(self, idx):

        sample_info = self.annotation_db[idx]
        #features_data = self.features_db[idx]

        #print(sample_info.keys())
        #dict_keys(['caption', 'sent_id', 'image_id', 'refBox'])
        try:
            image_id = sample_info['img_path']
        except:
            image_id = sample_info['image_id']
        current_sample = Sample()
        current_sample.image_id = torch.tensor(int(str(image_id).split('.')[0]))
        image_id = str(image_id).split('.')[0] + '.npy'
        
        features_data = self.features_db.from_path(image_id)

        import random
        #if random.random() < 0.5:
        #    is_correct = -1
        #else:
        #    is_correct = 1
        is_correct = -1

        if 'refBox' in sample_info:
            ref_box = sample_info['refBox']

        # image_info_0 contains roi features computed on a given image.
        if 'boxes' in features_data["image_info_0"]:
            features_data["image_info_0"]['bbox'] = features_data["image_info_0"]['boxes']
            features_data["image_info_0"]['image_width'] = features_data["image_info_0"]['image_w']
            features_data["image_info_0"]['image_height'] = features_data["image_info_0"]['image_h']
        boxes = torch.from_numpy(features_data["image_info_0"]["bbox"])
        features = features_data["image_feature_0"]
        #dict_keys(['image_id', 'image_height', 'image_width', 'num_boxes', 'objects', 'cls_prob', 'bbox', 'max_features'])

        num_boxes = features.size()[0]
        #print(num_boxes)
        #print(len(features_data["image_info_0"]["bbox"]))
        num_boxes = (len(features_data["image_info_0"]["bbox"]))
        #1/0
        #global_feature = torch.sum(features, 0) / num_boxes
        #global_feature = global_feature.unsqueeze(0)
        image_w = features_data["image_info_0"]["image_width"]
        image_h = features_data["image_info_0"]["image_height"]
        cls_prob_mask = torch.zeros((self.config.max_region_num, 1)).float()
        #print(features_data['image_info_0']['cls_prob'])
        #print(features_data['image_info_0']['objects'])
        try:
            
            try:
                num_reg = min(torch.tensor(features_data['image_info_0']['cls_prob']).size(0), self.config.max_features)
                #IndexError: too many indices for tensor of dimension 2
                objects = torch.tensor(features_data['image_info_0']['cls_prob']).argmax(1) +1
                cls_prob = torch.zeros((self.config.max_features, 1601))
                
                ind = torch.arange(num_reg)
                cls_prob[ind, objects[:num_reg]] = 1
                current_sample.cls_prob = cls_prob #torch.tensor(features['image_info_0']['cls_prob']) #torch.matmul(torch.tensor(features['image_info_0']['cls_prob']), self.vocab_map)
                cls_prob_mask[:num_reg, 0] = (torch.tensor(features_data['image_info_0']['cls_prob']).max(1)[0] > 0.1)[:num_reg]
                current_sample.cls_prob_mask = cls_prob_mask #(torch.tensor(features_data['image_info_0']['cls_prob']).max(1)[0] > 0.1)
                #current_sample.vocab_map = self.vocab_map
            
            except:
                objects = torch.tensor(features_data['image_info_0']['objects']) +1#torch.matmul(torch.tensor(features['image_info_0']['cls_prob']), self.vocab_map)
                aa = min(objects.size(0), self.config.max_features)
                #cls_prob = torch.zeros((aa, 1601))
                cls_prob = torch.zeros((self.config.max_features, 1601))
                #print(cls_prob.size())
                ind = torch.arange(aa)
                cls_prob[ind, objects[:aa]] = 1
                #cls_prob = cls_prob[1:, :]
                #cls_prob = cls_prob[:aa-1, :]
                current_sample.cls_prob = cls_prob #torch.tensor(features_data['image_info_0']['objects']) #torch.matmul(torch.tensor(features['image_info_0']['cls_prob']), self.vocab_map)'''
        except:
            1/0
            #current_sample.cls_prob = None

        all_boxes = boxes
        all_features = features
        total_num_boxes = min(
            int(num_boxes), self.config.max_region_num
        )

        if 'refBox' in sample_info:
            target = torch.zeros((self.config.max_region_num, 1)).float()
            targets = iou(
                torch.tensor(all_boxes[:, :4]).float(), torch.tensor([ref_box]).float()
            )
            targets[targets < 0.5] = 0
            target[:total_num_boxes] = targets[:total_num_boxes]
            current_sample.targets = target

        total_features = self.config.max_features
        #targets = target #s[total_features:]
        all_boxes_pad = torch.zeros((self.config.max_region_num, 4))
        if hasattr(self, "transformer_bbox_processor"):
            features_data["image_info_0"] = self.transformer_bbox_processor(
                features_data["image_info_0"]
            )
            boxes = (features_data["image_info_0"]["bbox"])
            all_boxes = boxes
            all_boxes_pad = torch.zeros((self.config.max_region_num, 5))
        all_features_pad = torch.zeros(
            (self.config.max_region_num, 2048), dtype=torch.float32
        )

        #all_boxes_pad[:total_num_boxes] = all_boxes[:-1]
        #all_features_pad[:total_num_boxes] = all_features[:-1]
        #all_boxes_pad[:total_num_boxes] = all_boxes[1:]
        #all_features_pad[:total_num_boxes] = all_features[1:]
        all_boxes_pad[:total_num_boxes] = all_boxes[:total_num_boxes]
        all_features_pad[:total_num_boxes] = all_features[:total_num_boxes]
        cls_prob_mask[:total_num_boxes] = 1
        current_sample.cls_prob_mask = cls_prob_mask.bool() #(torch.tensor(features_data['image_info_0']['cls_prob']).max(1)[0] > 0.1)

        current_sample.image_feature_0 = all_features_pad
        current_sample.image_info_0 = {}
        current_sample.image_info_0["image_width"] = features_data["image_info_0"][
            "image_width"
        ]
        current_sample.image_info_0["image_height"] = features_data["image_info_0"][
            "image_height"
        ]
        current_sample.image_info_0["bbox"] = all_boxes_pad
        current_sample.image_info_0["max_features"] = torch.tensor(total_num_boxes)
        current_sample.image_info_0["max_image_features"] = num_boxes
        #current_sample.targets = target

        current_sample.update(
            {
                "image_labels": self.masked_region_processor(
                    all_features_pad, (is_correct == -1), total_num_boxes
                )
            }
        )

        import random
        if 'sentences' in sample_info:
            rand_id = random.randint(0, len(sample_info["sentences"])-1)
            selected_caption = sample_info["sentences"][rand_id]
            if is_correct != -1:
                if random.random() < 0.5:
                    selected_caption = self._get_mismatching_caption(image_id)
                    is_correct = 0
                else:
                    is_correct = 1
        elif 'sentence' in sample_info:
            selected_caption = sample_info["sentence"]
        else:
            selected_caption = sample_info['caption']

        processed = self.masked_token_processor(
            {
                "text_a": selected_caption,
                "text_b": None,
                "is_correct": is_correct,
            }, None if is_correct == -1 else 0.0
        )
        processed.pop("tokens")
        current_sample.update(processed)

        return current_sample

    def _get_mismatching_caption(self, image_id):
        import random
        other_item = self.annotation_db[random.randint(0, len(self.annotation_db) - 1)]

        other_id = str(other_item['img_path']).split('.')[0]+'.npy'
        while other_id == image_id:
            other_item = self.annotation_db[
                random.randint(0, len(self.annotation_db) - 1)
            ]
            other_id = str(other_item['img_path']).split('.')[0]+'.npy'

        other_caption = other_item["sentences"][
            random.randint(0, len(other_item["sentences"]) - 1)
        ]
        return other_caption
