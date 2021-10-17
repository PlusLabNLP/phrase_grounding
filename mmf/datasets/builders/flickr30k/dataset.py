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


class Flickr30kDataset(MMFDataset):
    def __init__(self, config, *args, dataset_name="flickr30k", **kwargs):
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

        image_id = sample_info['image_id']
        image_id = str(image_id) + '.npy'
        features_data = self.features_db.from_path(image_id)

        ref_box = sample_info['refBox']



        if 'bbox' in features_data["image_info_0"]:
            features_data["image_info_0"]['boxes'] = features_data["image_info_0"]['bbox']
            features_data["image_info_0"]['image_w'] = features_data["image_info_0"]['image_width']
            features_data["image_info_0"]['image_h'] = features_data["image_info_0"]['image_height'] #'''
            # image_info_0 contains roi features computed on a given image.
        if 'image_info_1' in features_data and 'bbox' in features_data["image_info_1"]:
            features_data["image_info_1"]['boxes'] = features_data["image_info_1"]['bbox']
            features_data["image_info_1"]['image_w'] = features_data["image_info_1"]['image_width']
            features_data["image_info_1"]['image_h'] = features_data["image_info_1"]['image_height'] #'''
        if 'image_info_2' in features_data and 'bbox' in features_data["image_info_2"]:
            features_data["image_info_2"]['boxes'] = features_data["image_info_2"]['bbox']
            features_data["image_info_2"]['image_w'] = features_data["image_info_2"]['image_width']
            features_data["image_info_2"]['image_h'] = features_data["image_info_2"]['image_height'] #'''
        boxes = torch.from_numpy(features_data["image_info_0"]["boxes"])
        #print(sample_info.keys())
        #print(boxes)
        #boxes = (torch.tensor(sample_info['bboxes']).float())
        #features_data["image_info_0"]["boxes"] = boxes.numpy()
        #1/0
        features = features_data["image_feature_0"]

        num_boxes = len(boxes) #features.size()[0]
        #num_boxes = self.config.max_features  #len(boxes) #features.size()[0]
        max_features = self.config.max_features
        max_region_num = self.config.max_region_num 
        #max_features = num_boxes
        #max_region_num = num_boxes

        features = features[:num_boxes]

        #global_feature = torch.sum(features, 0) / num_boxes
        #global_feature = global_feature.unsqueeze(0)
        image_w = features_data["image_info_0"]["image_w"]
        image_h = features_data["image_info_0"]["image_h"]

        '''global_bounding_box = torch.tensor(
            [[0, 0, image_w, image_h]], dtype=torch.float
        )

        features = torch.cat((global_feature, features), 0)
        boxes = torch.cat((global_bounding_box, boxes), 0)
        num_boxes = num_boxes + 1'''

        # image_info_1 contains roi features computed on ground truth boxes the
        # expression was pointing towards. Final input is contanenation of the two
        # and the classifier votes for each of the four options using independent logits.
        #if True:
        gt_key = 'image_info_2'
        if gt_key in features_data:
            gt_num_boxes = features_data[gt_key]["num_boxes"]
            gt_boxes = torch.from_numpy(
                features_data[gt_key]["boxes"][:gt_num_boxes, :]
            )
            gt_features = features_data["image_feature_"+gt_key[-1]][:gt_num_boxes, :]

            all_boxes = torch.cat((boxes, gt_boxes), 0)
            all_features = torch.cat((features, gt_features), 0)
            total_num_boxes = min(
                int(num_boxes) + int(gt_num_boxes), max_region_num
            )
        else:
            gt_num_boxes = None
            all_boxes = boxes
            all_features = features
            total_num_boxes = min(
                int(num_boxes), max_region_num
            )

        # given the mix boxes, and ref_box, calculate the overlap.
        targets = iou(
            (all_boxes[:, :4]).float(), torch.tensor([ref_box]).float()
        )
        #targets = iou(
        #    torch.tensor(all_boxes[:, :4]).float(), torch.tensor([ref_box]).float()
        #)
        targets[targets < 0.5] = 0
        #targets[targets < 0.5] = 0
        total_features = max_features #+ 1
        #targets = targets[multiple_choice_idx].squeeze()

        target = torch.zeros((self.config.max_region_num, 1)).float()
        cls_prob_mask = torch.zeros((max_region_num, 1)).float()
        #print(2)
        #print(total_num_boxes)
        #print(target.size())
        #1/0
        #print(all_boxes.size())
    
        target[:total_num_boxes] = targets[:total_num_boxes]
        targets = target #s[total_features:]
        all_boxes_pad = torch.zeros((max_region_num, 4))
        if hasattr(self, "transformer_bbox_processor"):
            features_data["image_info_0"] = self.transformer_bbox_processor(
                features_data["image_info_0"]
            )
            if 'image_info_1' in features_data:
                features_data["image_info_1"] = self.transformer_bbox_processor(
                    features_data["image_info_1"]
                )
            '''global_bounding_box = torch.tensor(
                [[0, 0, 1, 1, 1]], dtype=torch.float
            )'''
            boxes = (features_data["image_info_0"]["bbox"])
            if gt_num_boxes is not None:
                gt_boxes = (
                    features_data["image_info_1"]["bbox"][:gt_num_boxes, :]
                )
                all_boxes = torch.cat((boxes, gt_boxes), 0)
            else:
                all_boxes = boxes
            all_boxes_pad = torch.zeros((max_region_num, 5))
        all_features_pad = torch.zeros(
            (max_region_num, 2048), dtype=torch.float32
        )

        all_boxes_pad[:total_num_boxes] = all_boxes[:total_num_boxes]
        all_features_pad[:total_num_boxes] = all_features[:total_num_boxes]
        cls_prob_mask[:total_num_boxes] = 1

        current_sample = Sample()
        if 'phrase' in sample_info:
            text_processor_argument = {"text": sample_info["sentence"]}
            processed_question = self.text_processor(text_processor_argument)
            words = sample_info['sentence'].split()
            word_lens = [len(self.text_processor._tokenizer.tokenize(w)) for w in words]
            word_ind = [0]
            for w_len in word_lens:
                word_ind.append(word_ind[-1]+w_len)
            start_ind = word_ind[sample_info['first_word_index']]
            end_ind = word_ind[sample_info['first_word_index'] + len(sample_info['phrase'].split())]
            current_sample.start_ind = start_ind
            current_sample.end_ind = end_ind
            
        else:
            text_processor_argument = {"text": sample_info["caption"]}
            processed_question = self.text_processor(text_processor_argument)

        try:
            cls_prob_mask[:total_num_boxes] = (torch.tensor(features_data['image_info_0']['cls_prob']).max(1)[0] > 0.0)[:total_num_boxes]
            current_sample.cls_prob_mask = cls_prob_mask.bool() #(torch.tensor(features_data['image_info_0']['cls_prob']).max(1)[0] > 0.1)[:total_num_boxes]
        except:
            current_sample.cls_prob_mask = cls_prob_mask.bool() #(torch.tensor(features_data['image_info_0']['cls_prob']).max(1)[0] > 0.2)
            pass
            #current_sample.cls_prob_mask = None
        #objects1 = torch.tensor(features_data['image_info_0']['cls_prob']).argmax(1)
        if 'objects' in features_data['image_info_0']:
            objects = torch.tensor(features_data['image_info_0']['objects'])+1 #torch.matmul(torch.tensor(features['image_info_0']['cls_prob']), self.vocab_map)
            aa = min(objects.size(0), max_region_num)
            cls_prob = torch.zeros((max_region_num, 1601))
            #print(cls_prob.size())
            ind = torch.arange(aa)
            cls_prob[ind, objects[:aa]] = 1
            #cls_prob = cls_prob[1:, :]
            #cls_prob = cls_prob[:aa-1, :]
            current_sample.cls_prob = cls_prob #[:total_num_boxes] #torch.tensor(features_data['image_info_0']['objects']) #torch.matmul(torch.tensor(features['image_info_0']['cls_prob']), self.vocab_map)'''
        #print(1)
        #print(objects1)
        #print(objects2)
        #1/0
        current_sample.image_feature_0 = all_features_pad
        current_sample.update(processed_question)
        current_sample.image_info_0 = {}
        current_sample.image_info_0["image_width"] = features_data["image_info_0"][
            "image_w"
        ]
        current_sample.image_info_0["image_height"] = features_data["image_info_0"][
            "image_h"
        ]
        current_sample.image_info_0["bbox"] = all_boxes_pad
        current_sample.image_info_0["max_features"] = torch.tensor(total_num_boxes)
        current_sample.image_info_0["max_gt_features"] = gt_num_boxes
        current_sample.image_info_0["max_image_features"] = num_boxes
        current_sample.targets = target

        return current_sample
