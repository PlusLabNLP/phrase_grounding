# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import numpy as np
import json_lines

from mmf.common.sample import Sample
from mmf.datasets.mmf_dataset import MMFDataset


def _load_annotationsQ_A(annotations_jsonpath):
    """Build an index out of FOIL annotations, mapping each image ID with its corresponding captions."""
    entries = []
    with open(annotations_jsonpath, "rb") as f:  # opening file in binary(rb) mode
        for annotation in json_lines.reader(f):
            # metadata_fn = json.load(open(os.path.join('data/VCR/vcr1images', annotation["metadata_fn"]), 'r'))
            # det_names = metadata_fn["names"]
            det_names = ""
            question = annotation["question"]
            try:
                ans_label = annotation["answer_label"]
            except:
                ans_label = 0
                
            img_id = (annotation["img_id"])
            img_fn = annotation["img_fn"]
            anno_id = int(annotation["annot_id"].split("-")[1])
            entries.append(
                {
                    "question": question,
                    "img_fn": img_fn,
                    "answers": annotation["answer_choices"],
                    "metadata_fn": annotation["metadata_fn"],
                    "target": ans_label,
                    "img_id": img_id,
                    "anno_id": anno_id,
                }
            )

    return entries


def _load_annotationsQA_R(annotations_jsonpath):
    """Build an index out of FOIL annotations, mapping each image ID with its corresponding captions."""
    entries = []
    with open(annotations_jsonpath, "rb") as f:  # opening file in binary(rb) mode
        for annotation in json_lines.reader(f):
            # metadata_fn = json.load(open(os.path.join('data/VCR/vcr1images', annotation["metadata_fn"]), 'r'))
            #if split == "test":
            if False:
                # for each answer
                for answer in annotation["answer_choices"]:
                    question = annotation["question"] + ["[SEP]"] + answer
                    img_id = _converId(annotation["img_id"])
                    ans_label = 0
                    img_fn = annotation["img_fn"]
                    anno_id = int(annotation["annot_id"].split("-")[1])
                    entries.append(
                        {
                            "question": question,
                            "img_fn": img_fn,
                            "answers": annotation["rationale_choices"],
                            "metadata_fn": annotation["metadata_fn"],
                            "target": ans_label,
                            "img_id": img_id,
                        }
                    )
            else:
                det_names = ""
                question = (
                    annotation["question"]
                    + ["[SEP]"]
                    + annotation["answer_choices"][annotation["answer_label"]]
                )
                ans_label = annotation["rationale_label"]
                # img_fn = annotation["img_fn"]
                img_id = (annotation["img_id"])
                img_fn = annotation["img_fn"]

                anno_id = int(annotation["annot_id"].split("-")[1])
                entries.append(
                    {
                        "question": question,
                        "img_fn": img_fn,
                        "answers": annotation["rationale_choices"],
                        "metadata_fn": annotation["metadata_fn"],
                        "target": ans_label,
                        "img_id": img_id,
                        "anno_id": anno_id,
                    }
                )

    return entries


class VCRDataset(MMFDataset):
    def __init__(self, config, *args, dataset_name="vcr", **kwargs):
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
        image_id = str(image_id) #+ '.npy'
        features_data = self.features_db.from_path(image_id)


        ref_box = sample_info['refBox']

        # image_info_0 contains roi features computed on a given image.
        boxes = torch.from_numpy(features_data["image_info_0"]["boxes"])
        features = features_data["image_feature_0"]

        num_boxes = features.size()[0]
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
        if True:
            gt_num_boxes = features_data["image_info_1"]["num_boxes"]
            gt_boxes = torch.from_numpy(
                features_data["image_info_1"]["boxes"][:gt_num_boxes, :]
            )
            gt_features = features_data["image_feature_1"][:gt_num_boxes, :]

            all_boxes = torch.cat((boxes, gt_boxes), 0)
            all_features = torch.cat((features, gt_features), 0)
            total_num_boxes = min(
                int(num_boxes + int(gt_num_boxes)), self.config.max_region_num
            )
        else:
            all_boxes = boxes
            all_features = features
            total_num_boxes = min(
                int(num_boxes), self.config.max_region_num
            )

        # given the mix boxes, and ref_box, calculate the overlap.
        targets = iou(
            torch.tensor(all_boxes[:, :4]).float(), torch.tensor([ref_box]).float()
        )
        targets[targets < 0.5] = 0
        total_features = self.config.max_features #+ 1
        #targets = targets[multiple_choice_idx].squeeze()

        target = torch.zeros((self.config.max_region_num, 1)).float()
        #print(2)
        #print(total_num_boxes)
        #print(target.size())
        #1/0
        #print(all_boxes.size())
    
        target[:total_num_boxes] = targets
        targets = target #s[total_features:]
        all_boxes_pad = torch.zeros((self.config.max_region_num, 4))
        if hasattr(self, "transformer_bbox_processor"):
            features_data["image_info_0"] = self.transformer_bbox_processor(
                features_data["image_info_0"]
            )
            features_data["image_info_1"] = self.transformer_bbox_processor(
                features_data["image_info_1"]
            )
            '''global_bounding_box = torch.tensor(
                [[0, 0, 1, 1, 1]], dtype=torch.float
            )'''
            boxes = (features_data["image_info_0"]["bbox"])
            gt_boxes = (
                features_data["image_info_1"]["bbox"][:gt_num_boxes, :]
            )
            all_boxes = torch.cat((boxes, gt_boxes), 0)
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
