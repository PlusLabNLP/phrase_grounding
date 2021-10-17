import torch
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

class RefCOCODataset(MMFDataset):
    def __init__(self, config, *args, dataset_name="refcoco+", **kwargs):
        super().__init__(dataset_name, config, *args, **kwargs)


    def __getitem__(self, idx):

        sample_info = self.annotation_db[idx]
        features_data = self.features_db[idx]

        image_id = sample_info['image_id']
        ref_box = sample_info['refBox']
        ref_box = [
            ref_box[0],
            ref_box[1],
            ref_box[0]+ref_box[2],
            ref_box[1]+ref_box[3],
        ]

        if 'bbox' in features_data["image_info_0"]:
            features_data["image_info_0"]['boxes'] = features_data["image_info_0"]['bbox']
            features_data["image_info_0"]['image_w'] = features_data["image_info_0"]['image_width']
            features_data["image_info_0"]['image_h'] = features_data["image_info_0"]['image_height'] #'''
        if "image_info_1" in features_data and 'bbox' in features_data["image_info_1"]:
            features_data["image_info_1"]['boxes'] = features_data["image_info_1"]['bbox']
            features_data["image_info_1"]['image_w'] = features_data["image_info_1"]['image_width']
            features_data["image_info_1"]['image_h'] = features_data["image_info_1"]['image_height'] #'''

        # image_info_0 contains roi features computed on a given image.
        boxes = torch.from_numpy(features_data["image_info_0"]["boxes"])
        features = features_data["image_feature_0"]

        num_boxes = (len(features_data["image_info_0"]["boxes"])) #features.size()[0]
        image_w = features_data["image_info_0"]["image_w"]
        image_h = features_data["image_info_0"]["image_h"]


        # image_info_1 contains roi features computed on ground truth boxes the
        # expression was pointing towards. Final input is contanenation of the two
        # and the classifier votes for each of the four options using independent logits.
        #if False: #'image_info_1' in features_data:
        if 'image_info_1' in features_data:
            #gt_num_boxes = features_data["image_info_1"]["num_boxes"]
            gt_num_boxes = (len(features_data["image_info_1"]["boxes"])) #features.size()[0]
            gt_boxes = torch.from_numpy(
                features_data["image_info_1"]["boxes"][:gt_num_boxes, :]
            )
            gt_features = features_data["image_feature_1"][:gt_num_boxes, :]
            gt_targets = iou(
                (gt_boxes[:, :4]).float(), torch.tensor([ref_box]).float()
            )
            gt_ind = (gt_targets>0).squeeze(1)

            gt_num_boxes2= gt_ind.sum()
            if gt_num_boxes2 > 0:
                try:
                    all_boxes = torch.cat((boxes[:-gt_num_boxes2], gt_boxes[gt_ind]), 0)
                    all_features = torch.cat((features[:-gt_num_boxes2], gt_features[gt_ind]), 0)
                    total_num_boxes = min(
                        int(num_boxes) , self.config.max_region_num
                    )
                except:
                    gt_num_boxes2 = 0
                    all_boxes = boxes
                    all_features = features
                    total_num_boxes = min(
                        int(num_boxes), self.config.max_region_num
                    )
            else:
                all_boxes = boxes
                all_features = features
                total_num_boxes = min(
                    int(num_boxes), self.config.max_region_num
                )
            '''all_boxes = torch.cat((boxes, gt_boxes), 0)
            all_features = torch.cat((features, gt_features), 0)
            total_num_boxes = min(
                int(num_boxes + int(gt_num_boxes)), self.config.max_region_num
            )#'''
        else:
            gt_num_boxes = None
            all_boxes = boxes
            all_features = features
            total_num_boxes = min(
                int(num_boxes), self.config.max_region_num
            )

        # given the mix boxes, and ref_box, calculate the overlap.
        targets = iou(
            torch.tensor(all_boxes[:, :4]).float(), torch.tensor([ref_box]).float()
        )
        total_features = self.config.max_features
        #targets = targets[multiple_choice_idx].squeeze()

        target = torch.zeros((self.config.max_region_num, 1)).float()
        #print(2)
        #print(total_num_boxes)
        #print(all_boxes.size())
        #targets = targets[total_features:]
        all_boxes_pad = torch.zeros((self.config.max_region_num, 4))
        if hasattr(self, "transformer_bbox_processor"):
            features_data["image_info_0"] = self.transformer_bbox_processor(
                features_data["image_info_0"]
            )
            #all_boxes_pad = torch.zeros((self.config.max_region_num, 5))
            #all_boxes =  (features_data["image_info_0"]["bbox"])
            boxes =  (features_data["image_info_0"]["bbox"])
            if gt_num_boxes is not None:
                if gt_num_boxes2 > 0:
                    features_data["image_info_1"] = self.transformer_bbox_processor(
                        features_data["image_info_1"]
                    )
                    gt_boxes = (
                        features_data["image_info_1"]["bbox"][:gt_num_boxes, :]
                    )
                    try:
                        all_boxes = torch.cat((boxes[:-gt_num_boxes2], gt_boxes[gt_ind]), 0)
                    except:
                        all_boxes = torch.cat((boxes[:1], gt_boxes[gt_ind]), 0)
                else:
                    all_boxes = boxes
            else:
                all_boxes = boxes
            all_boxes_pad = torch.zeros((self.config.max_region_num, 5))
        all_features_pad = torch.zeros(
            (self.config.max_region_num, 2048), dtype=torch.float32
        )

        cls_prob_mask = torch.zeros((self.config.max_region_num, 1)).float()
        #if "image_info_1" in features_data:
        #if False: #"image_info_1" in features_data:
        if False: #"image_info_1" in features_data:
            all_boxes = all_boxes[:total_num_boxes]
            all_features = all_features[:total_num_boxes]
            targets = targets[:total_num_boxes]
            perm = torch.randperm(total_num_boxes)
            all_boxes = all_boxes[perm]
            all_features = all_features[perm]
            targets = targets[perm]
        all_boxes_pad[:total_num_boxes] = all_boxes[:total_num_boxes]
        all_features_pad[:total_num_boxes] = all_features[:total_num_boxes]
        cls_prob_mask[:total_num_boxes] = 1
        #cls_prob_mask[:total_num_boxes] = (torch.tensor(features_data['image_info_0']['cls_prob']).max(1)[0] > 0.1)[:total_num_boxes]
        target[:total_num_boxes] = targets[:total_num_boxes]
        #for i, t in enumerate(targets):
        #    if t == 0.0:
        #        cls_prob_mask[i] = 0
        targets[targets < 0.5] = 0
        target[target < 0.5] = 0

        text_processor_argument = {"text": sample_info["caption"]}
        processed_question = self.text_processor(text_processor_argument)
        #start_ind = 0 #word_ind[sample_info['first_word_index']]
        #end_ind = len(self.text_processor._tokenizer.tokenize(sample_info['caption']))

        current_sample = Sample()
        #current_sample.start_ind = start_ind
        #current_sample.end_ind = end_ind
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
        current_sample.image_info_0["max_image_features"] = num_boxes
        current_sample.targets = target
        #print(features_data['image_info_0'].keys())
        #1/0
        #cls_prob_mask[:total_num_boxes] = (torch.tensor(features_data['image_info_0']['cls_prob']).max(1)[0] > 0.0)[:total_num_boxes]
        current_sample.cls_prob_mask = cls_prob_mask.bool() #(torch.tensor(features_data['image_info_0']['cls_prob']).max(1)[0] > 0.1)
        current_sample.start_ind = 0
        sent_len = len(self.text_processor._tokenizer.tokenize(sample_info["caption"]))
        current_sample.end_ind = sent_len

        return current_sample
