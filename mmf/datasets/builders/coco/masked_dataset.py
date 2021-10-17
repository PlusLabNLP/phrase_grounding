import random
import torch

from mmf.common.sample import Sample
from mmf.datasets.builders.coco import COCODataset


class MaskedCOCODataset(COCODataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(config, dataset_type, imdb_file_index, *args, **kwargs)
        self.dataset_name = "masked_coco"
        self._two_sentence = config.get("two_sentence", True)
        self._false_caption = config.get("false_caption", True)
        self._two_sentence_probability = config.get("two_sentence_probability", 0.5)
        self._false_caption_probability = config.get("false_caption_probability", 0.5)
        #self.vocab_map = torch.load('/nas/home/ziyidou/mmf/object_vocab_bertbaseuncased.pt')
        #self.save_vocab_map()

    def load_item(self, idx):
        sample_info = self.annotation_db[idx]
        current_sample = Sample()

        is_correct = -1

        if self._use_features:
            features = self.features_db[idx]
            #import time
            #start = time.time()
            #if len(self.annotation_db) > 80000:
            #    current_sample.image_text_label = torch.load(f'/nas/home/ziyidou/mmf_visual_electra/features/{idx}.pt')
            #else:
            #    current_sample.image_text_label = torch.matmul(torch.tensor(features['image_info_0']['cls_prob']), self.vocab_map)
            #current_sample.image_text_label = torch.matmul(torch.tensor(features['image_info_0']['cls_prob']), self.vocab_map)
            #current_sample.cls_prob = torch.tensor(features['image_info_0']['cls_prob']) #torch.matmul(torch.tensor(features['image_info_0']['cls_prob']), self.vocab_map)
            num_reg = min(torch.tensor(features['image_info_0']['cls_prob']).size(0), self.config.max_features)
            #print(num_reg)
            #objects = torch.tensor(features['image_info_0']['objects']) #torch.matmul(torch.tensor(features['image_info_0']['cls_prob']), self.vocab_map)
            #print(objects[:10])
            #print(features['image_info_0']['cls_prob'].argmax(1)[:10])
            objects = torch.tensor(features['image_info_0']['cls_prob']).argmax(1)[:num_reg]
            current_sample.cls_prob_mask = (torch.tensor(features['image_info_0']['cls_prob']).max(1)[0] > 0.1)[:num_reg]
            cls_prob = torch.zeros((num_reg, 1601))
            ind = torch.arange(num_reg)
            cls_prob[ind, objects] = 1
            current_sample.cls_prob = cls_prob #torch.tensor(features['image_info_0']['cls_prob']) #torch.matmul(torch.tensor(features['image_info_0']['cls_prob']), self.vocab_map)
            num_boxes = (len(features["image_info_0"]["bbox"]))
            #current_sample.vocab_map = self.vocab_map
            #end = time.time()
            #current_sample.image_text_label = self.image_text_labels[idx] #torch.matmul(torch.tensor(features['image_info_0']['cls_prob']), self.vocab_map)
            #current_sample.image_text_label = torch.matmul(torch.tensor(features['image_info_0']['cls_prob']), self.vocab_map)
            #end2 = time.time()
            #print(1)
            #print(end-start)
            #print(end2-end)
            if hasattr(self, "transformer_bbox_processor"):
                features["image_info_0"] = self.transformer_bbox_processor(
                    features["image_info_0"]
                )

            boxes = (features["image_info_0"]["bbox"])
            ffeatures = features["image_feature_0"]
            all_boxes_pad = torch.zeros((self.config.max_region_num, 5))
            all_features_pad = torch.zeros(
                (self.config.max_region_num, 2048), dtype=torch.float32
            )

            all_boxes_pad[:num_reg] = boxes[:num_reg]
            all_features_pad[:num_reg] = ffeatures[:num_reg]
            current_sample.image_feature_0 = all_features_pad
            current_sample.image_info_0 = {}
            current_sample.image_info_0["image_width"] = features["image_info_0"][
                "image_width"
            ]
            current_sample.image_info_0["image_height"] = features["image_info_0"][
                "image_height"
            ]
            current_sample.image_info_0["bbox"] = all_boxes_pad
            current_sample.image_info_0["max_features"] = torch.tensor(num_reg)
            current_sample.image_info_0["max_image_features"] = num_reg
            #current_sample.targets = target

            current_sample.update(
                {
                    "image_labels": self.masked_region_processor(
                        all_features_pad, (is_correct == -1)
                    )
                }
            )

            #current_sample.update(features)
            if True:
                def clip(x):
                    return max(min(1, x), 0)
                image_path = str(sample_info["image_name"]) + ".jpg"
                image_ori = self.image_db.from_path(image_path)["images"][0]
                import torchvision
                size = (256, 256)
                transform_list = []
                transform_list.append(torchvision.transforms.Resize(size=size))
                transform_list.append(torchvision.transforms.ToTensor())
                transform = torchvision.transforms.Compose(transform_list)
                image = transform(image_ori)

                a, b, c = image.size()

                images = [] #'''
                image_size = 256
                regress_mask = []
                cnt = 0
                for bbox in features["image_info_0"]['bbox']:
                    if cnt == self.config.max_region_num:
                        break
                    cnt += 1
                    bbox[0] = clip(bbox[0])
                    bbox[1] = clip(bbox[1])
                    bbox[2] = clip(bbox[2])
                    bbox[3] = clip(bbox[3])
                    b1 = int(b*bbox[1])
                    c1 = int(c*bbox[0])
                    b2 = int(b*bbox[3])
                    c2 = int(c*bbox[2])
                    #if b2-b1 > 0 and c2-c1 > 0:
                    if bbox[3]-bbox[1]> 0 and bbox[2]-bbox[0] > 0:
                        cur_image = image.clone()
                        #mask[:, b1:b2, c1:c2] = 1.0
                        cur_image = cur_image[:, b1:b2, c1:c2] #torch.zeros_like(cur_image)
                        import torch.nn.functional as F
                        cur_image = F.interpolate(cur_image.unsqueeze(0), size=(image_size)).squeeze(0)
                        #cur_image = cur_image*mask
                        images.append(cur_image.view(-1)) #'''
                        regress_mask.append(1)
                    else:
                        regress_mask.append(0)
                        cur_image = torch.zeros(3, image_size, image_size)
                        images.append(cur_image.view(-1))
                        #cur_image = torch.zeros_like(image)
                images_pad = torch.zeros((self.config.max_region_num, 3*image_size*image_size))
                
                images = torch.stack(images, 0) #'''
                images_pad[:len(regress_mask)] = images[:len(regress_mask)]
                current_sample.images = images_pad #torch.stack(images, 0) #'''
                current_sample.image = image #torch.stack(images, 0) #'''
                regress_mask_pad = torch.zeros(self.config.max_region_num)
                regress_mask = torch.tensor(regress_mask)
                regress_mask_pad[:len(regress_mask)] = regress_mask[:len(regress_mask)]
                current_sample.regress_mask = torch.tensor(regress_mask)

        else:

            image_path = str(sample_info["image_name"]) + ".jpg"
            image = self.image_db.from_path(image_path)["images"][0]

        current_sample = self._add_masked_caption(sample_info, current_sample, is_correct)
        image_id = sample_info["image_id"]
        current_sample.image_id = torch.tensor(int(str(image_id).split('.')[0]))
        return current_sample

    def _add_masked_caption(self, sample_info, current_sample, is_correct):
        captions = sample_info["captions"]
        image_id = sample_info["image_id"]
        num_captions = len(captions)
        selected_caption_index = random.randint(0, num_captions - 1)
        other_caption_indices = [
            i for i in range(num_captions) if i != selected_caption_index
        ]
        selected_caption = captions[selected_caption_index]
        other_caption = None

        if self._dataset_type == "train":
            if is_correct != -1:
                tt = random.random()
                if tt < 0.5:
                    selected_caption = self._get_mismatching_caption(image_id)
                    is_correct = 0
                else:
                    is_correct = 1

        processed = self.masked_token_processor(
            {
                "text_a": selected_caption,
                "text_b": other_caption,
                "is_correct": is_correct,
            }, None if is_correct == -1 else 0.0
        )
        processed.pop("tokens")
        current_sample.update(processed)

        return current_sample

    def _get_mismatching_caption(self, image_id):
        other_item = self.annotation_db[random.randint(0, len(self.annotation_db) - 1)]

        while other_item["image_id"] == image_id:
            other_item = self.annotation_db[
                random.randint(0, len(self.annotation_db) - 1)
            ]

        other_caption = other_item["captions"][
            random.randint(0, len(other_item["captions"]) - 1)
        ]
        return other_caption
