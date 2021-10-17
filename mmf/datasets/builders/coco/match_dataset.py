import random
import torch

from mmf.common.sample import Sample
from mmf.datasets.builders.coco import COCODataset


class MatchCOCODataset(COCODataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(config, dataset_type, imdb_file_index, *args, **kwargs)
        self.dataset_name = "match_coco"
        self._two_sentence = config.get("two_sentence", True)
        self._false_caption = config.get("false_caption", True)
        self._two_sentence_probability = config.get("two_sentence_probability", 0.5)
        self._false_caption_probability = config.get("false_caption_probability", 0.5)
        #self.vocab_map = torch.load('/nas/home/ziyidou/mmf/object_vocab_bertbaseuncased.pt')

    def load_item(self, idx):
        sample_info = self.annotation_db[idx]
        current_sample = Sample()

        if self._use_features:
            features = self.features_db[idx]
            current_sample.image_text_label = None #torch.matmul(torch.tensor(features['image_info_0']['cls_prob']), self.vocab_map)
            if hasattr(self, "transformer_bbox_processor"):
                features["image_info_0"] = self.transformer_bbox_processor(
                    features["image_info_0"]
                )

            '''if self.config.get("use_image_feature_masks", False):
                current_sample.update(
                    {
                        "image_labels": self.masked_region_processor(
                            features["image_feature_0"]
                        )
                    }
                )'''

            current_sample.update(features)
        else:
            image_path = str(sample_info["image_name"]) + ".jpg"
            current_sample.image = self.image_db.from_path(image_path)["images"][0]

        current_sample = self._add_masked_caption(sample_info, current_sample, idx)
        current_sample.targets = current_sample.is_correct
        return current_sample

    def _add_masked_caption(self, sample_info, current_sample, idx):
        captions = sample_info["captions"]
        image_id = sample_info["image_id"]
        num_captions = len(captions)
        selected_caption_index = random.randint(0, num_captions - 1) if self._dataset_type == "train" else 0
        other_caption_indices = [
            i for i in range(num_captions) if i != selected_caption_index
        ]
        selected_caption = captions[selected_caption_index]
        other_caption = None
        is_correct = True

        if self._dataset_type == "train":
            if random.random() < 0.5:
                selected_caption = self._get_mismatching_caption(image_id)
                is_correct = False
            else:
                is_correct = True
        else:
            if idx < 2500:
                selected_caption = self._get_mismatching_caption_eval(image_id, idx)
                is_correct = False
            else:
                is_correct = True

        processed = self.masked_token_processor(
            {
                "text_a": selected_caption,
                "text_b": other_caption,
                "is_correct": is_correct,
            }
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

    def _get_mismatching_caption_eval(self, image_id, idx):
        cur = idx+1
        other_item = self.annotation_db[cur]

        while other_item["image_id"] == image_id:
            cur = cur + 1
            other_item = self.annotation_db[
                cur
            ]

        other_caption = other_item["captions"][
            0
        ]
        return other_caption
