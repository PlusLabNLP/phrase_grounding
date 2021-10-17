# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.common.registry import registry
from mmf.datasets.builders.flickr30k.masked_dataset import MaskedFlickr30kDataset
from mmf.datasets.builders.vqa2 import VQA2Builder
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


#class Flickr30kBuilder(VQA2Builder):
@registry.register_builder("masked_flickr30k")
class Flickr30kBuilder(MMFDatasetBuilder):
    def __init__(
        self,
        dataset_name="masked_flickr30k",
        dataset_class=MaskedFlickr30kDataset,
        *args,
        **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)
        self.dataset_class = MaskedFlickr30kDataset

    @classmethod
    def config_path(self):
        return "configs/datasets/flickr30k/masked.yaml"

    def update_registry_for_model(self, config):
        if hasattr(self.dataset, "text_processor") and hasattr(
            self.dataset.text_processor, "get_vocab_size"
        ):
            registry.register(
                self.dataset_name + "_text_vocab_size",
                self.dataset.text_processor.get_vocab_size(),
            )
        #registry.register(
        #    self.dataset_name + "_num_final_outputs", config.num_final_outputs,
        #)
