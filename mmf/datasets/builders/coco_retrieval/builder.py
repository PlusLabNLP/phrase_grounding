# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.common.registry import registry
from mmf.datasets.builders.coco_retrieval.dataset import COCORetrievalDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("coco_retrieval")
class COCORetrievalBuilder(MMFDatasetBuilder):
    def __init__(
        self,
        dataset_name="coco_retrieval",
        dataset_class=COCORetrievalDataset,
        *args,
        **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)
        self.dataset_class = COCORetrievalDataset

    @classmethod
    def config_path(self):
        return "configs/datasets/coco_retrieval/defaults.yaml"

    def update_registry_for_model(self, config):
        if hasattr(self.dataset, "text_processor") and hasattr(
            self.dataset.text_processor, "get_vocab_size"
        ):
            registry.register(
                self.dataset_name + "_text_vocab_size",
                self.dataset.text_processor.get_vocab_size(),
            )
        registry.register(
            self.dataset_name + "_num_final_outputs", config.num_final_outputs,
        )
