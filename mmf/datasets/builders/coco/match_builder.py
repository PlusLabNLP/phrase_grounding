from mmf.common.registry import registry
from mmf.datasets.builders.coco.builder import COCOBuilder

from .match_dataset import MatchCOCODataset


@registry.register_builder("match_coco")
class MatchCOCOBuilder(COCOBuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "match_coco"
        self.set_dataset_class(MatchCOCODataset)

    def update_registry_for_model(self, config):
        registry.register(
            self.dataset_name + "_text_vocab_size",
            self.dataset.masked_token_processor.get_vocab_size(),
        )

    @classmethod
    def config_path(cls):
        return "configs/datasets/coco/match.yaml"
