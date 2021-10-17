from mmf.common.registry import registry
from mmf.datasets.builders.refcoco.dataset import RefCOCODataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("refcoco")
class RefCOCOBuilder(MMFDatasetBuilder):
    def __init__(
        self, dataset_name="refcoco", dataset_class=RefCOCODataset, *args, **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)
        self.dataset_class = RefCOCODataset

    @classmethod
    def config_path(self):
        return "configs/datasets/refcoco/defaults.yaml"

    '''def update_registry_for_model(self, config):
        if hasattr(self.dataset, "text_processor") and hasattr(
            self.dataset.text_processor, "get_vocab_size"
        ):
            registry.register(
                self.dataset_name + "_text_vocab_size",
                self.dataset.text_processor.get_vocab_size(),
            )
        registry.register(
            self.dataset_name + "_num_final_outputs", self._num_final_outputs,
        )'''
