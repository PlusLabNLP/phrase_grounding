# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import jsonlines
import torch
import random
import numpy as np
import _pickle as cPickle

from mmf.tools.refer.refer import REFER

class RefCOCODatabase(torch.utils.data.Dataset):
    def __init__(self, imdb_path, dataset_type, ):
        super().__init__()
        self.refer = REFER(dataroot, dataset='refcoco+', splitBy="unc") 
        self.ref_ids = self.refer.getRefIds(split=dataset_type) 
        self._dataset_type = dataset_type
        self._load_annotations()
        #self._load_annotations(imdb_path, test_id_file_path, hard_neg_file_path)
        #self._metadata = {}

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, x):
        self._metadata = x

    def _load_annotations(self): #, imdb_path, test_id_path, hard_neg_file_path):
        entries = []
        for ref_id in self.ref_ids:
            ref = self.refer.Refs[ref_id]
            image_id = ref['image_id']
            ref_id = ref['ref_id']
            refBox = refer.getRefBox(ref_id)
            for sent, sent_id in zip(ref['sentences'], ref['sent_ids']):
                caption = sent['raw']
                entries.append(
                    {'caption': caption, 'sent_id': sent_id, 'image_id': image_id, \
                    'refBox': refBox, 'ref_id': ref_id}
                )

        self._entries = entries
        self.db_size = len(self._entries)

    def __len__(self):
        return self.db_size

    def __getitem__(self, idx):

        entry = self._entries[idx]
        return entry
