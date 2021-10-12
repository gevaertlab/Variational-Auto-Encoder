""" organize embedding dataset, predict, save, load embeddings from all kinds of model on all kinds of datasets """

import json
import yaml
import os
import re
import os.path as osp
from typing import List, Union
from tqdm import tqdm
import numpy as np
from configs.config_vars import BASE_DIR
from evaluations.evaluator import BaseEvaluator
from experiment import VAEXperiment
from models import VAE_MODELS

from utils.funcs import get_order, reorder
import torch


class Embedding:
    """
    store embedding data format, can save and load embeddings from drive
    """
    LOG_DIR = osp.join(BASE_DIR, 'applications/logs')

    def __init__(self,
                 log_name: str,
                 version: int,
                 split=None):
        self.log_name = log_name
        self.version = version
        self.split = split
        self._data = {'index': [], 'embedding': []}
        self.save_dir = osp.join(
            self.LOG_DIR, f'{self.log_name}.{self.version}')
        if split:
            self.file_name = f'embedding_{self.split}.json'
        else:
            self.file_name = 'embedding.json'
        self.file_dir = osp.join(self.save_dir, self.file_name)
        self.saved = osp.exists(self.file_dir)  # HACK: could be empty
        pass

    def stack_embedding(self, index_lst: List[str], embedding_lst: List[Union[List, np.ndarray]]):
        self._data['index'] += index_lst
        self._data['embedding'] += [list(embedding)
                                    for embedding in embedding_lst]
        pass

    def append_embedding(self, index: str, embedding: Union[List, np.ndarray]):
        self._data['index'].append(index)
        self._data['embedding'].append(list(embedding))
        pass

    def save(self, reorder=True, verbose=True):
        print("saving embedding ...")
        if not osp.exists(self.save_dir):
            os.mkdir(self.save_dir)
        if reorder:
            self.reorder_embeddings()
        with open(self.file_dir, 'w') as f:
            json.dump(self._data, f)
        if verbose:
            print(f"{len(self._data['index'])} \
                embeddings saved to {osp.join(self.save_dir, self.file_name)}")
        pass

    def reorder_embeddings(self):
        ''' order then according to sorted(index) '''
        order = get_order(self._data['index'],
                          ref_lst=sorted(self._data['index']))
        self._data['index'] = reorder(self._data['index'],
                                      order)
        self._data['embedding'] = reorder(self._data['embedding'],
                                          order)
        pass

    def load(self):
        with open(self.file_dir, 'r') as f:
            data = json.load(f)
        if len(self._data['index']) == 0:
            self._data = data
        return data

    def get_embedding(self):
        """
        Get embedding
        Args:
            self.saved.
        Returns:
            np.array: embedding
        """
        if not self.saved:
            self.save()
        return np.array(self._data['embedding'])


class EmbeddingPredictor(BaseEvaluator):

    def __init__(self, base_model_name: str, log_name: str, version: int):
        super().__init__(base_model_name, log_name, version)

    def predict_embedding(self,
                          dataloader,
                          dl_params={'shuffle': False, 'drop_last': False},
                          split=None):
        """ 
        predict embedding with specified dataloader (dl) name and params 
        dl_params: e.g. {'shuffle':False, 'drop_last':False}
        """
        embedding = Embedding(self.log_name, self.version, split)
        dataloader = self._parse_dataloader(dataloader, dl_params=dl_params)
        print(f"[EmbeddingPredictor] Predicting embeddings \
            for {len(dataloader)} images")
        for batch, file_names in tqdm(dataloader):
            name_batch = [fn.replace('.nrrd', '') for fn in file_names]
            encoded = self.module.model.encode(batch)
            mean = encoded[0].detach().numpy().tolist()
            std = encoded[1].detach().numpy().tolist()
            embedding.stack_embedding(name_batch,
                                      np.concatenate((mean, std), 1))
        embedding.save()
        return embedding
