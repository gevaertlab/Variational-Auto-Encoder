""" organize embedding dataset, predict, save, load embeddings from all kinds of model on all kinds of datasets """

import json
import yaml
import os
import os.path as osp
from typing import List, Union
from tqdm import tqdm
import numpy as np
from configs.config_vars import BASE_DIR
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
        self.saved = osp.exists(self.file_dir)
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


class EmbeddingPredictor:

    def __init__(self,
                 base_model_name: str,
                 log_name: str,
                 version: int,
                 ):
        self.base_model_name = base_model_name
        self.log_name = log_name
        self.version = version
        # TODO: where to load embeddings
        self.load_dir = osp.join(self.LOG_DIR,
                                 log_name,
                                 f'version_{version}')
        self._config = self._load_config(self.load_dir)
        self.module = self._init_model(self.load_dir,
                                       self.base_model_name)
        pass

    def _load_config(self, load_dir):
        config_path = os.path.join(load_dir, 'config.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        # NOTE: HACK!!! the config file will have the hidden_dims reversed, 
        # so reverse it to have the same model
        # assume that the original is from small to large
        config['model_params']['hidden_dims'].sort()
        return config

    def _init_model(self, base_dir, base_model_name):
        vae_model = VAE_MODELS[base_model_name](**self._config['model_params'])
        ckpt_path = self.__getckptpath__(base_dir)
        return VAEXperiment.load_from_checkpoint(ckpt_path,
                                                 vae_model=vae_model,
                                                 params=self._config['exp_params'])

    def predict_embedding(self,
                          dataloader,
                          dl_params={'shuffle': False, 'drop_last': False},
                          split=None):
        """ 
        predict embedding with specified dataloader (dl) name and params 
        dl_params: e.g. {'shuffle':False, 'drop_last':False}
        """
        embedding = Embedding(self.log_name, self.version, split)
        if isinstance(dataloader, str):
            dataloader = getattr(self.module)(**dl_params)
        elif isinstance(dataloader, torch.utils.data.DataLoader):
            # input dataloader instance
            dataloader = dataloader
        else:
            raise NotImplementedError("[EmbeddingPredictor] \
                Not supported dataloder type: \'{type(dataloader)}\'")
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
