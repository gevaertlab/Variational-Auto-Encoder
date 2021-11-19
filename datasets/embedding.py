""" organize embedding dataset, predict, save, load embeddings from all kinds of model on all kinds of datasets """

import os
import os.path as osp
from typing import List, Union

import numpy as np
import ujson as json  # switch to ujson
from configs.config_vars import BASE_DIR
from evaluations.evaluator import BaseEvaluator
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from utils.funcs import get_order, reorder
from utils.python_logger import get_logger


class Embedding:
    """
    store embedding data format, can save and load embeddings from drive
    can get embeddings of various types (aug/no aug, train/val)
    """
    LOG_DIR = osp.join(BASE_DIR, 'applications/logs')

    def __init__(self,
                 log_name: str,
                 version: int,
                 tag: str = '',
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
        self.tag = tag
        if tag:
            self.file_name = f"{tag}_" + self.file_name
        self.file_dir = osp.join(self.save_dir, self.file_name)
        self.saved = osp.exists(self.file_dir)  # HACK: could be empty
        self.logger = get_logger(self.__class__.__name__)
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
        self.logger.info("saving embedding ...")
        if not osp.exists(self.save_dir):
            os.mkdir(self.save_dir)
        if reorder:
            self.reorder_embeddings()
        with open(self.file_dir, 'w') as f:
            json.dump(self._data, f)
        if verbose:
            self.logger.info(
                f"{len(self._data['index'])} embeddings saved to {osp.join(self.save_dir, self.file_name)}")
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

    def _filter_aug(self):
        """ 
        return self._data without augmented patches, if didn't aug,
        return self._data
        """
        if "Aug" in self._data['index'][0]:
            non_aug = [(i, name) for (i, name) in enumerate(
                self._data['index']) if "Aug00" in name]
            non_aug_idx = [i for (i, name) in non_aug]
            non_aug_name = [name for (i, name) in non_aug]
            non_aug_embedding = [self._data['embedding'][i]
                                 for i in non_aug_idx]
            return non_aug_embedding, non_aug_name
        else:
            return self.get_embedding(augment=True)

    def get_embedding(self, augment=False):
        """
        Get embedding
        Args:
            self.saved.
            augment: whether to read augmented embeddings, default to be false
        Returns:
            np.array: embedding
        """
        if not self.saved:
            self.save()
        if not augment:
            return self._filter_aug()

        else:
            return np.array(self._data['embedding']), self._data['index']


class EmbeddingPredictor(BaseEvaluator):

    def __init__(self,
                 log_name: str,
                 version: int,
                 base_model_name: str = 'VAE3D'):
        super().__init__(log_name=log_name,
                         version=version,
                         base_model_name=base_model_name)
        self.logger = get_logger(self.__class__.__name__)

    def predict_embedding(self,
                          dataloader,
                          dl_params={'shuffle': False, 'drop_last': False},
                          split=None):
        """ 
        predict embedding with specified dataloader (dl) name and params 
        dl_params: e.g. {'shuffle':False, 'drop_last':False}
        """
        embedding = Embedding(log_name=self.log_name,
                              version=self.version,
                              tag="" if not isinstance(
                                  dataloader, DataLoader) else dataloader.dataset.__class__.__name__,
                              split=split)
        dataloader = self._parse_dataloader(dataloader, dl_params=dl_params)
        self.logger.info(f"Predicting embeddings for {len(dataloader)} images")
        for batch, file_names in tqdm(dataloader):
            name_batch = [fn.replace('.nrrd', '') for fn in file_names]
            encoded = self.module.model.encode(batch)
            mean = encoded[0].detach().numpy().tolist()
            std = encoded[1].detach().numpy().tolist()
            embedding.stack_embedding(name_batch,
                                      np.concatenate((mean, std), 1))
        embedding.save()
        return embedding
