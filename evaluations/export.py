# exporting the embeddings and labels after matching

import os
import os.path as osp
from typing import Dict, List
import pandas as pd
from torch.utils.data.dataloader import DataLoader
from applications.tasks import TASK_DICT
from configs.config_vars import BASE_DIR
from datasets.embedding import Embedding, EmbeddingPredictor
from datasets.label.label_dict import LABEL_DICT
from utils.io import mkdir_safe
from utils.python_logger import get_logger
from utils.timer import Timer


class Exporter(EmbeddingPredictor):  # inherited from BaseEvaluator

    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    APP_DIR = os.path.join(BASE_DIR, 'applications')

    def __init__(self,
                 log_name: str,
                 version: int,
                 dataloaders: dict = {
                     'train': 'train_dataloader', 'val': 'val_dataloader'},
                 task_names: str = list(TASK_DICT.keys()),
                 base_model_name: str = 'VAE3D'):
        super().__init__(log_name=log_name,
                         version=version,
                         base_model_name=base_model_name)
        self.timer = Timer(
            name=(osp.basename(__file__), self.__class__.__name__))
        self.timer()
        self.task_names = task_names
        self.dataloaders = dataloaders
        self.logger = get_logger(self.__class__.__name__)
        # get load and save dir
        self.load_dir = osp.join(self.LOG_DIR,
                                 log_name,
                                 f'version_{version}')  # NOTE
        pass

    def get_data(self):
        """ loads embeddings, data_names and label_dict in default settings """
        embeddings, data_names = self.get_embeddings()
        label_dict = {task_name: self.get_labels(label_name=task_name,
                                                 data_names=data_names)
                      for task_name in self.task_names}
        return embeddings, data_names, label_dict

    def get_labels(self, label_name, data_names: List, label_kwds: dict = {}):
        """
        get labels assuming embedding is get,
        return dict of labels not label instance
        """
        label_instance = LABEL_DICT[label_name](**label_kwds)
        label = {}

        # get labels
        self.timer()
        label['train'] = label_instance.get_labels(data_names['train'])
        label['val'] = label_instance.get_labels(data_names['val'])
        self.timer('match labels')
        return label

    def get_embeddings(self,
                       tag="",
                       overwrite=False,
                       augment=False):
        # NOTE: self.dataloaders is a dictionary of str or dataloaders
        self.logger.info("initializing embeddings")
        embeddings = {}
        predictor = EmbeddingPredictor(log_name=self.log_name,
                                       version=self.version,
                                       base_model_name=self.base_model_name,)
        for key, dataloader in self.dataloaders.items():
            tag = "" if not isinstance(
                dataloader, DataLoader) else dataloader.dataset.__class__.__name__
            embeddings[key] = Embedding(log_name=self.log_name,
                                        version=self.version,
                                        tag=tag,
                                        split=key)  # key = split
            if not embeddings[key].saved or overwrite:
                embeddings[key] = predictor.predict_embedding(dataloader=self.dataloaders[key],
                                                              embedding=embeddings[key],
                                                              tag=tag,
                                                              split=key)
            else:
                _ = embeddings[key].load()

        embeds, data_names = {}, {}

        for key, embedding in embeddings.items():
            embeds[key], data_names[key] = embedding.get_embedding(
                augment=augment)
        return embeds, data_names

        # if (not tag) and (isinstance(self.dataloader, DataLoader)):
        #     tag = self.dataloader.dataset.__class__.__name__

        # self.logger.info("initializing embeddings")
        # embeddings = {'train': Embedding(log_name=self.log_name,
        #                                  version=self.version,
        #                                  tag=self.dataloader,
        #                                  split='train'),
        #               'val': Embedding(log_name=self.log_name,
        #                                version=self.version,
        #                                tag=tag,
        #                                split='val')}
        # data_names = {}
        # predictor = EmbeddingPredictor(log_name=self.log_name,
        #                                version=self.version,
        #                                base_model_name=self.base_model_name,)

        # if not embeddings['train'].saved:
        #     embeddings['train'] = predictor.predict_embedding(dataloader=self.dataloader['train'],
        #                                                       embedding=embeddings['train'],
        #                                                       tag=tag,
        #                                                       split='train')
        # else:
        #     _ = embeddings['train'].load()
        # if not embeddings['val'].saved:
        #     embeddings['val'] = predictor.predict_embedding(dataloader=self.dataloader['val'],
        #                                                     embedding=embeddings['val'],
        #                                                     tag=tag,
        #                                                     split='val')
        # else:
        #     _ = embeddings['val'].load()

        # embeddings['train'], data_names['train'] = \
        #     embeddings['train'].get_embedding(augment=augment)
        # embeddings['val'], data_names['val'] = \
        #     embeddings['val'].get_embedding(augment=augment)

        # return embeddings, data_names

    def save_for_r(self, save_dir: str):
        # save embedding as two csvs
        # save label_dict into two csvs (train + val)

        embeddings, data_names, label_dict = self.get_data()

        mkdir_safe(save_dir)

        self.save_csv(embeddings['train'], osp.join(
            save_dir, "embeddings_train.csv"))
        self.save_csv(embeddings['val'], osp.join(
            save_dir, "embeddings_val.csv"))

        # label_dict
        label_dict_train = {name: label_dict[name]['train']
                            for name in list(label_dict.keys())}
        label_dict_val = {name: label_dict[name]['val']
                          for name in list(label_dict.keys())}
        self.save_csv(label_dict_train, osp.join(
            save_dir, "label_dict_train.csv"))
        self.save_csv(label_dict_val, osp.join(save_dir, "label_dict_val.csv"))
        pass

    def save_csv(self, data, save_path: str):
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)
        self.logger.info(f"saved to {save_path}")
        pass
