# exporting the embeddings and labels after matching

import os
import os.path as osp

import pandas as pd
from applications import TASK_DICT
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
                 task_names: str = list(TASK_DICT.keys()),
                 base_model_name: str = 'VAE3D'):
        super().__init__(log_name=log_name,
                         version=version,
                         base_model_name=base_model_name)
        self.timer = Timer(
            name=(osp.basename(__file__), self.__class__.__name__))
        self.timer()
        self.task_names = task_names
        self.logger = get_logger(self.__class__.__name__)
        # get load and save dir
        self.load_dir = osp.join(self.LOG_DIR,
                                 log_name,
                                 f'version_{version}')  # NOTE
        pass

    def get_data(self):
        embeddings, data_names = self.get_embeddings()
        label_dict = {task_name: self.get_labels(label_name=task_name,
                                                 data_names=data_names)
                      for task_name in self.task_names}
        return embeddings, data_names, label_dict

    def get_labels(self, label_name, data_names):
        """
        get labels assuming embedding is get,
        return dict of labels not label instance
        """
        label_instance = LABEL_DICT[label_name]()
        label = {}

        # get labels
        self.timer()
        label['train'] = label_instance.get_labels(data_names['train'])
        label['val'] = label_instance.get_labels(data_names['val'])
        self.timer('match labels')
        return label

    def get_embeddings(self, augment=False):
        self.logger.info("initializing embeddings")
        embeddings = {'train': Embedding(self.log_name,
                                         self.version,
                                         split='train'),
                      'val': Embedding(self.log_name,
                                       self.version,
                                       split='val')}
        data_names = {}
        predictor = EmbeddingPredictor(log_name=self.log_name,
                                       version=self.version,
                                       base_model_name=self.base_model_name,)

        if not embeddings['train'].saved:
            embeddings['train'] = predictor.predict_embedding(dataloader='train_dataloader',
                                                              split='train')
        else:
            _ = embeddings['train'].load()
        if not embeddings['val'].saved:
            embeddings['val'] = predictor.predict_embedding(dataloader='val_dataloader',
                                                            split='val')
        else:
            _ = embeddings['val'].load()

        embeddings['train'], data_names['train'] = \
            embeddings['train'].get_embedding(augment=augment)
        embeddings['val'], data_names['val'] = \
            embeddings['val'].get_embedding(augment=augment)

        return embeddings, data_names

    def save_for_r(self, save_dir: str):
        # save embedding as two csvs
        # save label_dict into one csv

        embeddings, data_names, label_dict = self.get_data()

        mkdir_safe(save_dir)

        self.save_csv(embeddings['train'], osp.join(
            save_dir, "embeddings_train.csv"))
        self.save_csv(embeddings['val'], osp.join(
            save_dir, "embeddings_val.csv"))
        self.save_csv(label_dict, osp.join(save_dir, "label_dict.csv"))
        pass

    def save_csv(self, data, save_path: str):
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)
        self.logger.info(f"saved to {save_path}")
        pass
