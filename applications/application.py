''' This file utilize the trained encoder to do downstream tasks '''


from applications import TASK_DICT
from models import VAE_MODELS
from experiment import VAEXperiment
import torch
import yaml
import os
from typing import Union, List
import numpy as np
import json
from .models import predictTask
from tqdm import tqdm
from utils.visualization import ytrue_ypred_scatter, confusion_matrix_models
from utils.funcs import Timer


class Application:

    LOG_DIR = '/labs/gevaertlab/users/yyhhli/code/vae/logs/'
    APP_DIR = '/labs/gevaertlab/users/yyhhli/code/vae/applications/'

    def __init__(self, log_name: str, version: int, task_name: str, base_model_name: str = 'VAE3D'):
        self.base_model_name = base_model_name
        self.log_name = log_name
        self.version = version
        self.load_dir = os.path.join(
            self.LOG_DIR, log_name, f'version_{version}')  # NOTE
        self.task_name = task_name
        self.label = LABEL_DICT[task_name]()
        self.task = TASK_DICT[task_name]()
        self.embeddings = {}
        self.result_dict = {}
        self.timer = Timer()

        # further inits
        self._config = self.__load_config__(self.load_dir)

        # init the experiment and load checkpoint
        self.module = self.__init_model__(self.load_dir, self.base_model_name)

        # results
        self.result_dict = {}
        self.pred_dict = {}
        self.hparam_dict = {}
        self.pred_stats = {}
        pass

    def __load_config__(self, load_dir):
        config_path = os.path.join(load_dir, 'config.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        # NOTE: HACK!!! the config file will have the hidden_dims reversed, so reverse it to have the same model
        # assume that the original is from small to large
        config['model_params']['hidden_dims'].sort()
        return config

    def __init_model__(self, base_dir, base_model_name):
        vae_model = VAE_MODELS[base_model_name](**self._config['model_params'])
        ckpt_path = self.__getckptpath__(base_dir)
        return VAEXperiment.load_from_checkpoint(ckpt_path, vae_model=vae_model, params=self._config['exp_params'])

    def __getckptpath__(self, base_dir):
        # default to load the last checkpoint
        ckpt_dir = os.path.join(base_dir, 'checkpoints')
        ckpt_path = os.path.join(ckpt_dir, sorted(os.listdir(ckpt_dir))[-1])
        return ckpt_path

    def get_embeddings(self, split='train'):

        if split == 'train':
            dataloader = self.module.train_dataloader()
            dataloader.shuffle = False
        elif split == 'val':
            dataloader = self.module.val_dataloader()[0]
        else:
            raise NotImplementedError(f'split {split} does not supported')

        if split not in self.embeddings.keys():
            self.embeddings[split] = Embedding(
                self.log_name, self.version, split=split)

        if self.embeddings[split].saved:  
            # if saved directly load
            print(f"Loading embeddings: {split} ...")
            _ = self.embeddings[split].load()

        else:  
            # if not saved, predict
            print(f"New embedding, predicting: {split} ...")
            for data, file_names in tqdm(dataloader):
                idx = [file_name.replace('.nrrd', '')
                       for file_name in file_names]
                encoded = self.module.model.encode(data)
                mean = encoded[0].detach().numpy().tolist()
                std = encoded[1].detach().numpy().tolist()  # add std
                self.embeddings[split].stackEmbedding(
                    idx, np.concatenate((mean, std), 1))
            self.embeddings[split].save()
        pass

    def getLabels(self, split: str = 'train'):
        """ NOTE: could be time comsuming """
        assert split in self.embeddings.keys(
        ), f'Embedding not defined for {split}, cannot get labels'
        file_lst = self.embeddings[split].embeddings['index']
        self.timer()
        labels = self.task.get_labels(file_lst)
        # labels = self.task.match_labels(data_lst=file_lst)
        self.timer('match labels')
        return labels

    def save_labels(self, data):
        """ save matched label as npy file """
        save_path = os.path.join(self.embeddings['train'].LOG_DIR,
                                 "labels",
                                 f"Y_matched_{self.task.name}.npy")
        np.save(save_path, data)
        print(f"saved to {save_path}")
        pass
    # TODO: how to solve the problem of matching and loading logic!!!

    def load_labels(self):
        save_path = os.path.join(self.embeddings['train'].LOG_DIR,
                                 "labels",
                                 f"Y_matched_{self.task.name}.npy")
        assert os.path.exists(save_path), "file not found"
        data = np.load(save_path, allow_pickle=True).item()
        return data

    def taskPrediction(self, models='all'):
        """
        Predict a task
        Args:
            models (str or list of str, optional): models to use. Defaults to 'all'.
        Returns:
            dict: results (metrics) of predictions
        """
        
        # get X
        self.get_embeddings(split='train')
        self.get_embeddings(split='val')

        # format the X and Y so that they can be taken by sklearn models
        X = {'train': self.embeddings['train'].getEmbedding(),
             'val': self.embeddings['val'].getEmbedding()}

        # get Y
        
        # modified, using label instance, loading with file names
        data_names = {'train': self.embeddings['train'].embeddings['index'],
                      'val': self.embeddings['val'].embeddings['index']}

        Y = {'train': self.label.get_labels(data_names['train']),
             'val': self.label.get_labels(data_names['val'])}

        # save_path = os.path.join(self.embeddings['train'].LOG_DIR,  # NODE: duplicate code
        #                          "labels",
        #                          f"Y_matched_{self.task.name}.npy")
        # if os.path.exists(save_path):
        #     print("loading matched Y")
        #     Y = self.load_labels()

        # # 2. match labels using class task's pylidc method
        # else:
        #     Y = {'train': np.array(self.getLabels(split='train')),
        #          'val': np.array(self.getLabels(split='val'))}
        #     self.save_labels(Y)

        # # get aligned X and Y, HACK: currently doing nothing
        # X['train'], Y['train'] = self.task.merge(X['train'], Y['train'])
        # X['val'], Y['val'] = self.task.merge(X['val'], Y['val'])

        # save Ys
        # os.path.join(self.embeddings['train'].LOG_DIR, "labels")

        # preprocess X and Y
        X, Y = self.task.transform(X, Y)

        # TRAIN + PREDICT
        self.load_hparam_dict()
        results = predictTask(X, Y,
                              self.task.task_type,
                              models=models,
                              hparam_dict=self.hparam_dict)
        result_dict, pred_dict, pred_stats, hparam_dict = results

        # TODO: reverse transform Y
        self.result_dict = result_dict
        self.pred_dict = pred_dict
        self.hparam_dict = hparam_dict  # update hparam dict
        self.pred_stats = pred_stats
        return result_dict, pred_dict, hparam_dict

    def load_hparam_dict(self):
        """ load hparam dictionary which should be the same place as save"""

        hparam_log_file = os.path.join('results', '.'.join(
            [self.log_name, str(self.version), self.task_name, 'best_hparams', 'json']))
        hparam_log_file_path = os.path.join(self.APP_DIR, hparam_log_file)
        if os.path.exists(hparam_log_file_path):
            print("Loading best hparams ...")
            with open(hparam_log_file_path, 'r') as f:
                self.hparam_dict = json.load(f)
        else:
            print("New task, no hparams file loaded")
        pass

    def saveResults(self, verbose=True):
        """ 
        save the results after completing a task prediction job 
        saving:
        1. metrics for best models
        2. hyperparameters for best models 
        3. predictions for best models
        """

        # 1. saving metrics for best models
        assert self.result_dict.keys(), "result_dict doesn't exist"
        # save result_dict file
        result_log_file = os.path.join('results', '.'.join(
            [self.log_name, str(self.version), self.task_name, 'json']))
        with open(os.path.join(self.APP_DIR, result_log_file), 'w') as f:
            json.dump(self.result_dict, f)
        if verbose:
            print(f"Saved results to {result_log_file}")

        # 2. saveing hyperparameters for best models
        assert self.hparam_dict.keys(), "hparam_dict doesn't exist"
        # save hparam_dict file
        hparam_log_file = os.path.join('results', '.'.join(
            [self.log_name, str(self.version), self.task_name, 'best_hparams', 'json']))
        with open(os.path.join(self.APP_DIR, hparam_log_file), 'w') as f:
            json.dump(self.hparam_dict, f)
        if verbose:
            print(f"Saved results to {hparam_log_file}")

        # 3. saving predictions for best models: use NPY files
        assert self.pred_dict.keys(), "pred_dict doesn't exist"
        # save pred_dict file
        pred_dict_file = os.path.join('results', '.'.join(
            [self.log_name, str(self.version), self.task_name, 'preds', 'npy']))
        np.save(os.path.join(self.APP_DIR, pred_dict_file), self.pred_dict)
        if verbose:
            print(f"Saved results to {pred_dict_file}")

        # 4. saving pred_stats for best models: use NPY files
        # NOTE: is ok for the pred_stats to be not exist
        if self.pred_stats.keys():
            # save self.pred_stats file
            pred_stats_file = os.path.join('results', '.'.join(
                [self.log_name, str(self.version), self.task_name, 'pred_stats', 'npy']))
            np.save(os.path.join(self.APP_DIR, pred_stats_file), self.pred_stats)
            if verbose:
                print(f"Saved results to {pred_stats_file}")
            pass

    def draw_dignosis_figure(self, verbose=True):
        assert self.pred_dict.keys(), "pred_dict not exists"
        # based on the task type, draw different things
        diagnosis_figure_file = os.path.join('results', '.'.join(
            [self.log_name, str(self.version), self.task_name, 'figure.jpeg']))
        if self.task.task_type == 'regression':
            # draw scatter
            try:
                ytrue_ypred_scatter(self.pred_dict, os.path.join(
                    self.APP_DIR, diagnosis_figure_file))
            except Exception as e:
                print(e)
                return
        elif self.task.task_type == 'classification':
            try:
                confusion_matrix_models(self.pred_dict,
                                        os.path.join(self.APP_DIR,
                                                     diagnosis_figure_file),
                                        classes=list(range(1, 6)))
            except Exception as e:
                print(e)
                return
        if verbose:
            print(f"Saved figure to {diagnosis_figure_file}")
        pass


class Embedding:

    LOG_DIR = '/labs/gevaertlab/users/yyhhli/code/vae/applications/logs'

    def __init__(self, log_name: str, version: int, split: str = 'train'):
        self.log_name = log_name
        self.version = version
        self.split = split
        self.embeddings = {'index': [], 'embedding': []}
        self.save_dir = os.path.join(
            self.LOG_DIR, f'{self.log_name}.{self.version}')
        self.file_name = f'embedding_{self.split}.json'
        self.file_dir = os.path.join(self.save_dir, self.file_name)
        self.saved = os.path.exists(self.file_dir)
        pass

    def stackEmbedding(self, index_lst: List[str], embedding_lst: List[Union[List, np.ndarray]]):
        self.embeddings['index'] += index_lst
        self.embeddings['embedding'] += [list(embedding)
                                         for embedding in embedding_lst]
        pass

    def appendEmbedding(self, index: str, embedding: Union[List, np.ndarray]):
        self.embeddings['index'].append(index)
        self.embeddings['embedding'].append(list(embedding))
        pass

    def save(self, verbose=False):
        print("saving embedding ...")
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        with open(self.file_dir, 'w') as f:
            json.dump(self.embeddings, f)
        if verbose:
            print(
                f"{len(self.embeddings['index'])} embeddings saved to {os.path.join(self.save_dir, self.file_name)}")
        pass

    def load(self):
        with open(self.file_dir, 'r') as f:
            data = json.load(f)
        if len(self.embeddings['index']) == 0:
            self.embeddings = data
        return data

    def getEmbedding(self):
        """
        Get embedding
        Args:
            self.saved.
        Returns:
            np.array: embedding
        """
        if not self.saved:
            self.save()
        return np.array(self.embeddings['embedding'])


def debug():
    ap = Application(log_name='VAE32',
                     version=48,
                     task_name='malignancy',
                     base_model_name='VAE3D')
    ap.get_labels()
    print(volume)


if __name__ == '__main__':
    debug()
