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

class Application:
    
    LOG_DIR = '/home/yyhhli/code/vae/logs/'
    
    def __init__(self, log_name: str, version: int, task_name: str, base_model_name:str='VAE3D'):
        self.base_model_name = base_model_name
        self.log_name = log_name
        self.version = version
        self.load_dir = os.path.join(self.LOG_DIR, log_name, f'version_{version}') # NOTE
        self.task_name = task_name
        self.task = TASK_DICT[task_name]()
        self.embeddings = {}
        self.result_dict = {}
        # further inits
        self._config = self.__load_config__(self.load_dir)
        self.module = self.__init_model__(self.load_dir, self.base_model_name) # init the experiment and load checkpoint
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
    
    def embed(self, split='train'):
        
        if split == 'train':
            dataloader = self.module.train_dataloader()
        elif split == 'val':
            dataloader = self.module.val_dataloader()[0]
        else:
            raise NotImplementedError(f'split {split} does not supported')
        
        if split not in self.embeddings.keys():
            self.embeddings[split] = Embedding(self.log_name, self.version, split=split)
        
        if self.embeddings[split].saved: # if saved directly load
            print(f"Loading embeddings: {split} ...")
            _ = self.embeddings[split].load()
        else: # if not saved, predict
            print(f"New embedding, predicting: {split} ...")
            for data, file_names in dataloader:
                idx = [file_name.replace('.nrrd', '') for file_name in file_names]
                data_lst = self.module.model.encode(data)[0].detach().numpy().tolist()
                self.embeddings[split].stackEmbedding(idx, data_lst)
            self.embeddings[split].save()
        pass
    
    def getLabels(self, split: str = 'train'):
        assert split in self.embeddings.keys(), f'Embedding not defined for {split}, cannot get labels'
        file_lst = self.embeddings[split].embeddings['index']
        labels = self.task.getLabels(data_lst = file_lst)
        return labels
    
    def taskPrediction(self, models='all'):
        self.embed(split='train')
        self.embed(split='val')
        # format the X and Y so that they can be taken by sklearn models
        X = {'train': self.embeddings['train'].getEmbedding(), 
             'val'  : self.embeddings['val'].getEmbedding()}
        Y = {'train': np.array(self.getLabels(split='train')), 
             'val'  : np.array(self.getLabels(split='val'))}
        X['train'], Y['train'] = self.task.merge(X['train'], Y['train'])
        X['val'], Y['val'] = self.task.merge(X['val'], Y['val'])
        result_dict = predictTask(X, Y, self.task.task_type, models=models)
        self.result_dict = result_dict
        return result_dict
        
    def saveResults(self, verbose=True):
        assert self.result_dict.keys(), "result_dict not exists"
        result_log_file = os.path.join('results', '.'.join([self.log_name, str(self.version), self.task_name, 'json']))
        with open(result_log_file, 'w') as f:
            json.dump(self.result_dict, f)
        if verbose:
            print(f"Saved results to {result_log_file}")
    


class Embedding:
    
    LOG_DIR = '/home/yyhhli/code/vae/applications/logs'
    
    def __init__(self, log_name: str, version: int, split: str='train'):
        self.log_name = log_name
        self.version = version
        self.split = split
        self.embeddings = {'index':[], 'embedding':[]}
        self.save_dir = os.path.join(self.LOG_DIR, f'{self.log_name}.{self.version}')
        self.file_name = f'embedding_{self.split}.json'
        self.file_dir = os.path.join(self.save_dir, self.file_name)
        self.saved = os.path.exists(self.file_dir)
        pass
    
    def stackEmbedding(self, index_lst: List[str], embedding_lst: List[Union[List, np.ndarray]]):
        self.embeddings['index'] += index_lst
        self.embeddings['embedding'] += [list(embedding) for embedding in embedding_lst]
        pass
    
    def appendEmbedding(self, index: str, embedding: Union[List, np.ndarray]):
        self.embeddings['index'].append(index)
        self.embeddings['embedding'].append(list(embedding))
        pass
    
    def save(self, verbose=False):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        with open(self.file_dir, 'w') as f:
            json.dump(self.embeddings, f)
        if verbose:
            print(f"{len(self.embeddings['index'])} embeddings saved to {os.path.join(self.save_dir, self.file_name)}")
        pass
    
    def load(self):
        with open(self.file_dir, 'r') as f:
            data = json.load(f)
        if len(self.embeddings['index']) == 0:
            self.embeddings = data
        return data
    
    def getEmbedding(self, log_embedding=True):
        if log_embedding:
            self.save()
        return np.array(self.embeddings['embedding'])