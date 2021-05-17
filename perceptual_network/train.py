''' The training, evaluation and saving of perceptual network '''
import os
from typing import Dict, TypeVar

import numpy as np
import pytorch_lightning as pl
import torch
import torch.backends.cudnn as cudnn
from pytorch_lightning.callbacks import model_checkpoint

Tensor = TypeVar('torch.Tensor', bound = torch.Tensor)
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch import optim
from torch.utils.data import DataLoader

from dataset import sitk2tensor
from model import LeNet
from perceptual_dataset import LIDCPatch32VolumeDataset
from utils.custom_loggers import PerceptualLogger


class PercepturalNetwork(pl.LightningModule):
    
    def __init__(self, model, params):
        super(PercepturalNetwork, self).__init__()
        
        self.model = model
        self.params = params
        self.curr_device = None
        self.dataloader_params= {'num_workers':4, 
                                 'pin_memory':True}
        pass

    def forward(self, input: Tensor, **kwargs): #  -> Tensor
        return self.model(input, **kwargs)
    
    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        self.curr_device = imgs.device
        
        # convert labels shape
        labels = torch.reshape(labels, (-1, 1)).float()
        results = self.forward(imgs)
        loss = self.model.loss_function(results, labels)
        return {'loss': loss}
    
    def training_epoch_end(self, outputs): # log in epoch end
        avg_loss = torch.stack([l['loss'] for l in outputs]).mean()
        self.log('train_loss', avg_loss)
    
    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        
        # convert labels shape
        labels = torch.reshape(labels, (-1, 1)).float()
        results = self.forward(imgs)
        loss = self.model.loss_function(results, labels)
        return loss
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        self.log('val_loss', avg_loss)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.params['LR'])
        return optimizer
    
    def train_dataloader(self): #  -> DataLoader
        train_ds = LIDCPatch32VolumeDataset(root_dir=None, transform=sitk2tensor, split='train') # modified: using only LIDC dataset for simplicity
        self.num_train_imgs = len(train_ds)
        return DataLoader(train_ds, 
                          batch_size=self.params['batch_size'], 
                          shuffle=True, 
                          drop_last=True,
                          num_workers=4,
                          pin_memory=True)

    def val_dataloader(self):
        val_ds = LIDCPatch32VolumeDataset(root_dir=None, transform=sitk2tensor, split='val')
        self.num_val_imgs = len(val_ds)
        self.sample_dataloader = DataLoader(val_ds, 
                                            batch_size=self.params['batch_size'], 
                                            shuffle=True, 
                                            drop_last=True,
                                            num_workers=4,
                                            pin_memory=True) # let val = train in debugging mode to see if overfit
        return [self.sample_dataloader] # debug modified

    
        
def main(save_path: str, params: Dict):
    ### TRAIN/SAVE/EVAL ###
    # For reproducibility
    torch.manual_seed(9001)
    np.random.seed(9001)
    cudnn.deterministic = True
    cudnn.benchmark = False
    # define model
    model = LeNet()
    # define checkpoint
    checkpoint_callback = ModelCheckpoint(dirpath=save_path,
                                          filename='lanet-{epoch:02d}-{val_loss:.2f}',
                                          monitor='val_loss',
                                          save_top_k = 1,
                                          mode='min')
    # define logger
    p_logger = PerceptualLogger(save_dir=save_path,
                           name='lanet')
    # define trainer
    runner = Trainer(default_root_dir=save_path,
                     logger=p_logger,
                     flush_logs_every_n_steps=10, # modified
                     num_sanity_val_steps=5, 
                     callbacks=[checkpoint_callback],
                     distributed_backend='ddp',
                     gpus=2,
                     max_epochs=params['max_epochs'],
                     check_val_every_n_epoch=10)
    # define exp
    perceptural_network = PercepturalNetwork(model, params)
    # training
    runner.fit(perceptural_network)
    pass
    
    
if __name__ == '__main__':
    path = '/home/yyhhli/code/vae/perceptual_network/results'
    params = {'LR': 0.00001,
              'batch_size':32,
              'max_epochs': 30,
              'vis_interval': 10}
    main('/home/yyhhli/code/vae/perceptual_network/results', params)
