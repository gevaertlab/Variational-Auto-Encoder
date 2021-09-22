''' The training, evaluation and saving of perceptual network '''
import os
from typing import Dict, TypeVar

import numpy as np
import pytorch_lightning as pl
import torch
import torch.backends.cudnn as cudnn
from configs.config_vars import BASE_DIR
from datasets import REGISTERED_DATASETS
from datasets.utils import sitk2tensor
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import model_checkpoint
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch import optim
from torch.utils.data import DataLoader
from utils.custom_loggers import PerceptualLogger

from .model import LeNet

Tensor = TypeVar('Tensor', bound=torch.Tensor)


class PercepturalNetwork(pl.LightningModule):

    def __init__(self, model, params):
        super(PercepturalNetwork, self).__init__()

        self.model = model
        self.params = params
        self.curr_device = None
        self.dataloader_params = {'num_workers': 4,
                                  'pin_memory': True}
        self.dataset = REGISTERED_DATASETS[params['dataset']]
        pass

    def forward(self, input: Tensor, **kwargs):  # -> Tensor
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        self.curr_device = imgs.device

        # convert labels shape
        labels = torch.reshape(labels, (-1, 1)).float()
        results = self.forward(imgs)
        loss = self.model.loss_function(results, labels)
        return {'loss': loss}

    def training_epoch_end(self, outputs):  # log in epoch end
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

    def train_dataloader(self):  # -> DataLoader
        # modified: using only LIDC dataset for simplicity
        train_ds = self.dataset(root_dir=None,
                                transform=sitk2tensor,
                                split='train')
        self.num_train_imgs = len(train_ds)
        return DataLoader(train_ds,
                          batch_size=self.params['batch_size'],
                          shuffle=True,
                          drop_last=True,
                          num_workers=4,
                          pin_memory=True)

    def val_dataloader(self):
        val_ds = self.dataset(root_dir=None,
                              transform=sitk2tensor,
                              split='val')
        self.num_val_imgs = len(val_ds)
        self.sample_dataloader = DataLoader(val_ds,
                                            batch_size=self.params['batch_size'],
                                            shuffle=True,
                                            drop_last=True,
                                            num_workers=4,
                                            pin_memory=True)  # let val = train in debugging mode to see if overfit
        return [self.sample_dataloader]  # debug modified


