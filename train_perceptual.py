""" train perceptual network """


import os
from typing import Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from configs.config_vars import BASE_DIR
from perceptual_network.model import LeNet
from perceptual_network.plmodule import PercepturalNetwork
from utils.custom_loggers import PerceptualLogger


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
                                          save_top_k=1,
                                          mode='min')
    # define logger
    p_logger = PerceptualLogger(save_dir=save_path,
                                name='lanet')
    # define trainer
    runner = Trainer(default_root_dir=save_path,
                     logger=p_logger,
                     flush_logs_every_n_steps=10,  # modified
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
    PATH = os.path.join(BASE_DIR, 'perceptual_network/results')
    params = {'LR': 0.00001,
              'batch_size': 32,
              'max_epochs': 30,
              'vis_interval': 10,
              'dataset': 'LIDCPatchAugDataset'}
    main(PATH, params)
