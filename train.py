""" This file is implements the pytorch lightning module for 3D VAE """

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from configs.parse_configs import parse_config
from experiment import VAEXperiment
from models import VAE_MODELS
from utils.custom_loggers import VAELogger  # logger

if __name__ == '__main__':

    config = parse_config()

    vae_logger = VAELogger(
        save_dir=config['logging_params']['save_dir'],
        name=config['logging_params']['name'],
        config_file=config,
        vis_interval=config['logging_params']['visualize_interval']
    )

    # For reproducibility
    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    cudnn.deterministic = True
    cudnn.benchmark = False

    # model
    model = VAE_MODELS[config['model_params']
                       ['name']](**config['model_params'])

    # callback
    callback = ModelCheckpoint(monitor='val_loss',  # if not specified, default save dir
                               save_top_k=1,
                               mode='min')

    # trainer
    runner = Trainer(default_root_dir=f"{vae_logger.save_dir}",
                     logger=vae_logger,
                     callbacks=callback,  # specify callback
                     flush_logs_every_n_steps=10,
                     num_sanity_val_steps=5,
                     distributed_backend='ddp',
                     gpus=2,
                     **config['trainer_params'])

    # experiment
    # import some of the training params
    config['exp_params']['max_epochs'] = config['trainer_params']['max_epochs']
    experiment = VAEXperiment(model, config['exp_params'])

    print(f"======= Training {config['model_params']['name']} =======")

    # train
    runner.fit(experiment)
