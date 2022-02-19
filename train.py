""" This file is implements the pytorch lightning module for 3D VAE """

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from configs.parse_configs import parse_config, process_config
from experiment import VAEXperiment
from models import VAE_MODELS
from utils.custom_loggers import VAELogger  # logger


def main(config_name=None):
    if config_name:
        config = process_config(config_name)
    else:
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

    # callback
    model_checkpoint = ModelCheckpoint(monitor='val_loss',  # if not specified, default save dir
                                       save_top_k=1,
                                       mode='min')

    early_stopping = EarlyStopping(monitor="val_loss",
                                   min_delta=0.0,
                                   patience=2,
                                   verbose=True,
                                   mode="min")

    # trainer
    runner = Trainer(default_root_dir=f"{vae_logger.save_dir}",
                     logger=vae_logger,
                     # specify callback
                     callbacks=[model_checkpoint, early_stopping],
                     flush_logs_every_n_steps=10,
                     num_sanity_val_steps=100,
                     distributed_backend='ddp',
                     auto_select_gpus=True,
                     gpus=1,  # NOTE: training stucked, see https://github.com/PyTorchLightning/pytorch-lightning/issues/5865
                     **config['trainer_params'])

    # experiment
    # import some of the training params
    config['exp_params']['max_epochs'] = config['trainer_params']['max_epochs']
    experiment = VAEXperiment(config['model_params'], config['exp_params'])

    if "info" in config:
        experiment.verbose_info()
        pass
    else:
        print(f"======= Training {config['model_params']['name']} =======")

        # train
        runner.fit(experiment)
        pass


if __name__ == '__main__':
    main()
    pass
