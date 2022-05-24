""" This file is implements the pytorch lightning module for 3D VAE """

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from configs.parse_configs import parse_config, process_config
from experiment import VAEXperiment
from utils.custom_loggers import VAELogger  # logger
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.plugins import DDPPlugin


def main(config_name=None):
    # script config name override arg name..
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

    print("early_stopping:", config['exp_params']['early_stopping'])  # debug

    callbacks = [model_checkpoint] if config['exp_params']['early_stopping'] == "False" else [
        model_checkpoint, early_stopping]

    # experiment
    # import some of the training params
    config['exp_params']['max_epochs'] = config['trainer_params']['max_epochs']
    config['exp_params']['max_steps'] = config['trainer_params']['max_steps']
    experiment = VAEXperiment(config['model_params'], config['exp_params'])

    # trainer
    # NOTE: training stucked, see https://github.com/PyTorchLightning/pytorch-lightning/issues/5865
    # see https://stackoverflow.com/questions/68000761/pytorch-ddp-finding-the-cause-of-expected-to-mark-a-variable-ready-only-once
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/9242#issuecomment-951820434
    # dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # ddp = DDP(module=experiment, find_unused_parameters=False)

    runner = Trainer(default_root_dir=f"{vae_logger.save_dir}",
                     logger=vae_logger,
                     # specify callback
                     callbacks=callbacks,
                     num_sanity_val_steps=100,
                     #  strategy=DDPStrategy(),
                     accelerator="gpu",
                     #  strategy="ddp",  # DDPPlugin(find_unused_parameters=False)
                     #  auto_select_gpus=True,
                     devices=1,
                     auto_scale_batch_size=True,
                     #  accelerator="ddp",
                     **config['trainer_params'])

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
