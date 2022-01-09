""" continue to train from pretrained models """
import os
import os.path as osp
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml

from configs.parse_configs import parse_config, process_config
from experiment import VAEXperiment
from models import VAE_MODELS
from utils.custom_loggers import VAELogger  # logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train pretrained VAE models')
    parser.add_argument('--pretrain_ckpt_load_dir', '-d',
                        dest="pretrain_ckpt_load_dir",
                        help='the log directory of pretrained model checkpoint',
                        default="logs/VAE3D32AUG/version_18")
    parser.add_argument('--check_val_every_n_epoch',
                        dest="check_val_every_n_epoch",
                        help='check validation result for every n epochs',
                        default=20)
    parser.add_argument('--max_epochs',
                        dest="max_epochs",
                        help='num of epoch to train',
                        default=100)
    args = parser.parse_args()
    return args


def load_config(log_dir="logs/VAE3D32AUG/version_18"):
    with open(osp.join(log_dir, "config.yaml"), 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_ckpt(log_dir="logs/VAE3D32AUG/version_18"):
    ckpt_dir = osp.join(log_dir, "checkpoints")
    ckpt_path = osp.join(ckpt_dir, sorted(os.listdir(ckpt_dir))[-1])
    ckpt = torch.load(ckpt_path)
    print(f"loaded from {ckpt_path}")
    return ckpt


def main():
    args = parse_args()
    config = load_config(args.pretrain_ckpt_load_dir)

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
                     auto_select_gpus=True,
                     gpus=2,
                     check_val_every_n_epoch=args.check_val_every_n_epoch,
                     max_epochs=args.max_epochs)

    # experiment
    # import some of the training params
    config['exp_params']['max_epochs'] = config['trainer_params']['max_epochs']
    experiment = VAEXperiment(model, config['exp_params'])

    # loading weights
    ckpt = load_ckpt()
    experiment.load_state_dict(ckpt['state_dict'])
    print(f"======= Training {config['model_params']['name']} =======")

    # train
    runner.fit(experiment)
    pass


if __name__ == '__main__':
    main()
    pass
