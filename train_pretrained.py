""" continue to train from pretrained models """
import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from configs.parse_configs import parse_config, process_config
from experiment import VAEXperiment
from models import VAE_MODELS
from utils.custom_loggers import VAELogger  # logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train pretrained VAE models')
    parser.add_argument('--pretrain_ckpt_load_dir', '-d',
                        dest="pretrain_ckpt_load_dir",
                        help='the log directory of pretrained model checkpoint',
                        default="logs/VAE3D32AUG/version_33")
    parser.add_argument('--dataset',
                        dest="dataset",
                        help='dataset name to train on',
                        default="StanfordRadiogenomicsPatchDataset")  # LNDbPatch32Dataset
    parser.add_argument('--max_epochs',
                        dest="max_epochs",
                        help='num of epoch to train',
                        default=1000)
    parser.add_argument('--note',
                        dest="note",
                        help='any note for training, will be saved in config file',
                        default="")
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
    config['note'] = args.note
    # change training dataset
    config['exp_params']['dataset'] = args.dataset

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

    # callbacks
    model_checkpoint = ModelCheckpoint(monitor='val_loss',  # if not specified, default save dir
                                       save_top_k=1,
                                       mode='min')

    early_stopping = EarlyStopping(monitor="val_loss",
                                   min_delta=0.00,
                                   patience=10,
                                   verbose=True,
                                   mode="auto")

    # trainer
    runner = Trainer(default_root_dir=f"{vae_logger.save_dir}",
                     logger=vae_logger,
                     # specify callback
                     callbacks=[model_checkpoint, early_stopping],
                     flush_logs_every_n_steps=10,
                     num_sanity_val_steps=5,
                     distributed_backend='ddp',
                     auto_select_gpus=True,
                     gpus=1,  # debug
                     check_val_every_n_epoch=config['trainer_params']['check_val_every_n_epoch'],
                     max_epochs=int(args.max_epochs))

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
