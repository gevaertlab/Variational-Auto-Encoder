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
from datasets import PATCH_DATASETS
from experiment import VAEXperiment
from models import VAE_MODELS
from models.vae_base import VAEBackbone
from utils.custom_loggers import VAELogger, get_logger  # logger
LOGGER = get_logger()


def parse_args():
    parser = argparse.ArgumentParser(description='Train pretrained VAE models')
    parser.add_argument('--pretrain_ckpt_load_dir', '-d',
                        dest="pretrain_ckpt_load_dir",
                        help='the log directory of pretrained model checkpoint',
                        default="logs/VAE3D32AUG/version_57")
    parser.add_argument('--dataset',
                        dest="dataset",
                        help=f'dataset name to train on, patch_datasets: {PATCH_DATASETS.keys()}',
                        default="StanfordRadiogenomicsPatchAugDataset")  # LNDbPatch32Dataset
    parser.add_argument('--learning_rate', '-lr',
                        dest="learning_rate",
                        help='float; learning rate to train, typically smaller, 0 means the same',
                        default=0)  # LNDbPatch32Dataset

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


def load_config(log_dir="logs/VAE3D32AUG/version_57"):
    with open(osp.join(log_dir, "config.yaml"), 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_ckpt(log_dir="logs/VAE3D32AUG/version_18"):
    ckpt_dir = osp.join(log_dir, "checkpoints")
    ckpt_path = osp.join(ckpt_dir, sorted(os.listdir(ckpt_dir))[-1])
    ckpt = torch.load(ckpt_path)
    LOGGER.info(f"loaded from {ckpt_path}")
    return ckpt


def freeze_model(model: VAEBackbone, freeze_params: dict):
    for name, layer_list in freeze_params.items():
        part = model.__getattr__(name)
        for l in layer_list:
            for pname, param in part[l].named_parameters():
                param.requires_grad = False
    return model


def main():
    # parse args
    args = parse_args()
    config = load_config(args.pretrain_ckpt_load_dir)
    config['note'] = args.note
    # change training dataset
    config['exp_params']['dataset'] = args.dataset
    # change LR
    if args.learning_rate:
        config['exp_params']['LR'] = float(args.learning_rate)

    # logger
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
                                   min_delta=0.0,
                                   patience=4,
                                   verbose=True,
                                   mode="min")

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
    ckpt = load_ckpt(args.pretrain_ckpt_load_dir)
    experiment.load_state_dict(ckpt['state_dict'])
    print(f"======= Training {config['model_params']['name']} =======")

    # freeze some layers
    # len encoder = 4, len decoder = 3
    freeze_params = {"encoder": [0, 1], "decoder": [2]}
    experiment.model = freeze_model(
        experiment.model, freeze_params=freeze_params)

    # train
    runner.fit(experiment)
    pass


if __name__ == '__main__':
    main()
    pass
