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

from experiment import VAEXperiment
from models import VAE_MODELS
from models.vae_base import VAEBackbone
from train import process_config
from utils.custom_loggers import VAELogger, get_logger  # logger

LOGGER = get_logger()


def parse_config():
    # arg parser
    parser = argparse.ArgumentParser(description='Train VAE models')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help='config file name in /configs folder',
                        default='exp_train_v2/lidc->stf_4e-6_step5000')
    parser.add_argument('--note', "-N",
                        dest="note",
                        help='any note for training, will be saved in config file',
                        default="")
    parser.add_argument("--info", "-I",
                        dest="info",
                        help="flag to output information but not train",
                        action="store_true")
    parser.add_argument("--template", "-T",
                        dest="template",
                        help="flag to output template",
                        default="template_finetune")
    args = parser.parse_args()
    config = process_config(args.filename, template=args.template)
    config['note'] = args.note
    if args.info:
        config['info'] = True
    return config


# def load_config(log_dir="logs/VAE3D32AUG/version_57"):
#     with open(osp.join(log_dir, "config.yaml"), 'r') as file:
#         config = yaml.safe_load(file)
#     return config


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
    # parse args and config dict
    # args = parse_args()
    # config = parse_config(args.config)
    config = parse_config()

    # config["logging_params"]['pretrain_ckpt_load_dir'] = args.pretrain_ckpt_load_dir
    # if args.name != "same":
    #     config['logging_params']['name'] = args.name

    # change training dataset
    # config['exp_params']['dataset'] = args.dataset
    # change LR
    # if args.learning_rate:
    #     config['exp_params']['LR'] = float(args.learning_rate)

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

    print("early_stopping:", config['exp_params']['early_stopping'])  # debug

    callbacks = [model_checkpoint] if config['exp_params']['early_stopping'] == "False" else [
        model_checkpoint, early_stopping]

    # trainer
    runner = Trainer(default_root_dir=f"{vae_logger.save_dir}",
                     logger=vae_logger,
                     # specify callback
                     callbacks=callbacks,
                     num_sanity_val_steps=100,
                     accelerator="gpu",
                     auto_select_gpus=True,
                     devices=1,
                     **config['trainer_params'])

    # experiment
    # import some of the training params
    config['exp_params']['max_epochs'] = config['trainer_params']['max_epochs']
    config['exp_params']['max_steps'] = config['trainer_params']['max_steps']
    experiment = VAEXperiment(model, config['exp_params'])

    # loading weights
    ckpt = load_ckpt(config['pretrain_ckpt_load_dir'])
    experiment.load_state_dict(ckpt['state_dict'])
    print(f"======= Training {config['model_params']['name']} =======")

    # freeze some layers
    # don't freeze the decoder!
    # freeze_params = {"encoder": [0, 1]}
    experiment.model = freeze_model(experiment.model,
                                    freeze_params=config['freeze_params'])

    # train
    runner.fit(experiment)
    pass


if __name__ == '__main__':
    main()
    pass
