import argparse
import os

import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
import torch.backends.cudnn as cudnn
import yaml
import torch

from experiment import VAEXperiment
from models import *
from utils.visualization import visLossCurve

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae_debug.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    config = yaml.safe_load(file)


tt_logger = TestTubeLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['name'],
    description=str(config['logging_params']['visualize_interval']),
    debug=False,
    create_git_tag=False,
)

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model, config['exp_params'])

runner = Trainer(default_save_path=f"{tt_logger.save_dir}", # same as config['logging_params']['save_dir'] # modified
                 min_nb_epochs=1, # modified
                 logger=tt_logger,
                 log_save_interval=100,
                 train_percent_check=1.,
                 val_percent_check=1.,
                 num_sanity_val_steps=5,
                 early_stop_callback = False,
                 **config['trainer_params'])

print(f"======= Training {config['model_params']['name']} =======")
# train
runner.fit(experiment)
# visualize loss curve
visLossCurve(f'{experiment.logger.save_dir}{experiment.logger.name}/version_{experiment.logger.version}')
# save config file
with open(os.path.join(os.getcwd(), 
                       config['logging_params']['save_dir'], 
                       str(experiment.logger.name),
                       'version_'+str(experiment.logger.version), 
                       'config.yaml'), 'w') as f:
    yaml.dump(config, f)
