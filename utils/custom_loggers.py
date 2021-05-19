from typing import Any, Dict, Optional, Union
from pytorch_lightning.loggers import CSVLogger, TestTubeLogger
from pytorch_lightning.utilities import _module_available
import pandas as pd

import os
from .funcs import getVersion, saveConfig
from .visualization import vis_loss_curve, vis_loss_curve_diff_scale

_TESTTUBE_AVAILABLE = _module_available("test_tube")

if _TESTTUBE_AVAILABLE:
    from test_tube import Experiment
else:
    Experiment = None


class VAELogger(TestTubeLogger):

    def __init__(self,
                 save_dir: str,
                 name: str,
                 description: Optional[str] = '',
                 debug: bool = False,
                 version: Optional[int] = None,
                 create_git_tag: bool = False,
                 log_graph: bool = False,
                 prefix: str = '',
                 vis_interval=10,
                 config_file=None):
        super().__init__(save_dir,
                         name=name,
                         description=description,
                         debug=debug,
                         version=version,
                         create_git_tag=create_git_tag,
                         log_graph=log_graph,
                         prefix=prefix)

        self.vis_interval = vis_interval
        self.config_file = config_file
        self._version = getVersion(os.path.join(self._save_dir, self._name))

    def draw_loss_curve(self):
        rel_log_path = os.path.join(
            self._save_dir, self._name, f'version_{self.version}')
        abs_log_path = os.path.join(os.getcwd(), rel_log_path)
        log = pd.read_csv(os.path.join(abs_log_path, 'metrics.csv'))

        train_loss_df = log[['epoch', 'loss']].dropna()
        val_loss_df = log[['epoch', 'val_loss']].dropna()

        train_loss = list(train_loss_df['loss'])
        val_loss = list(val_loss_df['val_loss'])

        train_loss_dict = {'epoch': list(train_loss_df['epoch']),
                           'loss': train_loss}
        val_loss_dict = {'epoch': list(val_loss_df['epoch']),
                         'loss': val_loss}

        vis_loss_curve(log_path=abs_log_path,
                       data={'train': train_loss_dict, 'val': val_loss_dict})

        pass

    def draw_kl_recon_loss(self):
        rel_log_path = os.path.join(
            self._save_dir, self._name, f'version_{self.version}')
        abs_log_path = os.path.join(os.getcwd(), rel_log_path)
        log = pd.read_csv(os.path.join(abs_log_path, 'metrics.csv'))

        recon_loss_df = log[['epoch', 'Reconstruction_Loss']].dropna()
        kl_loss_df = log[['epoch', 'KLD']].dropna()

        recon_loss_dict = {'epoch': list(recon_loss_df['epoch']),
                           'loss': list(recon_loss_df['Reconstruction_Loss'])}
        kl_loss_dict = {'epoch': list(kl_loss_df['epoch']),
                        'loss': list(kl_loss_df['KLD'])}

        vis_loss_curve_diff_scale(log_path=os.path.join(abs_log_path,
                                                        "loss_curve_kl_recon.jpeg"),
                                  data={'recon loss': recon_loss_dict,
                                        'kl loss': kl_loss_dict})
        pass

    def finalize(self, status: str):  # -> None
        super().finalize(status)
        # TODO: HACK figure out where this is being set to true
        self.experiment.debug = self.debug
        log_dir = os.path.join(self._save_dir, self._name,
                               f'version_{self.version}')
        self.draw_loss_curve()
        self.draw_kl_recon_loss()
        saveConfig(log_dir, self.config_file)
        self.save()
        self.close()


class PerceptualLogger(CSVLogger):

    def __init__(self,
                 save_dir: str,
                 name: str
                 ):
        super().__init__(save_dir,
                         name=name)
        self._version = getVersion(os.path.join(self._save_dir, self._name))

    def draw_loss_curve(self):
        log = pd.read_csv(self.experiment.metrics_file_path)

        train_loss_df = log[['epoch', 'train_loss']].dropna()
        val_loss_df = log[['epoch', 'val_loss']].dropna()

        train_loss = list(train_loss_df['train_loss'])
        val_loss = list(val_loss_df['val_loss'])

        train_loss_dict = {'epoch': list(
            train_loss_df['epoch']), 'loss': train_loss}
        val_loss_dict = {'epoch': list(val_loss_df['epoch']), 'loss': val_loss}

        vis_loss_curve(log_path=self.experiment.log_dir,
                       data={'training loss': train_loss_dict, 'validation loss': val_loss_dict})

        pass

    def finalize(self, status: str):  # -> None
        super().finalize(status)
        self.save()
        self.draw_loss_curve()
        self.close()
