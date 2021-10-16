import logging
from typing import Optional
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
        if version is None:
            self._version = getVersion(
                os.path.join(self._save_dir, self._name))
        else:
            self._version = version
        self.log_file = None
        self.abs_log_path = None

    def _get_vis_loss_dict(self,
                           log: pd.DataFrame,
                           col_name: str):
        df = log[['epoch', col_name]].dropna()
        loss = list(df[col_name])
        epoch = list(df['epoch'])
        return {'epoch': epoch,
                'loss': loss}

    def _load_log_file(self):
        # renew log file each time calling this function

        rel_log_path = os.path.join(self._save_dir,
                                    self._name,
                                    f'version_{self.version}')
        abs_log_path = os.path.join(os.getcwd(), rel_log_path)
        self.abs_log_path = abs_log_path
        log_file = pd.read_csv(os.path.join(abs_log_path, 'metrics.csv'))
        self.log_file = log_file
        return self.log_file

    def draw_loss_curve(self):
        log = self._load_log_file()

        train_loss_dict = self._get_vis_loss_dict(log, 'loss')
        val_loss_dict = self._get_vis_loss_dict(log, 'val_loss')

        vis_loss_curve(log_path=self.abs_log_path,
                       data={'train': train_loss_dict, 'val': val_loss_dict})

        pass

    def draw_kl_recon_loss(self):
        log = self._load_log_file()

        recon_loss_dict = self._get_vis_loss_dict(log, 'Reconstruction_Loss')
        kl_loss_dict = self._get_vis_loss_dict(log, 'KLD')

        vis_loss_curve_diff_scale(log_path=self.abs_log_path,
                                  data={'recon loss': recon_loss_dict,
                                        'kl loss': kl_loss_dict})
        pass

    def draw_multiple_loss_curves(self):
        log = self._load_log_file()

        recon_loss_dict = self._get_vis_loss_dict(log, 'Reconstruction_Loss')
        kl_loss_dict = self._get_vis_loss_dict(log, 'KLD')
        train_loss_dict = self._get_vis_loss_dict(log, 'loss')
        val_loss_dict = self._get_vis_loss_dict(log, 'val_loss')
        lr_dict = self._get_vis_loss_dict(log, 'lr')

        vis_loss_curve_diff_scale(log_path=self.abs_log_path,
                                  data={'train val losses': [{'train loss': train_loss_dict,
                                                              'val loss': val_loss_dict}],
                                        'recon loss': recon_loss_dict,
                                        'kl loss': kl_loss_dict,
                                        'learning rate': lr_dict},
                                  name="diagnostic_loss_curve.jpeg")

    def finalize(self, status: str):  # -> None
        super().finalize(status)
        self.experiment.debug = self.debug
        log_dir = os.path.join(self._save_dir, self._name,
                               f'version_{self.version}')
        self.draw_loss_curve()
        self.draw_kl_recon_loss()
        self.draw_multiple_loss_curves()
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

