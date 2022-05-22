import datetime
import logging
import os
from readline import set_pre_input_hook
from typing import Optional

import pandas as pd
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.utilities import _module_available

from .funcs import getVersion, saveConfig
from .python_logger import get_logger
from .visualization import vis_loss_curve, vis_loss_curve_diff_scale

LOGGER = get_logger()

_TESTTUBE_AVAILABLE = _module_available("test_tube")

if _TESTTUBE_AVAILABLE:
    from test_tube import Experiment
else:
    Experiment = None


class VAELogger(CSVLogger):

    def __init__(self,
                 save_dir: str,
                 name: str,
                 #  description: Optional[str] = '',
                 debug: bool = False,
                 version: Optional[int] = None,
                 #  create_git_tag: bool = False,
                 log_graph: bool = False,
                 prefix: str = '',
                 vis_interval=10,
                 config_file=None):
        super().__init__(save_dir=save_dir,
                         name=name,
                         version=version,
                         flush_logs_every_n_steps=vis_interval,
                         prefix=prefix)

        self.vis_interval = vis_interval
        self.config_file = config_file
        self.debug = debug
        if version is None:
            self._version = getVersion(
                os.path.join(self.save_dir, self.name))
        else:
            self._version = version

        # combining tensorboard and csv logger
        self.tb_logger = TensorBoardLogger(save_dir,
                                           name=name,
                                           #  description=description,
                                           #  debug=debug,
                                           version=self._version,
                                           #  create_git_tag=create_git_tag,
                                           default_hp_metric=False,
                                           sub_dir="tb_log",
                                           log_graph=log_graph,
                                           prefix=prefix)
        self.log_file = None
        self.abs_log_path = None

    def log_hyperparams(self, params):
        super().log_hyperparams(params)
        self.tb_logger.log_hyperparams(params)
        pass

    def log_metrics(self, metrics, step=None):
        super().log_metrics(metrics, step=step)
        self.tb_logger.log_metrics(metrics, step=step)
        pass

    def log_metrics_from_log_file(self, metrics, step=None):
        super().log_metrics(metrics, step=step)
        self.tb_logger.log_metrics(metrics, step=step)
        pass

    def save(self):
        # save config_file
        # log_dir = os.path.join(os.getcwd(), self.save_dir, self.name,
        #                        f'version_{self.version}')
        # saveConfig(log_dir, self.config_file)
        super().save()
        self.tb_logger.save()
        pass

    def _get_vis_loss_dict(self,
                           log: pd.DataFrame,
                           col_name: str,
                           per: str = "step"):
        # get dict of epochs and losses
        df = log[[per, col_name]].dropna()
        value = list(df[col_name])
        per = list(df[per])
        return {f'{per}': per,
                'value': value}

    def _load_log_file(self):
        # renew log file each time calling this function

        rel_log_path = os.path.join(self.save_dir,
                                    self.name,
                                    f'version_{self.version}')
        abs_log_path = os.path.join(os.getcwd(), rel_log_path)
        self.abs_log_path = abs_log_path
        log_file = pd.read_csv(os.path.join(abs_log_path, 'metrics.csv'))
        self.log_file = log_file
        return self.log_file

    def draw_loss_curve(self):
        log = self._load_log_file()
        try:
            train_loss_dict = self._get_vis_loss_dict(log, 'loss')
            val_loss_dict = self._get_vis_loss_dict(log, 'val_loss')

            vis_loss_curve(log_path=self.abs_log_path,
                           data={'train': train_loss_dict, 'val': val_loss_dict})
        except Exception as e:
            LOGGER.warning(e)
        pass

    def draw_kl_recon_loss(self):
        log = self._load_log_file()
        try:
            recon_loss_dict = self._get_vis_loss_dict(
                log, 'Reconstruction_Loss')
            kl_loss_dict = self._get_vis_loss_dict(log, 'KLD')

            vis_loss_curve_diff_scale(log_path=self.abs_log_path,
                                      data={'recon loss': recon_loss_dict,
                                            'kl loss': kl_loss_dict})
        except Exception as e:
            LOGGER.warning(e)
        pass

    def draw_multiple_loss_curves(self):
        log = self._load_log_file()
        # define loss dict {"name to be plotted": "name in log file"}
        mandatory = {'train loss': 'loss', 'val loss': 'val_loss'}
        optional_keys = {'recon loss': 'Reconstruction_Loss', 
        'kl loss': 'KLD', 
        'learning rate':'lr', 
        "perceptual loss": 'perceptual_loss'}
        mloss_dict = {}
        loss_dict = {}
        for pname, lname in mandatory.items():
            mloss_dict[pname] = self._get_vis_loss_dict(log, lname)
        for pname, lname in optional_keys.items():
            if lname in log.columns:
                loss_dict[pname] = lname
        loss_dict['train val losses'] = mloss_dict
        vis_loss_curve_diff_scale(log_path=self.abs_log_path,
                                  data=loss_dict,
                                  name="diagnostic_loss_curve.jpeg")
        # pass



        # try:
        #     recon_loss_dict = self._get_vis_loss_dict(
        #         log, 'Reconstruction_Loss')
        #     kl_loss_dict = self._get_vis_loss_dict(log, 'KLD')
        #     train_loss_dict = self._get_vis_loss_dict(log, 'loss')
        #     val_loss_dict = self._get_vis_loss_dict(log, 'val_loss')
        #     lr_dict = self._get_vis_loss_dict(log, 'lr')
        #     perceptual_loss_dict = self._get_vis_loss_dict(log, 'perceptual_loss')

        #     vis_loss_curve_diff_scale(log_path=self.abs_log_path,
        #                               data={'train val losses': [{'train loss': train_loss_dict,
        #                                                           'val loss': val_loss_dict}],
        #                                     'recon loss': recon_loss_dict,
        #                                     'kl loss': kl_loss_dict,
        #                                     'learning rate': lr_dict,
        #                                     "perceptual loss": perceptual_loss_dict},
        #                               name="diagnostic_loss_curve.jpeg")
        # except Exception as e:
        #     LOGGER.warning(e)

    def add_notes(self):
        # add notes to config_file
        # 1. add time used
        log = self._load_log_file()
        if not "created_at" in log.columns:
            pass
        else:
            time_col = self._get_vis_loss_dict(log, "created_at")
            time_col['loss'] = [datetime.datetime.strptime(
                tstr, '%Y-%m-%d %H:%M:%S.%f') for tstr in time_col['loss']]
            time_used = str(max(time_col['loss']) -
                            min(time_col['loss'])).split('.')[0]
            print(f"time used: {time_used}")
            self.config_file['trainer_params']['time_used'] = time_used
            pass

    def finalize(self, status: str):  # -> None

        self.experiment.debug = self.debug
        log_dir = os.path.join(self.save_dir, self.name,
                               f'version_{self.version}')
        self.draw_loss_curve()
        self.draw_kl_recon_loss()
        self.draw_multiple_loss_curves()
        self.add_notes()
        saveConfig(log_dir, self.config_file)

        super().finalize(status)
        self.tb_logger.finalize(status)
        self.save()
        pass


class PerceptualLogger(CSVLogger):

    def __init__(self,
                 save_dir: str,
                 name: str
                 ):
        super().__init__(save_dir,
                         name=name)
        self.save_dir = save_dir
        self.name = name
        self._version = getVersion(os.path.join(self.save_dir, self.name))

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
        pass
