import inspect
import os
import os.path as osp
import re
from typing import Any, List

import torch
import yaml
from configs.config_vars import BASE_DIR
from experiment import VAEXperiment
from models import VAE_MODELS
from tqdm import tqdm
# from utils.custom_metrics import SSIM, MSE, PSNR
from utils import custom_metrics
from utils.io import mkdir_safe
from utils.python_logger import get_logger
from utils.visualization import vis3d_tensor

METRICS_FUNCS = inspect.getmembers(custom_metrics, inspect.isfunction)
METRICS_DICT = dict(METRICS_FUNCS)


class BaseEvaluator:
    """ load weight and init model as well as dataset for prediction and stuff """

    LOG_DIR = osp.join(BASE_DIR, "logs")

    def __init__(self,
                 log_name: str,
                 version: int,
                 base_model_name: str = 'VAE3D',
                 ):
        self.log_name = log_name
        self.version = version
        self.base_model_name = base_model_name
        self.load_dir = osp.join(self.LOG_DIR,
                                 log_name,
                                 f'version_{version}')
        self._config = self._load_config(self.load_dir)
        self.module = self._init_model(self.load_dir,
                                       self.base_model_name)
        self.logger = get_logger(cls_name=self.__class__.__name__)
        pass

    def _load_config(self, load_dir):
        config_path = os.path.join(load_dir, 'config.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        # NOTE: HACK!!! the config file will have the hidden_dims reversed,
        # so reverse it to have the same model
        # assume that the original is from small to large
        config['model_params']['hidden_dims'].sort()
        return config

    def _init_model(self, base_dir, base_model_name):
        vae_model = VAE_MODELS[base_model_name](**self._config['model_params'])
        ckpt_path = self._getckptpath(base_dir)  # implement this
        return VAEXperiment.load_from_checkpoint(ckpt_path,
                                                 vae_model=vae_model,
                                                 params=self._config['exp_params'])

    @staticmethod
    def _getckptpath(base_dir):
        # get checkpoint path of the max epoch saved model
        ckpt_path = osp.join(base_dir, 'checkpoints')
        files = os.listdir(ckpt_path)
        if len(files) > 1:
            epochs = [int(re.findall('\d+', f)[0]) for f in files]
            return osp.join(ckpt_path, files[epochs.index(max(epochs))])
        else:
            return osp.join(ckpt_path, files[0])

    def _parse_dataloader(self, dataloader, dl_params=None):
        if not dl_params:
            dl_params = {'shuffle': False, 'drop_last': False}
        if isinstance(dataloader, str):
            dataloader = getattr(self.module, dataloader)(**dl_params)
            if isinstance(dataloader, List):
                dataloader = dataloader[0]
        elif isinstance(dataloader, torch.utils.data.DataLoader):
            # input dataloader instance
            dataloader = dataloader
        else:
            raise NotImplementedError("[EmbeddingPredictor] \
                Not supported dataloder type: \'{type(dataloader)}\'")
        return dataloader


class MetricEvaluator(BaseEvaluator):
    """ calculate more custom metrics """

    def __init__(self,
                 metrics: List,
                 log_name: str,
                 version: int,
                 base_model_name: str = 'VAE3D',):
        super().__init__(log_name=log_name,
                         version=version,
                         base_model_name=base_model_name)

        try:
            self.metrics = {name: METRICS_DICT[name] for name in metrics}
        except KeyError as e:
            print(
                e, f"[MetricEvaluator] Failed to find metric name {metrics}. Available metrics: {METRICS_DICT.keys}")
        pass

    def calc_metrics(self,
                     dataloader='val_dataloader',
                     dl_params={'shuffle': False, 'drop_last': False}):
        dataloader = self._parse_dataloader(dataloader, dl_params=dl_params)
        metrics_dict = {name: [] for name in self.metrics.keys()}
        for batch, file_names in tqdm(dataloader):
            output = self.module.model.forward(batch)
            # detach
            batch, recon_batch = batch.detach(), output[0].detach()

            for i in range(batch.shape[0]):
                img, recon = batch[i, 0, ::], recon_batch[i, 0, ::]

                for name, func in self.metrics.items():
                    result = func(img, recon)
                    metrics_dict[name].append(result)

        return metrics_dict

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.calc_metrics(*args, **kwds)


class ReconEvaluator(BaseEvaluator):
    """ visualize reconstructed patches """

    def __init__(self,
                 vis_dir: str,
                 log_name: str,
                 version: int,
                 base_model_name: str = 'VAE3D',):
        super().__init__(log_name=log_name,
                         version=version,
                         base_model_name=base_model_name)
        mkdir_safe(vis_dir)
        self.vis_dir = vis_dir
        pass

    def visualize(self,
                  dataloader='val_dataloader',
                  dl_params={'shuffle': False, 'drop_last': False},
                  num_batches=10):
        dataloader = self._parse_dataloader(dataloader, dl_params=dl_params)
        name_prefix = f"{self.log_name}.{self.version}."
        for i, (batch, file_names) in enumerate(dataloader):
            output = self.module.model.forward(batch)
            # detach
            batch, recon_batch = batch.detach(), output[0].detach()
            vis3d_tensor(img_tensor=batch,
                         save_path=osp.join(self.vis_dir,
                                            f"{name_prefix}{str(i).zfill(2)}_image.jpeg"))
            vis3d_tensor(img_tensor=recon_batch,
                         save_path=osp.join(self.vis_dir,
                                            f"{name_prefix}{str(i).zfill(2)}_recon.jpeg"))
            if i >= num_batches:
                return
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.visualize(*args, **kwds)
