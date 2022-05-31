import inspect
from logging import Logger
import os
import os.path as osp
import re
from typing import Any, List, Union
from numpy.lib.arraysetops import isin

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
                 version: Union[int, str],
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
                 version: Union[int, str],
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
        self.name_prefix = f"{self.log_name}.{self.version}."
        self.vis_dir = vis_dir
        pass

    def generate(self, latent_vector: torch.Tensor):
        """ use decoder to generate 3D images """
        synth_imgs = self.module.model.decode(latent_vector)
        return synth_imgs

    
    def visualize(self,
                  dataloader='val_dataloader',
                  dl_params={'shuffle': False, 'drop_last': False},
                  num_batches=10):
        """ create visualizations for a dataloader """
        if (not isinstance(dataloader, int)) and ("name" not in dl_params):
            self.logger.warning(
                "didn't specify dataset name, default as \'unknown\'")
            dl_params['name'] = "unknown"
        elif isinstance(dataloader, int):
            dl_params['name'] = dataloader

        dataloader = self._parse_dataloader(dataloader, dl_params=dl_params)
        for i, (batch, file_names) in enumerate(dataloader):
            output = self.module.model.forward(batch)
            # detach
            batch, recon_batch = batch.detach(), output[0].detach()
            vis3d_tensor(img_tensor=batch,  # TODO: add names
                         save_path=osp.join(self.vis_dir,
                                            f"{self.name_prefix}_{dl_params['name']}_{str(i).zfill(2)}_image.jpeg"))
            vis3d_tensor(img_tensor=recon_batch,
                         save_path=osp.join(self.vis_dir,
                                            f"{self.name_prefix}_{dl_params['name']}_{str(i).zfill(2)}_recon.jpeg"))
            if i >= num_batches:
                return
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.visualize(*args, **kwds)


class SynthesisGaussian(ReconEvaluator):
    """ synthesize nodule patches with random gaussian values """

    def __init__(self,
                 vis_dir: str,
                 log_name: str,
                 version: int,
                 base_model_name: str = 'VAE3D',):
        super().__init__(vis_dir=vis_dir,
                         log_name=log_name,
                         version=version,
                         base_model_name=base_model_name)
        self.modes = {'random_gaussian': self.random_gaussian}
        pass

    def generate(self, latent_vector: torch.Tensor):
        """ use decoder to generate 3D images """
        synth_imgs = self.module.model.decode(latent_vector)
        return synth_imgs

    def random_gaussian(self, batch_num, seed=None):
        """ generate a batch of random gaussian vectors as latent vectors """
        batch_size = self.module.params['batch_size']
        latent_dim = self.module.model.latent_dim
        # NOTE: 0 and 1 are chosen according to kl loss
        z = torch.normal(mean=0, std=1, size=(batch_size, latent_dim))
        return z

    def synthesize(self, mode='random_gaussian', kwargs={}):
        func = self.modes[mode]
        z = func(**kwargs)
        synth_imgs = self.generate(z)
        return synth_imgs

    def synth_and_vis(self, mode='random_gaussian', kwargs={}, num_batches=1):
        for i in range(num_batches):
            synth_imgs = self.synthesize(mode=mode, kwargs=kwargs, batch_num=i)
            path = osp.join(
                self.vis_dir, f"{self.name_prefix}synth_batch_{mode}_{i}.jpeg")
            vis3d_tensor(synth_imgs, save_path=path)
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.synth_and_vis(*args, **kwds)
        pass


class SynthsisReParam(ReconEvaluator):

    def __init__(self,
                 vis_dir: str,
                 log_name: str,
                 version: int,
                 base_model_name: str = 'VAE3D'):
        super().__init__(vis_dir=vis_dir,
                         log_name=log_name,
                         version=version,
                         base_model_name=base_model_name)
        pass

    def repeat_reparam(self, mu, logvar, num_reparam=5):
        latent_list = []
        for j in range(num_reparam):
            z = self.module.model.reparameterize(mu, logvar)
            latent_list.append(z)
        latent = torch.stack(latent_list, axis=1).detach()
        latent = latent.view((latent.shape[0] * latent.shape[1],
                              latent.shape[2]))
        return latent

    def generate(self, latent_vector: torch.Tensor):
        """ use decoder to generate 3D images """
        synth_imgs = self.module.model.decode(latent_vector)
        return synth_imgs

    def reparametrize(self,
                      num_batches=1,
                      dataloader='val_dataloader',
                      dl_params={'shuffle': False, 'drop_last': False},
                      num_reparametrization=5):
        """ more re-parametrization of a single nodule """
        dataloader = self._parse_dataloader(dataloader, dl_params=dl_params)
        for i, (batch, file_names) in enumerate(dataloader):
            mu, log_var = self.module.model.encode(batch)
            self.logger.info(
                f"average std = {torch.exp(0.5 * log_var).mean().detach().numpy()}")
            # re-parametrize
            latent_vector = self.repeat_reparam(mu=mu, logvar=log_var)
            # synthesize
            img_batch = self.generate(latent_vector=latent_vector)
            # detach
            vis3d_tensor(img_tensor=img_batch,
                         nrow=num_reparametrization,
                         save_path=osp.join(self.vis_dir,
                                            f"{self.name_prefix}reparam_{str(i).zfill(2)}.jpeg"))
            if i >= num_batches:
                return
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.reparametrize(*args, **kwds)
        pass


class SynthesisRange(ReconEvaluator):

    def __init__(self,
                 vis_dir: str,
                 log_name: str,
                 version: int,
                 base_model_name: str = 'VAE3D'):
        super().__init__(vis_dir,
                         log_name,
                         version,
                         base_model_name=base_model_name)
        pass

    def reparametrize(self,
                      num_batches=1,
                      dataloader='val_dataloader',
                      dl_params={'shuffle': False, 'drop_last': False},
                      num_points=10,
                      feature_idx: Union[int, list] = 0):
        """more re-parametrization of a single nodule: setting a value range (-5, 5) * std and tune a set of features (feature_idx)
        TODO
        Args:
            num_batches (int, optional): [description]. Defaults to 1.
            dataloader (str, optional): [description]. Defaults to 'val_dataloader'.
            dl_params (dict, optional): [description]. Defaults to {'shuffle': False, 'drop_last': False}.
            num_points (int, optional): [description]. Defaults to 10.
            feature_idx (Union[int, list], optional): [can be a list or an int, if NEGATIVE, sample from (5 -> -5)*std. e.g. [-5, -6] -> sample feature num 5 and 6, but from largest to smallest values. Defaults to 0.
        """
        dataloader = self._parse_dataloader(dataloader, dl_params=dl_params)
        for i, (batch, file_names) in enumerate(dataloader):
            mu, log_var = self.module.model.encode(batch)
            self.logger.info(
                f"average std = {torch.exp(0.5 * log_var).mean().detach().numpy()}")
            # re-parametrize
            latent_vector = self.feature_range(mu=mu,
                                               logvar=log_var,
                                               idx=feature_idx,
                                               num_points=num_points)
            # synthesize
            img_batch = self.generate(latent_vector=latent_vector)
            # detach
            if isinstance(feature_idx, int):
                filename = f"{self.name_prefix}range_feature{feature_idx}_{str(i).zfill(2)}.jpeg"
            else:
                filename = f"{self.name_prefix}range_num={len(feature_idx)}_{str(i).zfill(2)}.jpeg"
            vis3d_tensor(img_tensor=img_batch,
                         nrow=num_points,
                         save_path=osp.join(self.vis_dir,
                                            filename))
            if i == num_batches - 1:
                return
        pass

    def feature_range(self,
                      mu: torch.Tensor,
                      logvar: torch.Tensor,
                      idx: Union[int, list],
                      num_points: int = 10,
                      ):
        """AI is creating summary for feature_range

        Args:
            mu (torch.Tensor): [description]
            logvar (torch.Tensor): [description]
            idx (Union[int, list]): if negative, sample inversely.
            num_points (int, optional): [description]. Defaults to 10.

        Returns:
            [type]: [description]
        """
        # idx = list
        if isinstance(idx, int):
            idx = [idx]
        pos_idx = [i for i in idx if i >= 0]
        neg_idx = [-i for i in idx if i < 0]
        # get latent variable
        latent = mu
        # std matrix
        posstd = torch.exp(0.5 * logvar[:, pos_idx])
        negstd = torch.exp(0.5 * logvar[:, neg_idx])
        std_linspace = torch.linspace(start=-5, end=5, steps=num_points)
        # std_matrix = torch.outer(std, std_linspace)
        posstd_matrix = (posstd.unsqueeze(2).repeat(1, 1, num_points)
                         * std_linspace).permute(0, 2, 1)
        negstd_matrix = (negstd.unsqueeze(2).repeat(1, 1, num_points)
                         * -std_linspace).permute(0, 2, 1)  # negative of std_linspace

        # add std
        latent = latent.unsqueeze(1).repeat(1, num_points, 1)
        latent[:, :, pos_idx] = latent[:, :, pos_idx] + posstd_matrix
        latent[:, :, neg_idx] = latent[:, :, neg_idx] + negstd_matrix
        return latent

    def generate(self, latent_vector: torch.Tensor):
        """ use decoder to generate 3D images """
        synth_imgs = self.module.model.decode(latent_vector)
        return synth_imgs

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.reparametrize(*args, **kwds)
        pass
