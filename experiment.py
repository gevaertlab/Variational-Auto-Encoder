''' Defines a torch_lightning Module '''
import os
from typing import Union
import inspect

import pytorch_lightning as pl
from torch import optim
import torch
from torch.utils.data import DataLoader

from datasets import PATCH_DATASETS
from datasets.concat_ds import get_concat_dataset
from datasets.utils import sitk2tensor
from models import VAE_MODELS
from models._type import Tensor
from models.vae_base import VAEBackbone
from utils.visualization import vis3d_tensor
from utils import get_logger


class VAEXperiment(pl.LightningModule):

    def __init__(self, vae_model: Union[dict, VAEBackbone], params: dict):  # -> None
        super(VAEXperiment, self).__init__()
        # initializing model
        if isinstance(vae_model, dict):  # model is actually param dict
            vae_model = VAE_MODELS[vae_model['name']](**vae_model)
            self.model = vae_model
        elif isinstance(vae_model, VAEBackbone):
            self.model = vae_model

        self.params = params
        # self.curr_device = None
        self.hold_graph = False
        self.dataloader_params = {'num_workers': 12,
                                  'pin_memory': True}
        if ";" in params['dataset']:
            ds_name_list = params['dataset'].split(";")
            self.dataset = get_concat_dataset(ds_name_list)
        else:
            self.dataset = PATCH_DATASETS[params['dataset']]
        self.save_hyperparameters(ignore=["vae_model"])  # for loading later
        self.LOGGER = get_logger(cls_name=self.__class__.__name__)
        # the weight of KL loss calculated, should be adjustable
        self.train_dataloader()
        self.M_N = self.params['batch_size'] / self.num_train_imgs
        self.params['kl_actual_ratio'] = self.M_N * self.model.beta
        pass

    def forward(self, input: Tensor, **kwargs):  # -> Tensor
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        real_img, labels = batch

        results = self.forward(real_img, labels=labels)
        recons, inputs, mu, log_var = results

        return recons, inputs, mu, log_var

    def training_step_end(self, output):
        recons, inputs, mu, log_var = output
        train_loss = self.model.loss_function(recons=recons, inputs=inputs,
                                              mu=mu, log_var=log_var,
                                              M_N=self.M_N,
                                              #   batch_idx=batch_idx
                                              )
        return train_loss

    def training_epoch_end(self, outputs):
        output_dict = {}
        for key in outputs[0].keys():
            output_dict[key] = torch.stack([o[key] for o in outputs]).mean()
        for key, val in output_dict.items():
            self.log(key, val.item(),
                     on_step=False, on_epoch=True, prog_bar=False,
                     logger=True, sync_dist=True)
        self.log("step", float(self.global_step),)
        pass

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch

        results = self.forward(real_img, labels=labels)  # modified
        recons, inputs, mu, log_var = results
        return recons, inputs, mu, log_var

    def validation_step_end(self, outputs):
        recons, inputs, mu, log_var = outputs
        val_loss = self.model.loss_function(recons=recons, inputs=inputs,
                                            mu=mu, log_var=log_var,
                                            M_N=self.M_N
                                            )
        return val_loss

    def validation_epoch_end(self, outputs):
        val_loss = [o['loss'] for o in outputs]
        # called at the end of the epoch,
        # returns will be logged into metrics file.
        # visualize according to interval
        # log val_loss and Learning rate
        self.log('val_loss', torch.mean(torch.stack(val_loss)).item(),
                 on_step=False, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'],
                 on_step=False, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True)
        self.log("step", self.global_step,)
        if self.current_epoch % int(self.logger.vis_interval) == \
                int(self.logger.vis_interval) - 1:
            self.sample_images()
        pass

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_img_file_names = next(iter(self.sample_dataloader))
        device = next(self.model.parameters()).device
        test_input = test_input.to(device)  # self.curr_device
        # modified we don't need label
        recons = self.model.generate(test_input)

        # visualization using our codes
        midia_dir = f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/media"
        if not os.path.exists(midia_dir):
            os.makedirs(midia_dir)
        vis3d_tensor(recons.data, save_path=os.path.join(
            midia_dir, f"recons_{self.logger.name}_{self.current_epoch}.png"))
        vis3d_tensor(test_input.data, save_path=os.path.join(
            midia_dir, f"real_img_{self.logger.name}_{self.current_epoch}.png"))
        del test_input, recons  # , samples

        # draw loss curves
        self.logger.draw_loss_curve()
        self.logger.draw_kl_recon_loss()
        self.logger.draw_multiple_loss_curves()
        pass

    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'])
        optims.append(optimizer)

        # NOTE: to get steps_per_epoch, need to configure train_loader first to get num_train_imgs
        # lr scheduler
        if self.params['max_lr'] is not None:
            # debug: // 2
            step_per_epoch = self.num_train_imgs // self.params['batch_size']
            step_per_epoch = 1 if step_per_epoch == 0 else step_per_epoch
            # epochs or steps
            if self.params["max_epochs"] is not None:
                p = {"epochs": self.params["max_epochs"], "steps_per_epoch": step_per_epoch}
            elif self.params["max_steps"] is not None:
                p = {"total_steps": self.params["max_steps"]}
            scheduler = optim.lr_scheduler.OneCycleLR(optims[0], **p,
                                                      max_lr=self.params['max_lr'],
                                                      final_div_factor=self.params['final_div_factor'])
            lr_dict = {'scheduler': scheduler,
                       'interval': 'step'}  # one cycle lr in each epoch
            scheds.append(lr_dict)
            self.optims = optims
            self.scheds = scheds
            return optims, scheds
        else:
            self.optims = optims
            return optims

    def train_dataloader(self, root_dir=None, shuffle=True, drop_last=True):  # -> DataLoader
        # if self.dataset is already a dataset then proceed
        if inspect.isclass(self.dataset):  # isinstance(self.dataset, GenericMeta)
            train_ds = self.dataset(root_dir=root_dir,
                                    transform=sitk2tensor,
                                    split='train')
        self.num_train_imgs = len(train_ds)
        return DataLoader(dataset=train_ds,
                          batch_size=self.params['batch_size'],
                          shuffle=shuffle,
                          drop_last=drop_last,
                          **self.dataloader_params)

    def val_dataloader(self, root_dir=None, shuffle=False, drop_last=True):
        # isinstance(self.dataset, GenericMeta):
        if inspect.isclass(self.dataset):
            val_ds = self.dataset(root_dir=root_dir,
                                  transform=sitk2tensor,
                                  split='val')
        self.num_val_imgs = len(val_ds)
        self.sample_dataloader = DataLoader(val_ds,
                                            batch_size=self.params['batch_size'],
                                            shuffle=shuffle,
                                            drop_last=drop_last,
                                            **self.dataloader_params)
        return [self.sample_dataloader]

    def verbose_info(self):
        self.LOGGER.info(
            f"Implemented vae models: {VAE_MODELS.keys()}")
        self.LOGGER.info(
            f"Implemented patch datasets: {PATCH_DATASETS.keys()}")
        pass
