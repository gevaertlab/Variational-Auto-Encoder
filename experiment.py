''' Defines a torch_lightning Module '''
import os
from typing import GenericMeta

import pytorch_lightning as pl
from torch import optim
from torch.utils.data import DataLoader

from datasets import PATCH_DATASETS
from datasets.concat_ds import get_concat_dataset
from datasets.utils import sitk2tensor
from models._type import Tensor
from models.vae_base import VAEBackbone
from utils.visualization import vis3d_tensor


class VAEXperiment(pl.LightningModule):

    def __init__(self, vae_model: VAEBackbone, params: dict):  # -> None
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.dataloader_params = {'num_workers': 12,
                                  'pin_memory': True}
        if ";" in params['dataset']:
            ds_name_list = params['dataset'].split(";")
            self.dataset = get_concat_dataset(ds_name_list)
        else:
            self.dataset = PATCH_DATASETS[params['dataset']]
        self.save_hyperparameters()  # for loading later
        pass

    def forward(self, input: Tensor, **kwargs):  # -> Tensor
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        # the weight of KL loss calculated, should be adjustable
        M_N = self.params['batch_size'] / self.num_train_imgs
        self.params['kl_actual_ratio'] = M_N * self.model.beta
        train_loss = self.model.loss_function(*results,
                                              M_N=M_N,
                                              batch_idx=batch_idx)

        for key, val in train_loss.items():
            self.log(key, val.item(), logger=True)

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)  # modified
        val_loss = self.model.loss_function(*results,
                                            M_N=self.params['batch_size'] /
                                            self.num_val_imgs,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)
        # log val_loss and Learning rate
        self.log('val_loss', val_loss['loss'].item(), logger=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], logger=True)
        return val_loss

    def validation_epoch_end(self, outputs):
        # called at the end of the epoch,
        # returns will be logged into metrics file.
        # visualize according to interval
        if self.current_epoch % int(self.logger.vis_interval) == int(self.logger.vis_interval) - 1:
            self.sample_images()
        pass

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_img_file_names = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        # modified we don't need label
        recons = self.model.generate(test_input)

        # visualization using our codes
        vis3d_tensor(recons.data, save_path=os.path.join(f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/media",
                                                         f"recons_{self.logger.name}_{self.current_epoch}.png"))

        vis3d_tensor(test_input.data, save_path=os.path.join(f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/media",
                                                             f"real_img_{self.logger.name}_{self.current_epoch}.png"))
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
        self.train_dataloader()
        # lr scheduler
        if self.params['max_lr'] is not None:
            # debug: // 2
            step_per_epoch = self.num_train_imgs // self.params['batch_size']
            scheduler = optim.lr_scheduler.OneCycleLR(optims[0],
                                                      epochs=self.params['max_epochs'],
                                                      steps_per_epoch=step_per_epoch,
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
        if isinstance(self.dataset, GenericMeta):
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
        if isinstance(self.dataset, GenericMeta):
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
