''' Defines a torch_lightning Module '''
import os

import pytorch_lightning as pl
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
# import torchvision.utils as vutils

from models._type import *
from dataset import LIDCPathDataset, LNDbDebugDataset, LNDbPathDataset, sitk2tensor
from models.vae_base import VAESkeleton
from utils.visualization import vis3DTensor


class VAEXperiment(pl.LightningModule):

    def __init__(self, vae_model: VAESkeleton, params: dict): # -> None
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        pass

    def forward(self, input: Tensor, **kwargs): #  -> Tensor
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['batch_size'] /
                                              self.num_train_imgs,
                                              batch_idx=batch_idx)

        self.logger.experiment.log({key: val.item()
                                    for key, val in train_loss.items()})

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels) # modified
        val_loss = self.model.loss_function(*results,
                                            M_N=self.params['batch_size'] /
                                            self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx=batch_idx)

        return val_loss

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        # visualize according to interval
        if self.current_epoch % int(self.logger.description) == int(self.logger.description) - 1:
            self.sample_images()
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_img_file_names = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        # test_label = test_label.to(self.curr_device)
        recons = self.model.generate(test_input) # modified we don't need label

        # visualization using our codes
        vis3DTensor(recons.data, save_dir = os.path.join(f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/media",
                                                         f"recons_{self.logger.name}_{self.current_epoch}.png"))

        vis3DTensor(test_input.data, save_dir = os.path.join(f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/media",
                                                             f"real_img_{self.logger.name}_{self.current_epoch}.png"))
        
        
        # vutils.save_image(recons.data,
        #                   f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        #                   f"recons_{self.logger.name}_{self.current_epoch}.png",
        #                   normalize=True,
        #                   nrow=12)

        # vutils.save_image(test_input.data,
        #                   f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        #                   f"real_img_{self.logger.name}_{self.current_epoch}.png",
        #                   normalize=True,
        #                   nrow=12)

        # try:
        #     samples = self.model.sample(144,
        #                                 self.curr_device,
        #                                 labels=test_label)
        #     vutils.save_image(samples.cpu().data,
        #                       f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        #                       f"{self.logger.name}_{self.current_epoch}.png",
        #                       normalize=True,
        #                       nrow=12)
        # except:
        #     pass

        del test_input, recons  # , samples

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)
            return optims, scheds
        except:
            return optims

    def train_dataloader(self) -> DataLoader:
        lidc_train = LNDbDebugDataset(size=20, transform=sitk2tensor) # modified
        self.num_train_imgs = len(lidc_train)
        return DataLoader(lidc_train, batch_size=self.params['batch_size'], shuffle=True, drop_last=True)

    def val_dataloader(self):
        lndb_val = LNDbDebugDataset(size=20, transform=sitk2tensor) # modified
        self.num_val_imgs = len(lndb_val)
        self.sample_dataloader = DataLoader(
            lndb_val, batch_size=self.params['batch_size'], shuffle=True, drop_last=True) # let val = train in debugging mode to see if overfit
        return [self.sample_dataloader] # debug modified
