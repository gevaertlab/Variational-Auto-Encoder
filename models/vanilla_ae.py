# vanilla auto-encoder without reparameterization
from abc import abstractmethod

import torch
from torch import nn
from torch.nn import functional as F

from ._type import *
from .vae_3d import VAE3D


class VanillaAE(VAE3D):
    
    def __init__(self,
                **kwargs): #  -> None
        super(VanillaAE, self).__init__(**kwargs)
        pass
    
    def reparameterize(self, mu: Tensor, logvar: Tensor): # overwrite
        return mu
    
    def loss_function(self,
                      *args,
                      **kwargs): #  -> dict
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset

        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD': kld_loss}