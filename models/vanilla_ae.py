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