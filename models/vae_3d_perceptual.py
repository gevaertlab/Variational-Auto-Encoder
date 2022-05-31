""" VAE 3d with perceptual loss """
from typing import List

import torch
from torch.nn import functional as F

from perceptual_network.medical_net import MedicalNet

from .vae_3d import VAE3D


class VAE3DPerceptual(VAE3D):

    def __init__(self,
                 perceptual_net,
                 in_channels: int = 1,
                 latent_dim: int = 1024,
                 hidden_dims: List = None,
                 beta=1,
                 gamma=1,
                 **kwargs):  # -> None
        super(VAE3DPerceptual, self).__init__(in_channels=in_channels,
                                              latent_dim=latent_dim,
                                              hidden_dims=hidden_dims,
                                              beta=beta,
                                              **kwargs)
        
        if isinstance(perceptual_net, dict):
            self.perceptual_net = MedicalNet(**perceptual_net)
        else:
            self.perceptual_net = perceptual_net
        # freeze Parameters
        for param in self.perceptual_net.parameters():
            param.requires_grad = False
        self.gamma = gamma
        pass

    def loss_function(self, 
                    recons, 
                    inputs,
                    mu, 
                    log_var, 
                    M_N, 
                    *args,
                      **kwargs):
        loss_dict = VAE3D.loss_function(self=self, 
                                    recons=recons, 
                                    inputs=inputs, 
                                    mu=mu, 
                                    log_var=log_var, 
                                    M_N=M_N, 
                                    *args,
                                    **kwargs)

        # for input
        input_perc_layers = []
        self.perceptual_net.forward(inputs)
        for layer in self.perceptual_net.fea_layers:
            input_perc_layers.append(self.perceptual_net._features[layer])
        
        # for recon
        recon_perc_layers = []
        self.perceptual_net.forward(recons)
        for layer in self.perceptual_net.fea_layers:
            recon_perc_layers.append(self.perceptual_net._features[layer])
        
        # perceptual loss: MSE calc
        perc_loss = []
        for i in range(len(input_perc_layers)):
            perc_loss.append(F.mse_loss(input_perc_layers[i], recon_perc_layers[i]))
        perc_loss = torch.mean(torch.stack(perc_loss))

        # overall loss
        loss = loss_dict['loss'] + self.gamma * perc_loss

        loss_dict.update({"loss": loss, "perceptual_loss": perc_loss})
        return loss_dict
        