# model that's similar to VAE's encoder

from torch import nn

from typing import List

import torch


class CNNClassification(nn.Module):

    def __init__(self,
                 in_channels: int = 1,
                 n_classes: int = 2,
                 hidden_dims: List = None,
                 **kwargs):  # -> None
        super(CNNClassification, self).__init__(**kwargs)
        modules = []
        if hidden_dims is None:
            hidden_dims = [8, 32, 128, 512]
        # don't modify hidden_dims
        self.hidden_dims = hidden_dims.copy()
        hidden_dims_variable = hidden_dims.copy()

        # Build Encoder
        # formula: o = floor( (i + p - k) / s) + 1
        for h_dim in hidden_dims_variable:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(in_channels=in_channels,
                              out_channels=h_dim,  # need 3d conv layers
                              kernel_size=3,
                              stride=2,  # will reduce the lengths of each dim to half: size = 1/8 size
                              padding=1),
                    nn.BatchNorm3d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.latent_layers = nn.Sequential(*modules)

        # final FC layer and activation using softmax
        self.fc = nn.Linear(hidden_dims_variable[-1]*8, n_classes)
        self.fc_activation = nn.Softmax(dim=1)
        pass

    def forward(self, x, **kwargs):
        # x: (batch_size, in_channels, x, y, z)
        # output: (batch_size, 1)
        x = self.latent_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = self.fc_activation(x)
        return x
