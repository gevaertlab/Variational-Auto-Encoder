''' The model for perceptual network '''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LeNet(nn.Module):
    '''
    LeNet (3D version) structure + an single dimension output for regression job
    Typically, LeNet takes input of shape (B, 32, 32, 1)
    '''
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv3d(in_channels = 1, out_channels = 6, 
                            kernel_size = 5, stride = 1, padding = 0)
        self.conv2 = nn.Conv3d(in_channels = 6, out_channels = 16, 
                            kernel_size = 5, stride = 1, padding = 0)
        self.conv3 = nn.Conv3d(in_channels = 16, out_channels = 120, 
                            kernel_size = 5, stride = 1, padding = 0)
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10)
        self.output = nn.Linear(10, 1) # added layer
        self.tanh = nn.Tanh()
        self.avgpool = nn.AvgPool3d(kernel_size = 2, stride = 2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.avgpool(x)
        x = self.conv2(x)
        x = self.tanh(x)
        x = self.avgpool(x)
        x = self.conv3(x)
        x = self.tanh(x)
        
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        x = self.tanh(x)
        output = self.output(x)
        return output
        
    def loss_function(self, y_pred: torch.Tensor, y_true: torch.Tensor): # -> torch.Tensor
        return F.mse_loss(y_pred, y_true)
        
        
    