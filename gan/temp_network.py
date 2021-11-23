import torch
from torch import nn
from .utils import Reshape, OOGANInput


class TemporalNetwork(nn.Module):
    
    n_bars = 2
    
    def __init__(self,
                 dimension: int=3,
                 hid_channels: int=1024):
        super().__init__()        
        self.net = nn.Sequential(
            # input shape: (batch_size, dimension) 
            Reshape(shape=[dimension, 1, 1]),
            # output shape: (batch_size, dimension, 1, 1)
            nn.ConvTranspose2d(dimension, hid_channels,
                               kernel_size=(2, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(inplace=True),
            # output shape: (batch_size, hid_channels, 2, 1)
            nn.ConvTranspose2d(hid_channels, dimension,
                               kernel_size=(self.n_bars-1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(dimension),
            nn.ReLU(inplace=True),
            # output shape: (batch_size, dimension, 1, 1)
            Reshape(shape=[dimension, self.n_bars])
        )

    def forward(self, v):
        # print("v:", *v.shape, v.dtype)
        fx = self.net(v)
        return fx
