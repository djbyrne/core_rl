"""Series of networks used"""

import numpy as np
import torch
from torch import nn
from typing import Tuple


class CNN(nn.Module):
    """
    Simple MLP network

    Args:
        input_shape: observation shape of the environment
        n_actions: number of discrete actions available in the environment
    """
    def __init__(self, input_shape, n_actions):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

class MLP(nn.Module):
    """
    Simple MLP network

    Args:
        input_shape: observation shape of the environment
        n_actions: number of discrete actions available in the environment
        hidden_size: size of hidden layers
    """

    def __init__(self, input_shape: Tuple, n_actions: int, hidden_size: int = 128):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_shape[0], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x.float())

class DuelingMLP(nn.Module):
    """
    MLP network with duel heads for val and advantage

    Args:
        input_shape: observation shape of the environment
        n_actions: number of discrete actions available in the environment
        hidden_size: size of hidden layers
    """

    def __init__(self, input_shape: Tuple, n_actions: int, hidden_size: int = 128):
        super(DuelingMLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_shape[0], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )
        self.fc_val = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        adv, val = self.adv_val(x)
        q = val + (adv - adv.mean(dim=1, keepdim=True))
        return q

    def adv_val(self, x):
        fx = x.float()
        base_out = self.net(fx)
        return self.fc_adv(base_out), self.fc_val(base_out)
