"""
Taken from medipixel/rl_algorithms with minor changes.
See https://github.com/medipixel/rl_algorithms/blob/master/algorithms/common/networks/mlp.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

from rlib.networks.pytorch.utils import identity, init_layer_uniform


class Policy(nn.Module):
    def __init__(self, s_size=4, h_size=16, a_size=2):
        super(Policy, self).__init__()

        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class Network(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_sizes: list,
                 hidden_activation: Callable = F.relu,
                 output_activation: Callable = identity,
                 layer_type=nn.Linear,
                 use_output_layer: bool = True,
                 n_category: int = -1,
                 init_fn: Callable = init_layer_uniform
                 ):
        """Initialization.

        Args:
            use_output_layer (bool): whether or not to use the last layer
            n_category (int): category number (-1 if the action is continuous)

        """
        super(Network, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_type = layer_type
        self.use_output_layer = use_output_layer
        self.n_category = n_category

        # Initialize hidden layers
        self.hidden_layers: list = []
        input_size = self.input_size

        for i, hidden_size in enumerate(hidden_sizes):
            fc = self.layer_type(input_size, hidden_size)
            input_size = hidden_size
            self.__setattr__("hidden_fc{}".format(i), fc)
            self.hidden_layers.append(fc)

        # Initialize output layer
        if self.use_output_layer:
            self.output_layer = self.layer_type(input_size, output_size)
            self.output_layer = init_fn(self.output_layer)
        else:
            self.output_layer = identity
            self.output_activation = identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for hidden_layer in self.hidden_layers:
            x = self.hidden_activation(hidden_layer(x))
        return self.output_activation(self.output_layer(x))
