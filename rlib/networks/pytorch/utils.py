import torch
import torch.nn as nn


def identity(x: torch.Tensor) -> torch.Tensor:
    """Return input without any change."""
    return x


def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Initialize uniform parameters on a single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)
    return layer
