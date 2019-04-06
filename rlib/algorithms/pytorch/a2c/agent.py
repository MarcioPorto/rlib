import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from rlib.algorithms.base import Agent


class A2CAgent(Agent):
    """A2C Agent implementation."""

    def __init__(self,
                 num_workers=5,
                 rollout_length=5):
        # TODO: Create a parallelized task for each worker
        # TODO: Need some idea of rollout (N-step bootstrapping)
        raise NotImplementedError()
