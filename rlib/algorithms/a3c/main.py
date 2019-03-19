import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from rlib.algorithms.base import Agent


class A3C(Agent):
    """A3C Agent implementation."""
    raise NotImplementedError()
