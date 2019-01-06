import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from rlib.algorithms.base import Agent
from rlib.algorithms.vpg.model import Policy


class VPGAgent(Agent):
    """Interacts with and learns from the environment."""

    # TODO: Consider how to extend this to accept multiple agents?
    # TODO: Ensure that this cannot be changed in other ways
    # TODO: Look up original value for these params
    REQUIRED_HYPERPARAMETERS = {
        "gamma": 1.0,
        "learning_rate": 1e-2
    }

    def __init__(self,
                 state_size,
                 action_size,
                 policy=None,
                 optimizer=None,
                 new_hyperparameters=None,
                 seed=0,
                 device="cpu",
                 model_output_dir=None):
        super(VPGAgent, self).__init__(
            new_hyperparameters=new_hyperparameters
        )

        # TODO: Single interface for seeding
        self.seed = random.seed(seed)

        self.device = device

        if policy:
            self.policy = policy
        else:
            self.policy = Policy(
                s_size=state_size,
                a_size=action_size
            ).to(self.device)

        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(
                self.policy.parameters(),
                lr=self.LEARNING_RATE
            )

        self.state_dicts = [
            (self.policy, "policy_params"),
            (self.optimizer, "optimizer_params"),
        ]

        self.saved_log_probs = []
        self.model_output_dir = model_output_dir

    def reset(self):
        self.saved_log_probs = []

    def act(self, state, add_noise=False):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.policy.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        self.saved_log_probs.append(log_prob)
        return action.item()

    def update(self, rewards):
        discounts = [
            self.GAMMA**i
            for i in range(len(rewards)+1)
        ]
        R = sum([a * b for a, b in zip(discounts, rewards)])

        policy_loss = []
        for log_prob in self.saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
