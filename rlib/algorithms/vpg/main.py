import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from rlib.algorithms.base import Agent
from rlib.algorithms.vpg.model import Policy


class VPG(Agent):
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
                 model_output_dir=None,
                 enable_logger=False,
                 logger_path=None,
                 logger_comment=None):
        super(VPG, self).__init__(
            new_hyperparameters=new_hyperparameters,
            enable_logger=enable_logger,
            logger_path=logger_path,
            logger_comment=logger_comment
        )

        # TODO: Single interface for seeding
        self.seed = random.seed(seed)
        self.time_step = 0

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

    def origin(self):
        print('https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf')
    
    def description(self):
        description = (
            'The key idea underlying policy gradients is to push up the '
            'probabilities of actions that lead to higher return, and '
            'push down the probabilities of actions that lead to lower '
            'return, until you arrive at the optimal policy.'
        )
        print(description)

    def reset(self):
        self.saved_log_probs = []

    def step(self, state, action, reward, next_state, done, logger=None):
        self.time_step += 1

    def act(self, state, add_noise=False, logger=None):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.policy.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        self.saved_log_probs.append(log_prob)
        return action.item()

    def update(self, rewards, logger=None):
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

        if logger:
            policy_loss = policy_loss.cpu().detach().item()
            logger.add_scalar(
                'loss', policy_loss, self.time_step
            )
