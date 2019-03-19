import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from rlib.algorithms.base import Agent
from rlib.algorithms.vpg.model import Policy


class VPGAgent(Agent):
    """VPG Agent implementation."""

    # TODO: Consider how to extend this to accept multiple agents?
    # TODO: Ensure that this cannot be changed in other ways
    # TODO: Look up original value for these params
    REQUIRED_HYPERPARAMETERS = {
        "gamma": 1.0,
        "learning_rate": 1e-2
    }

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 policy=None,
                 optimizer=None,
                 new_hyperparameters=None,
                 seed: int = 0,
                 device: str = "cpu",
                 model_output_dir=None,
                 enable_logger: bool = False,
                 logger_path: str = None,
                 logger_comment: str = None):
        """Initialize an VPGAgent object.

        Args:
            state_size (int): dimension of each state.
            action_size (int): dimension of each action.
            policy (torch.nn.Module): Policy model.
            optimizer (torch.optim): Model optimizer.
            new_hyperparameters (dict): New hyperparameter values.
            seed (int): Random seed.
            device (str): Identifier for device to be used by PyTorch.
            model_output_dir (str): Directory where state dicts will be saved to.

        Returns:
            An instance of VPGAgent.
        """
        super(VPGAgent, self).__init__(
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

    def __str__(self) -> str:
        """Helper to output network architecture for the agent.
        
        Returns:
            A string representation of this algorithm.
        """
        return ("{}\n{}".format(
            "Policy:",
            self.policy,
        ))

    def origin(self) -> str:
        """Helper to get the original paper for this algorithm.

        Returns: 
            The original paper for this algorithm.
        """
        return 'https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf'
    
    def description(self) -> str:
        """Helper to get a brief description of this algorithm.

        Returns:
            A brief description of this algorithm.
        """
        description = (
            'The key idea underlying policy gradients is to push up the '
            'probabilities of actions that lead to higher return, and '
            'push down the probabilities of actions that lead to lower '
            'return, until you arrive at the optimal policy.'
        )
        return description

    def reset(self) -> None:
        """Reset VPGAgent."""
        self.saved_log_probs = []

    def step(self, state, action, reward, next_state, done, logger=None) -> None:
        """Increment step count."""
        self.time_step += 1

    def act(self, state, add_noise=False, logger=None):
        """Chooses an action for the current state based on the current policy.

        Args:
            state: The current state of the environment.
            add_noise (bool): Controls addition of noise.
            logger (Logger): An instance of Logger.

        Returns: 
            Action for given state as per current policy.
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.policy.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        self.saved_log_probs.append(log_prob)
        return action.item()

    def update(self, rewards, logger=None) -> None:
        """Updates policy.

        Args:
            rewards: Environment rewards.
            logger (Logger): An instance of Logger.
        """
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
