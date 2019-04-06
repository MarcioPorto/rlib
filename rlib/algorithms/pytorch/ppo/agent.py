import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from rlib.algorithms.base import Agent
from rlib.algorithms.pytorch.ppo.model import Policy


class PPOAgent(Agent):
    """PPO Agent implementation."""

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
                 logger=None):
        """Initialize an PPOAgent object.

        Args:
            state_size (int): dimension of each state.
            action_size (int): dimension of each action.
            policy (torch.nn.Module): Policy model.
            optimizer (torch.optim): Model optimizer.
            new_hyperparameters (dict): New hyperparameter values.
            seed (int): Random seed.
            device (str): Identifier for device to be used by PyTorch.
            model_output_dir (str): Directory where state dicts will be saved to.
            logger (Logger): Tensorboard logger helper.

        Returns:
            An instance of PPOAgent.
        """
        super(PPOAgent, self).__init__(
            new_hyperparameters=new_hyperparameters,
            logger=logger
        )

        random.seed(seed)

        self.seed = seed
        self.time_step = 0

        self.device = device

        if policy:
            self.policy = policy.to(self.device)
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
        """Reset PPOAgent."""
        self.saved_log_probs = []

    def step(self, state, action, reward, next_state, done) -> None:
        """Increment step count."""
        self.time_step += 1

    def act(self, state, add_noise: bool = False):
        """Chooses an action for the current state based on the current policy.

        Args:
            state: The current state of the environment.
            add_noise (bool): Controls addition of noise.

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

    def update(self, states, actions, rewards) -> None:
        """Updates policy.

        Args:
            rewards: Environment rewards.
        """
        discounts = [
            self.GAMMA ** i
            for i in range(len(rewards) + 1)
        ]
        # R is discounted future rewards
        R = sum([a * b for a, b in zip(discounts, rewards)])

        policy_loss = []
        for log_prob in self.saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        if self.logger:
            policy_loss = policy_loss.cpu().detach().item()
            self.logger.add_scalar(
                'loss', policy_loss, self.time_step
            )


    @staticmethod
    def clipped_surrogate(policy, old_probs, states, actions, rewards,
                          discount=0.995, epsilon=0.1, beta=0.01):
        """ Clipped surrogate function.
        Returns the sum of log-prob divided by T.
        Same thing as -policy_loss.
        """
        discount = discount ** np.arange(len(rewards))
        rewards = np.asarray(rewards)*discount[:,np.newaxis]

        # convert rewards to future rewards
        rewards_future = rewards[::-1].cumsum(axis=0)[::-1]

        mean = np.mean(rewards_future, axis=1)
        std = np.std(rewards_future, axis=1) + 1.0e-10

        rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]

        # convert everything into pytorch tensors and move to gpu if available
        actions = torch.tensor(actions, dtype=torch.int8, device=device)
        old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
        rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

        # convert states to policy (or probability)
        new_probs = states_to_prob(policy, states)
        new_probs = torch.where(actions == RIGHT, new_probs, 1.0-new_probs)

        # ratio for clipping
        ratio = new_probs/old_probs

        # clipped function
        clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
        clipped_surrogate = torch.min(ratio*rewards, clip*rewards)

        # include a regularization term
        # this steers new_policy towards 0.5
        # add in 1.e-10 to avoid log(0) which gives nan
        entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
            (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))

        # this returns an average of all the entries of the tensor
        # effective computing L_sur^clip / T
        # averaged over time-step and number of trajectories
        # this is desirable because we have normalized our rewards
        return torch.mean(clipped_surrogate + beta*entropy)
