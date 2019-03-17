import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from rlib.algorithms.base import Agent
from rlib.algorithms.vpg.model import Policy


class PPO(Agent):
    """Interacts with and learns from the environment."""

    # TODO: Consider how to extend this to accept multiple agents?
    # TODO: Ensure that this cannot be changed in other ways
    # TODO: Look up original value for these params
    REQUIRED_HYPERPARAMETERS = {
        "gamma": 1.0,
        "learning_rate": 1e-2,
        "beta": .01,
        "discount_rate": .99,
        "update_every": 4  # TODO: rename
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
        raise NotImplementedError()

        super(PPO, self).__init__(
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

    def reset(self):
        self.saved_log_probs = []

    def step(self, state, action, reward, next_state, done, logger=None):
        # NOTE: Think this makes sense here
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
