import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from rlib.algorithms.base import Agent
from rlib.algorithms.dqn.model import QNetwork
from rlib.shared.replay_buffer import ReplayBuffer
from rlib.shared.utils import hard_update, soft_update


class DQNAgent(Agent):
    """Interacts with and learns from the environment."""

    # TODO: Consider how to extend this to accept multiple agents?
    # TODO: Add noise to DQN?

    # TODO: Ensure that this cannot be changed in other ways
    # TODO: Look up original value for these params
    REQUIRED_HYPERPARAMETERS = {
        "buffer_size": int(2e5),
        "batch_size": 64,
        "gamma": 0.95,
        "learning_rate": 5e-4,
        "tau": 1e-3,
        "learn_every": 4,
        "hard_update_every": 5
    }

    def __init__(self,
                 state_size,
                 action_size,
                 qnetwork_local=None,
                 qnetwork_target=None,
                 optimizer=None,
                 new_hyperparameters=None,
                 seed=0,
                 device="cpu",
                 model_output_dir=None,
                 opt_soft_update=False,
                 opt_ddqn=False):
        r"""Initialize an Agent object.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            qnetwork_local (torch.nn.Module): Local Q-Network model
            qnetwork_target (torch.nn.Module): Target Q-Network model
            optimizer (torch.optim): Local Q-Network optimizer
            new_hyperparameters (dict): New hyperparameter values
            seed (int): Random seed
            device (str): Identifier for device to be used by PyTorch
            model_output_dir (str): Directory where state dicts will be saved to
            opt_soft_update (bool): Use soft update instead of hard update
            opt_ddqn (bool): Use Double DQN for `expected_Q`
        """
        if new_hyperparameters:
            self._set_hyperparameters(new_hyperparameters)

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device

        if qnetwork_local:
            self.qnetwork_local = qnetwork_local
        else:
            self.qnetwork_local = QNetwork(state_size, action_size).to(self.device)

        if qnetwork_target:
            self.qnetwork_target = qnetwork_target
        else:
            self.qnetwork_target = QNetwork(state_size, action_size).to(self.device)

        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(
                self.qnetwork_local.parameters(),
                lr=self.REQUIRED_HYPERPARAMETERS["learning_rate"]
            )

        self.memory = ReplayBuffer(
            self.REQUIRED_HYPERPARAMETERS["buffer_size"],
            self.REQUIRED_HYPERPARAMETERS["batch_size"],
            self.device,
            seed
        )

        self.time_step = 0

        # User options
        self.opt_soft_update = opt_soft_update
        self.opt_ddqn = opt_ddqn

        self.model_output_dir = model_output_dir

        self.struct = [
            (self.qnetwork_local, "qnetwork_local_params"),
            (self.optimizer, "optimizer_params"),
        ]

        # Ensure local and target networks have the same initial weight
        hard_update(self.qnetwork_local, self.qnetwork_target)

    def step(self, state, action, reward, next_state, done):
        r"""Saves experience to replay memory and updates model weights"""
        self.memory.add(state, action, reward, next_state, done)

        # Learn every `learn_every` time steps
        self.time_step += 1
        if self.time_step % self.REQUIRED_HYPERPARAMETERS["learn_every"] == 0:
            if len(self.memory) > self.REQUIRED_HYPERPARAMETERS["batch_size"]:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, eps=0.0, add_noise=False):
        r"""Returns actions for given state as per current policy.

        Params
        ======
            state (numpy array): Current state
            eps (float): Epsilon, for Epsilon-greedy action selection
            add_noise (boolean): Add noise to the agent's actions?
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        r"""Updates value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        """
        states, actions, rewards, next_states, dones = experiences

        gamma = self.REQUIRED_HYPERPARAMETERS["gamma"]

        if self.opt_ddqn:
            # Double DQN
            non_final_next_states = next_states * (1 - dones)
            # Get the actions themselves, not their output value
            _, next_state_actions = self.qnetwork_local(non_final_next_states).max(1, keepdim=True)
            next_Q_targets = self.qnetwork_target(non_final_next_states).gather(1, next_state_actions)
            target_Q = rewards + (gamma * next_Q_targets * (1 - dones))
        else:
            # Vanilla DQN
            next_max_a = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            target_Q = rewards + (gamma * next_max_a * (1 - dones))

        expected_Q = self.qnetwork_local(states)
        expected_Q = torch.gather(expected_Q, 1, actions.long())

        # Compute and minimize the loss
        loss = F.mse_loss(expected_Q, target_Q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        if self.opt_soft_update:
            soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        else:
            if self.time_step % self.REQUIRED_HYPERPARAMETERS["hard_update_every"] == 0:
                hard_update(self.qnetwork_local, self.qnetwork_target)

    def __str__(self):
        r"""Helper to output network architecture for the agent."""
        return ("{}\n{}\n{}\n{}".format(
            "Q-Network (Local):",
            self.qnetwork_local,
            "Q-Network (Target):",
            self.qnetwork_target
        ))
