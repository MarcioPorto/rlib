import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from rlib.algorithms.dqn.model import QNetwork
from rlib.shared.replay_buffer import ReplayBuffer
from rlib.shared.utils import hard_update, soft_update


class DQNAgent:
    """Interacts with and learns from the environment."""

    # TODO: Consider how to extend this to accept multiple agents?

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
                lr=REQUIRED_HYPERPARAMETERS["learning_rate"]
            )

        self.memory = ReplayBuffer(
            REQUIRED_HYPERPARAMETERS["buffer_size"],
            REQUIRED_HYPERPARAMETERS["batch_size"],
            seed
        )

        self.time_step = 0

        # User options
        self.opt_soft_update = opt_soft_update
        self.opt_ddqn = opt_ddqn

        # Ensure local and target networks have the same initial weight
        hard_update(self.qnetwork_local, self.qnetwork_target)

    def get_hyperparameters(self):
        r"""Returns the current state of the required hyperparameters"""
        return self.REQUIRED_HYPERPARAMETERS

    def _set_hyperparameters(self, new_hyperparameters):
        r"""Adds user defined hyperparameter values to the list required
        hyperparameters.
        """
        for key, value in new_hyperparameters.items():
            if key in self.REQUIRED_HYPERPARAMETERS.keys():
                self.REQUIRED_HYPERPARAMETERS[key] = value

    def step(self, state, action, reward, next_state, done):
        r"""Saves experience to replay memory and updates model weights"""
        self.memory.add(state, action, reward, next_state, done)

        # Learn every `learn_every` time steps
        self.time_step += 1
        if self.time_step % REQUIRED_HYPERPARAMETERS["learn_every"] == 0:
            if len(self.memory) > REQUIRED_HYPERPARAMETERS["batch_size"]:
                experiences = self.memory.sample()
                self.learn(experiences, REQUIRED_HYPERPARAMETERS["gamma"])

    def act(self, state, eps=0.0):
        r"""Returns actions for given state as per current policy.

        Params
        ======
            state (numpy array): Current state
            eps (float): Epsilon, for Epsilon-greedy action selection
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
            expected_Q = self.qnetwork_local(states).gather(1, actions)
        else:
            # Vanilla DQN
            next_max_a = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            # (1 - dones) ignores the actions that ended the game
            target_Q = rewards + (gamma * next_max_a * (1 - dones))
            expected_Q = self.qnetwork_local(states).gather(1, actions)

        # Compute and minimize the loss
        loss = F.mse_loss(expected_Q, target_Q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        if self.opt_soft_update:
            soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        else:
            if self.time_step % REQUIRED_HYPERPARAMETERS["hard_update_every"] == 0:
                hard_update(self.qnetwork_local, self.qnetwork_target)

    def print_network(self):
        r"""Helper to print network architecture for the agent."""
        print("Q-Network (Local):")
        print(self.qnetwork_local)
        print("Q-Network (Target):")
        print(self.qnetwork_target)

    def save_model(self):
        r"""Saves model weights to file."""
        # TODO: Move to base class of create helpers
        torch.save(
            self.qnetwork_local.state_dict(),
            os.path.join(self.model_dir, 'qnetwork_local_params.pth')
        )
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(self.model_dir, 'optimizer_params.pth')
        )

    def load_model(self):
        r"""Loads weights from saved model."""
        # TODO: Move to base class of create helpers
        self.qnetwork_local.load_state_dict(
            torch.load(os.path.join(self.model_dir, 'qnetwork_local_params.pth'))
        )
        self.optimizer.load_state_dict(
            torch.load(os.path.join(self.model_dir, 'optimizer_params.pth'))
        )
