import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from rlib.algorithms.base import Agent
from rlib.algorithms.dqn.model import QNetwork
from rlib.shared.replay_buffer import ReplayBuffer
from rlib.shared.utils import hard_update, soft_update


class DQN(Agent):
    """Interacts with and learns from the environment."""

    # TODO: Consider how to extend this to accept multiple agents?
    # TODO: Add noise to DQN?

    # TODO: Ensure that this cannot be changed in other ways
    # TODO: Look up original value for these params
    REQUIRED_HYPERPARAMETERS = {
        "buffer_size": int(1e7),
        "batch_size": 32,
        "gamma": 0.99,
        "learning_rate": 2.5e-4,
        "tau": 1e-3,
        "learn_every": 4,
        "hard_update_every": 10000
    }
    
    ALGORITHM = "DQN"

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
        """Initialize an Agent object.

        Args:
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
        super(DQN, self).__init__(
            new_hyperparameters=new_hyperparameters
        )

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.time_step = 0

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
                lr=self.LEARNING_RATE
            )

        # Replay memory
        self.memory = ReplayBuffer(self.BUFFER_SIZE, self.BATCH_SIZE, self.device, seed)

        # User options
        self.opt_soft_update = opt_soft_update
        self.opt_ddqn = opt_ddqn

        self.model_output_dir = model_output_dir

        self.state_dicts = [
            (self.qnetwork_local, "qnetwork_local_params"),
            (self.optimizer, "optimizer_params"),
        ]

        # Ensure local and target networks have the same initial weight
        hard_update(self.qnetwork_local, self.qnetwork_target)

    def __str__(self):
        """Helper to output network architecture for the agent."""
        return ("{}\n{}\n{}\n{}".format(
            "Q-Network (Local):",
            self.qnetwork_local,
            "Q-Network (Target):",
            self.qnetwork_target
        ))

    def origin(self):
        print('https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf')
    
    def description(self):
        description = (
            'DQN is an algorithm created by DeepMind that brings together the power '
            'of the Q-Learning algorithm with the advantages of generalization through '
            'function approximation. It uses a deep neural network to estimate a Q-value '
            'function. As such, the input to the network is the current state of the '
            'environment, and the output is the Q-value for each possible action.'
        )
        print(description)

    def step(self, state, action, reward, next_state, done, logger=None):
        """Saves experience to replay memory and updates model weights"""
        self.memory.add(state, action, reward, next_state, done)

        # Learn every `learn_every` time steps
        self.time_step += 1
        if self.time_step % self.LEARN_EVERY == 0:
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, logger=logger)

    def act(self, state, eps=0.0, add_noise=False, logger=None):
        """Returns actions for given state as per current policy.

        Args:
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

    def learn(self, experiences, logger=None):
        """Updates value parameters using given batch of experience tuples.

        Args:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        """
        states, actions, rewards, next_states, dones = experiences

        if self.opt_ddqn:
            # Double DQN
            non_final_next_states = next_states * (1 - dones)
            # Get the actions themselves, not their output value
            _, next_state_actions = self.qnetwork_local(non_final_next_states).max(1, keepdim=True)
            next_Q_targets = self.qnetwork_target(non_final_next_states).gather(1, next_state_actions)
            target_Q = rewards + (self.GAMMA * next_Q_targets * (1 - dones))
        else:
            # Vanilla DQN
            next_max_a = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            target_Q = rewards + (self.GAMMA * next_max_a * (1 - dones))

        expected_Q = self.qnetwork_local(states)
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(1)
        expected_Q = torch.gather(expected_Q, 1, actions.long())

        # Compute and minimize the loss
        loss = F.mse_loss(expected_Q, target_Q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        if self.opt_soft_update:
            soft_update(self.qnetwork_local, self.qnetwork_target, self.TAU)
        elif self.time_step % self.HARD_UPDATE_EVERY == 0:
            hard_update(self.qnetwork_local, self.qnetwork_target)

        if logger:
            loss = loss.cpu().detach().item()
            logger.add_scalar(
                'loss', loss, self.time_step
            )
