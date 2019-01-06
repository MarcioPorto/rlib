import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from rlib.algorithms.base import Agent
from rlib.algorithms.ddpg.model import Actor, Critic
from rlib.shared.noise import OUNoise
from rlib.shared.replay_buffer import ReplayBuffer
from rlib.shared.utils import hard_update, soft_update


class DDPG(Agent):
    """Interacts with and learns from the environment."""

    REQUIRED_HYPERPARAMETERS = {
        "buffer_size": int(1e6),
        "batch_size": 64,
        "gamma": 0.99,
        "tau": 1e-3,
        "learning_rate_actor": 1e-4,
        "learning_rate_critic": 1e-3,
        "weight_decay": 1e-2
    }

    def __init__(self,
                 state_size,
                 action_size,
                 num_agents,
                 actor_local=None,
                 actor_target=None,
                 actor_optimizer=None,
                 critic_local=None,
                 critic_target=None,
                 critic_optimizer=None,
                 new_hyperparameters=None,
                 seed=0,
                 device="cpu",
                 model_output_dir=None,
                 opt_soft_update=False):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents in the environment
        """
        super(DDPG, self).__init__(
            new_hyperparameters=new_hyperparameters
        )

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.num_agents = num_agents
        self.device = device

        # Actor Network (w/ Target Network)
        self.actor_local = actor_local if actor_local else Actor(state_size, action_size, seed).to(device)
        self.actor_target = actor_target if actor_target else Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = actor_optimizer if actor_optimizer else optim.Adam(self.actor_local.parameters(), lr=self.LEARNING_RATE_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = critic_local if critic_local else Critic(state_size, action_size, seed).to(device)
        self.critic_target = critic_target if critic_target else Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = critic_optimizer if critic_optimizer else optim.Adam(self.critic_local.parameters(), lr=self.LEARNING_RATE_CRITIC, weight_decay=self.WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, seed)

        # Replay memory
        self.memory = ReplayBuffer(self.BUFFER_SIZE, self.BATCH_SIZE, self.device, seed)

        # User options
        self.opt_soft_update = opt_soft_update

        self.model_output_dir = model_output_dir

        self.state_dicts = [
            (self.actor_local, "actor_local_params"),
            (self.actor_optimizer, "actor_optimizer_params"),
            (self.critic_local, "critic_local_params"),
            (self.critic_optimizer, "critic_optimizer_params"),
        ]

        # Ensure local and target networks have the same initial weight
        hard_update(self.actor_local, self.actor_target)
        hard_update(self.critic_local, self.critic_target)

    def __str__(self):
        r"""Helper to output network architecture for the agent."""
        return ("{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}".format(
            "Actor (Local):",
            self.actor_local,
            "Actor (Target):",
            self.actor_target,
            "Critic (Local):",
            self.critic_local,
            "Critic (Target):",
            self.critic_target
        ))

    def step(self, states, actions, rewards, next_states, dones):
        # TODO: Refactor num_Agents out
        # TODO: How to use shared experience replay?

        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward tuple for each agent to a shared replay buffer before sampling
        for i in range(self.num_agents):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, self.GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(state).float().to(self.device)
        actions = np.zeros((self.num_agents, self.action_size))
        self.actor_local.eval()
        with torch.no_grad():
            for i, state in enumerate(states):
                # Populate list of actions one state at a time
                actions[i, :] = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample()

        # TODO: Have parameter that controls this?
        # return np.clip(action, -1, 1)
        return actions

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        """
        states, actions, rewards, next_states, dones = experiences

        ### Update critic
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.GAMMA * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)  # adds gradient clipping to stabilize learning
        self.critic_optimizer.step()

        ### Update actor
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        ### Update target networks
        if self.opt_soft_update:
            soft_update(self.actor_local, self.actor_target, self.TAU)
            soft_update(self.critic_local, self.critic_target, self.TAU)
        else:
            hard_update(self.actor_local, self.actor_target)
            hard_update(self.critic_local, self.critic_target)