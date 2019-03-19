import copy
import os
import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from rlib.algorithms.base import Agent
from rlib.algorithms.maddpg.model import Actor, Critic
from rlib.shared.noise import OUNoise
from rlib.shared.replay_buffer import ReplayBuffer
from rlib.shared.utils import hard_update, soft_update


class MADDPGAgent(Agent):
    """MADDPG implementation."""

    REQUIRED_HYPERPARAMETERS = {
        "buffer_size": int(1e6),
        "batch_size": 64,
        "gamma": 0.99,
        "tau": 1e-3,
        "learning_rate_actor": 1e-4,
        "learning_rate_critic": 1e-3,
        "weight_decay": 1e-2,
        "learn_every": 4,
        "hard_update_every": 5
    }

    def __init__(self,
                 state_size,
                 action_size,
                 num_agents,
                 agents=None,
                 new_hyperparameters=None,
                 seed=0,
                 device="cpu",
                 model_output_dir=None,
                 enable_logger=False,
                 logger_path=None,
                 logger_comment=None,
                 opt_soft_update=False):
        """Initialize a MADDPGAgent wrapper.
       
        Args:
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): the number of agents in the environment
        """
        raise NotImplementedError()

        super(DDPG, self).__init__(
            new_hyperparameters=new_hyperparameters,
            enable_logger=enable_logger,
            logger_path=logger_path,
            logger_comment=logger_comment
        )

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(seed)
        self.device = device
        self.time_step = 0

        if agents:
            self.agents = agents
        else:
            self.agents = [DDPGAgent(state_size, action_size, agent_id=i+1, handler=self) for i in range(num_agents)]

        # Replay memory
        self.memory = ReplayBuffer(self.BUFFER_SIZE, self.BATCH_SIZE, self.device, seed)

        # User options
        self.opt_soft_update = opt_soft_update

        self.model_output_dir = model_output_dir

    def reset(self):
        """Resets OU Noise for each agent."""
        for agent in self.agents:
            agent.reset()

    def act(self, observations, add_noise=False, logger=None):
        """Picks an action for each agent given their individual observations
        and the current policy."""
        actions = []
        for agent, observation in zip(self.agents, observations):
            action = agent.act(observation, add_noise=add_noise)
            actions.append(action)
        return np.array(actions)

    def step(self, observations, actions, rewards, next_observations, dones, logger=None):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        observations = observations.reshape(1, -1)
        actions = actions.reshape(1, -1)
        next_observations = next_observations.reshape(1, -1)

        self.memory.add(observations, actions, rewards, next_observations, dones)

        # Learn every `learn_every` time steps
        self.time_step += 1
        if self.time_step % self.LEARN_EVERY == 0:
            if len(self.memory) > self.BATCH_SIZE:
                for a_i, agent in enumerate(self.agents):
                    experiences = self.memory.sample()
                    self.learn(experiences, a_i, logger=logger)

    def learn(self, experiences, agent_number, logger=None):
        """Helper to pick actions from each agent for the `experiences` tuple that
        will be used to update the weights to agent with ID = `agent_number`.
        Each observation in the `experiences` tuple contains observations from each
        agent, so before using the tuple of update the weights of an agent, we need
        all agents to contribute in generating `next_actions` and `actions_pred`.
        This happens because the critic will take as its input the combined
        observations and actions from all agents."""
        next_actions = []
        actions_pred = []
        states, _, _, next_states, _ = experiences

        next_states = next_states.reshape(-1, self.num_agents, self.state_size)
        states = states.reshape(-1, self.num_agents, self.state_size)

        for a_i, agent in enumerate(self.agents):
            agent_id_tensor = self._get_agent_number(a_i)

            state = states.index_select(1, agent_id_tensor).squeeze(1)
            next_state = next_states.index_select(1, agent_id_tensor).squeeze(1)

            next_actions.append(agent.actor_target(next_state))
            actions_pred.append(agent.actor_local(state))

        next_actions = torch.cat(next_actions, dim=1).to(device)
        actions_pred = torch.cat(actions_pred, dim=1).to(device)

        agent = self.agents[agent_number]
        agent.learn(experiences, next_actions, actions_pred, logger=logger)

    def _get_agent_number(self, i):
        """Helper to get an agent's number as a Torch tensor."""
        return torch.tensor([i]).to(device)


class DDPGAgent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 agent_id,
                 handler,
                 actor_local=None,
                 actor_target=None,
                 actor_optimizer=None,
                 critic_local=None,
                 critic_target=None,
                 critic_optimizer=None,
                 seed=0,
                 device="cpu"):
        """Initialize a DDPGAgent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            agent_id (int): identifier for this agent
        """
        self.state_size = state_size
        self.action_size = action_size
        self.agent_id = agent_id
        self.seed = random.seed(seed)
        self.device = device

        # Actor Network (w/ Target Network)
        self.actor_local = actor_local if actor_local else Actor(state_size, action_size, seed).to(device)
        self.actor_target = actor_target if actor_target else Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = actor_optimizer if actor_optimizer else optim.Adam(self.actor_local.parameters(), lr=self.handler.LEARNING_RATE_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = critic_local if critic_local else Critic(state_size, action_size, seed).to(device)
        self.critic_target = critic_target if critic_target else Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = critic_optimizer if critic_optimizer else optim.Adam(self.critic_local.parameters(), lr=self.handler.LEARNING_RATE_CRITIC, weight_decay=self.handler.WEIGHT_DECAY)

        self.noise = OUNoise(action_size)
        self.noise_amplification = self.handler.NOISE_AMPLIFICATION
        self.noise_amplification_decay = self.handler.NOISE_AMPLIFICATION_DECAY

        # Ensure local and target networks have the same initial weight
        hard_update(self.actor_local, self.actor_target)
        hard_update(self.critic_local, self.critic_target)

    def __str__(self):
        """Helper to print network architecture for this agent's actors and critics."""
        print("Agent #{}".format(self.agent_id))
        print("Actor (Local):")
        print(self.actor_local)
        print("Actor (Target):")
        print(self.actor_target)
        print("Critic (Local):")
        print(self.critic_local)
        print("Critic (Target):")
        print(self.critic_target)
        if self.agent_id != NUM_AGENTS:
            print("_______________________________________________________________")

    def act(self, state, add_noise=False, logger=None):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample()
            self._decay_noise_amplification()

        return np.clip(action, -1, 1)

    def learn(self, experiences, next_actions, actions_pred, logger=None):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(next_state) -> action
            critic_target(next_state, next_action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            next_actions (list): next actions computed from each agent
            actions_pred (list): prediction for actions for current states from each agent
        """
        states, actions, rewards, next_states, dones = experiences
        agent_id_tensor = torch.tensor([self.agent_id - 1]).to(device)

        ### Update critic
        self.critic_optimizer.zero_grad()
        Q_targets_next = self.critic_target(next_states, next_actions)
        Q_targets = rewards.index_select(1, agent_id_tensor) + (self.handler.GAMMA * Q_targets_next * (1 - dones.index_select(1, agent_id_tensor)))
        Q_expected = self.critic_local(states, actions)
        # Minimize the loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        critic_loss.backward()
        self.critic_optimizer.step()

        ### Update actor
        self.actor_optimizer.zero_grad()
        # Minimize the loss
        actor_loss = -self.critic_local(states, actions_pred).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        ### Update target networks
        if self.opt_soft_update:
            soft_update(self.actor_local, self.actor_target, self.handler.TAU)
            soft_update(self.critic_local, self.critic_target, self.handler.TAU)
        elif self.time_step % self.handler.HARD_UPDATE_EVERY == 0:
            hard_update(self.actor_local, self.actor_target)
            hard_update(self.critic_local, self.critic_target)

        if logger:
            actor_loss = actor_loss.cpu().detach().item()
            critic_loss = critic_loss.cpu().detach().item()
            logger.add_scalars(
                'loss', {
                    "actor loss": actor_loss,
                    "critic loss": critic_loss,
                }, self.time_step
            )

    def _decay_noise_amplification(self):
        """Helper for decaying exploration noise amplification."""
        self.noise_amplification *= self.noise_amplification_decay
