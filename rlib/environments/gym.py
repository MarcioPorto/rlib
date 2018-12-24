from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np

from rlib.environments.base import BaseEnvironment


class GymEnvironment(BaseEnvironment):
    def __init__(self, env_name, seed=0):
        self.env = gym.make(env_name)
        self.env.seed(seed)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.episode_scores = []

    @property
    def num_agents(self):
        return len(self.agents)

    @property
    def observation_size(self):
        return self.observation_space.shape[0]

    @property
    def action_size(self):
        return self.action_space.shape[0]

    def set_agents(self, agents):
        r"""Sets the agents that will be used in this environment."""
        if not isinstance(agents, list):
            agents = [agents]
        self.agents = agents

    def train(self, num_episodes=100, max_t=100, scores_deque_size=100):
        self.episode_scores = []

        for i_episode in range(1, num_episodes+1):
            observation = self.env.reset()
            scores = np.zeros(self.num_agents)
            self.reset_noise()

            # TODO: Refactor to act as while when max_t is None
            for t in range(1, max_t+1):
                action = self.act(observation, add_noise=True)
                next_observation, reward, done, _ = self.env.step(action)
                self.step(observation, action, reward, next_observation, done)

                observation = next_observation
                scores += reward

                if done:
                    break

            self.episode_scores.append(scores)

            # TODO: Add a save_every option
            # TODO: Add a print progress option (progressbar)

        return self.episode_scores

    def test(self, load_state_dicts):
        # TODO: Implement
        # TODO: Handle environments with multiple agents
        pass
