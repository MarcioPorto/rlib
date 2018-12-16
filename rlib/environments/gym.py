from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np

from rlib.environments.base import BaseEnvironment


class GymEnvironment(BaseEnvironment):
    def __init__(self, env_name, seed=0):
        self.env = gym.make(env_name)
        self.env.seed(seed)
        # TODO: Expose state and action sizes
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.episode_scores = []

        # TODO: Handle environments with multiple agents
        self.agents = []

    # TODO: Check that these properties are working
    @property
    def num_agents(self):
        return len(self.agents)

    @property
    def observation_size(self):
        return self.observation_space.shape

    @property
    def action_size(self):
        return self.action_space.n

    def set_agents(self, agents):
        self.agents = agents

    def train(self, num_episodes, max_t, scores_deque_size=100):
        self.episode_scores = []

        for i_episode in range(1, num_episodes+1):
            state = self.env.reset()
            scores = np.zeros(self.num_agents)

            # TODO: Explore how to only do this if an agent has noise defined
            # TODO: Handle environments with multiple agents
            # self.agent.reset()

            # TODO: Refactor to act as while when max_t is None
            for t in range(1, max_t+1):
                # TODO: Handle environments with multiple agents
                action = self.agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.step(state, action, reward, next_state, done)
                state = next_state
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
