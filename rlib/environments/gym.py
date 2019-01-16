import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import progressbar as pb

from rlib.environments.base import BaseEnvironment


class GymEnvironment(BaseEnvironment):
    def __init__(self, env_name,
                 seed=0,
                 logger=None,
                 gifs_recorder=None):
        r"""Initializes an OpenAI Gym environment

        Params
        ======
        env_name (str): Name of an OpenAI Gym environment
        seed (int): Environment seed
        logger (Logger): Tensorboard logger helper
        gifs_recorder (GIFRecorder): GIF recorder helper
        """
        self._env_name = env_name
        self.seed = seed
        self.logger = logger
        self.gifs_recorder = gifs_recorder

        self.start_env()

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.num_agents = 1  # TODO: Get this dynamically once we have capability
        self.observation_type = None
        self.observation_size = None
        if isinstance(self.observation_space, gym.spaces.box.Box):
            self.num_agents = 1 if len(self.observation_space.shape) == 1 else self.observation_space.shape[0]
            self.observation_size = self.observation_space.shape[0]
            self.observation_type = list
        elif isinstance(self.observation_space, gym.spaces.discrete.Discrete):
            self.observation_size = self.observation_space.n
            self.observation_type = int

        self.action_type = None
        self.action_size = None
        if isinstance(self.action_space, gym.spaces.box.Box):
            self.action_size = self.action_space.shape[0]
            self.action_type = list
        elif isinstance(self.action_space, gym.spaces.discrete.Discrete):
            self.action_size = self.action_space.n
            self.action_type = int

        self.episode_scores = []

    def __str__(self):
        r"""Helper to print information about this environment."""
        return ("{}\n{}\n{}".format(
            "Environment name: {}".format(self._env_name),
            "Observation space: {}".format(self.observation_space),
            "Action space: {}".format(self.action_space),
        ))

    def start_env(self):
        r"""Helper to start an environment"""
        self.env = gym.make(self._env_name)
        self.env.seed(self.seed)

    def close_env(self):
        r"""Helper to close an environment"""
        self.env.close()

    def set_algorithm(self, algorithm):
        self.algorithm = algorithm

    def train(self, num_episodes=100, max_t=None, add_noise=True, scores_window_size=100,
              save_every=None):
        r"""Trains agent(s) through interaction with this environment.

        Params
        ======
        num_episodes (int): Number of episodes to train the agent for
        max_t (int): Maximum number of timesteps in an episode
        add_noise (boolean): Add noise to actions
        scores_window_size (int): Window size for average score display
        save_every (int): Save state dicts every `save_every` episodes
        """
        widget = [
            "Episode: ", pb.Counter(), '/' , str(num_episodes), ' ',
            pb.Percentage(), ' ', pb.ETA(), ' ', pb.Bar(marker=pb.RotatingMarker()), ' ',
            'Rolling Average: ', pb.FormatLabel('')
        ]
        timer = pb.ProgressBar(widgets=widget, maxval=num_episodes).start()

        self.start_env()
        self.episode_scores = []

        for i_episode in range(1, num_episodes+1):
            current_average = self.get_current_average_score(scores_window_size)
            widget[12] = pb.FormatLabel(str(current_average)[:6])
            timer.update(i_episode)

            save_info = save_every and i_episode % save_every == 0

            observation = self.env.reset()
            scores = np.zeros(self.num_agents)
            rewards = []
            self.algorithm.reset()

            frames = []

            if save_info:
                frames.append(self.env.render("rgb_array"))

            t = 1
            while True:
                if max_t and t == max_t + 1:
                    break

                action = self.act(observation, add_noise=add_noise, logger=self.logger.writer)
                next_observation, reward, done, _ = self.env.step(action)
                self.algorithm.step(
                    observation, action, reward, next_observation, done,
                    logger=self.logger.writer
                )

                observation = next_observation
                scores += reward
                rewards.append(reward)
                t += 1

                if save_info:
                    frames.append(self.env.render("rgb_array"))

                if done:
                    break

            self.episode_scores.append(scores)
            self.algorithm.update(rewards, logger=self.logger.writer)

            if save_info:
                # TODO: Only save if best weights so far
                self.algorithm.save_state_dicts()

                if self.logger:
                    self.logger.writer.add_scalar("data/avg_rewards", np.mean(rewards), i_episode)

                if self.gifs_recorder:
                    self.gifs_recorder.save_gif("episode-{}.gif".format(i_episode), frames)

        self.close_env()
        if self.logger:
            self.logger.close()

        return self.episode_scores

    def test(self, num_episodes=5, load_state_dicts=False, render=True):
        r"""Runs trained agent(s) in this environment.

        Params
        ======
        num_episodes (int): Number of episodes to run this test for
        load_state_dicts (boolean): Load state dicts for each agent
        render (boolean): Render environment state
        """
        if load_state_dicts:
            self.algorithm.load_state_dicts()

        self.start_env()

        for i in range(1, num_episodes+1):
            observation = self.env.reset()
            scores = np.zeros(self.num_agents)

            while True:
                if render:
                    self.env.render()

                action = self.act(observation)
                next_observation, reward, done, _ = self.env.step(action)
                scores += reward

                if np.any(done):
                    break

                observation = next_observation

        self.close_env()
