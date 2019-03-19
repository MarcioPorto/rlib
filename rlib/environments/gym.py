import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import progressbar as pb
from typing import List

from rlib.environments.base import BaseEnvironment
from rlib.shared.utils import Logger, GIFRecorder


class GymEnvironment(BaseEnvironment):
    def __init__(self, 
                 env,
                 algorithm,
                 seed: int = 0,
                 logger: Logger = None,
                 gifs_recorder: GIFRecorder = None):
        """Initializes an OpenAI Gym environment

        Args:
            env: A Gym environment.
            algorithm: An instance of an algorithm.
            seed (int): Environment seed.
            logger (Logger): Tensorboard logger helper.
            gifs_recorder (GIFRecorder): GIF recorder helper.

        Returns:
            An instance of GymEnvironment.
        """
        self.env = env
        self._env_name = self.env.unwrapped.spec.id
        self.algorithm = algorithm
        self.seed = seed
        self.logger = logger
        self.gifs_recorder = gifs_recorder

        # TODO: Get this dynamically once we have capability
        self.num_agents = 1

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.episode_scores = []

    def __str__(self) -> str:
        """Helper to print information about this environment.
        
        Returns:
            A description of this GymEnvironment.
        """
        return ("{}\n{}\n{}".format(
            "Environment name: {}".format(self._env_name),
            "Observation space: {}".format(self.observation_space),
            "Action space: {}".format(self.action_space),
        ))

    def start_env(self) -> None:
        """Helper to start an environment."""
        self.env = gym.make(self._env_name)
        self.env.seed(self.seed)

    def close_env(self) -> None:
        """Helper to close an environment."""
        self.env.close()

    def is_observation_box(self):
        return isinstance(self.observation_space, gym.spaces.box.Box)

    def is_observation_discrete(self):
        return isinstance(self.observation_space, gym.spaces.discrete.Discrete)

    def is_action_box(self):
        return isinstance(self.action_space, gym.spaces.box.Box)

    def is_action_discrete(self):
        return isinstance(self.action_space, gym.spaces.discrete.Discrete)

    def normalize_observation(self, obs):
        """ Normalizes the observation received from the environment.

        NOTE: Users must override this function if any transformation is needed. 

        Returns:
            The normalized observation.
        """
        return obs

    def train(self, num_episodes: int = 100, max_t: int = None, add_noise: bool = True, 
              scores_window_size: int = 100, save_every: int = None) -> List[float]:
        """Trains agent(s) through interaction with this environment.

        Args:
            num_episodes (int): Number of episodes to train the agent for.
            max_t (int): Maximum number of timesteps in an episode.
            add_noise (boolean): Add noise to actions.
            scores_window_size (int): Window size for average score display.
            save_every (int): Save state dicts every `save_every` episodes.

        Returns:
            The scores for each episode.
        """
        widget = [
            "Episode: ", pb.Counter(), '/' , str(num_episodes), ' ',
            pb.Percentage(), ' ', pb.ETA(), ' ', pb.Bar(marker=pb.RotatingMarker()), ' ',
            'Rolling Average: ', pb.FormatLabel('')
        ]
        timer = pb.ProgressBar(widgets=widget, maxval=num_episodes).start()

        self.start_env()
        self.episode_scores = []

        for i_episode in range(1, num_episodes + 1):
            current_average = self.get_current_average_score(scores_window_size)
            widget[12] = pb.FormatLabel(str(current_average)[:6])
            timer.update(i_episode)

            save_info = save_every and i_episode % save_every == 0

            observation = self.env.reset()
            observation = self.normalize_observation(observation)
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

                action = self.act(observation, add_noise=add_noise, logger=self.logger)
                next_observation, reward, done, _ = self.env.step(action)
                next_observation = self.normalize_observation(next_observation)

                self.algorithm.step(
                    observation, action, reward, next_observation, done,
                    logger=self.logger
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
            self.algorithm.update(rewards, logger=self.logger)

            if save_info:
                # TODO: Only save if best weights so far
                self.algorithm.save_state_dicts()

                if self.logger:
                    self.logger.add_scalar("avg_rewards", np.mean(rewards), i_episode)

                if self.gifs_recorder:
                    self.gifs_recorder.save_gif("episode-{}.gif".format(i_episode), frames)

        self.close_env()
        
        if self.logger:
            self.logger.close()

        return self.episode_scores

    def test(self, num_episodes: int = 5, load_state_dicts: bool = False, 
             render: bool = True) -> None:
        """Runs trained agent(s) in this environment.

        Args:
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
