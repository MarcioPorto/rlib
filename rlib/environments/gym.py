import gym
import matplotlib.pyplot as plt
import numpy as np
import progressbar as pb

from rlib.environments.base import BaseEnvironment


class GymEnvironment(BaseEnvironment):
    def __init__(self, env_name,
                 seed=0,
                 enable_logger=False,
                 logger_path=None,
                 logger_comment=None,):
        r"""Initializes an OpenAI Gym environment

        Params
        ======
        env_name (str): Name of an OpenAI Gym environment
        seed (int): Environment seed
        enable_logger (bool): Enable Tensorboard logger
        logger_path (str): Location to store logs
        logger_comment (str): Logs description
        """
        super(GymEnvironment, self).__init__(
            env_name=env_name,
            enable_logger=enable_logger,
            logger_path=logger_path,
            logger_comment=logger_comment
        )

        self._env_name = env_name
        self.seed = seed

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

            observation = self.env.reset()
            scores = np.zeros(self.num_agents)
            rewards = []
            self.algorithm.reset()

            t = 1
            while True:
                if max_t and t == max_t + 1:
                    break

                action = self.act(observation, add_noise=add_noise)
                next_observation, reward, done, _ = self.env.step(action)
                self.algorithm.step(
                    observation, action, reward, next_observation, done,
                    logger=self.logger
                )

                observation = next_observation
                scores += reward
                rewards.append(reward)
                t += 1

                if done:
                    break

            self.episode_scores.append(scores)
            self.algorithm.update(rewards)

            if save_every and i_episode % save_every == 0:
                # TODO: Only save if best weights
                self.algorithm.save_state_dicts()

                if self.logger:
                    self.logger.add_scalar("data/avg_rewards", np.mean(rewards), i_episode)

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
