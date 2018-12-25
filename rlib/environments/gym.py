import gym
import matplotlib.pyplot as plt
import numpy as np
import progressbar as pb

from rlib.environments.base import BaseEnvironment


class GymEnvironment(BaseEnvironment):
    def __init__(self, env_name, seed=0):
        self.env = gym.make(env_name)
        self.env.seed(seed)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.episode_scores = []

    @property
    def observation_size(self):
        return self.observation_space.shape[0]

    @property
    def action_size(self):
        return self.action_space.shape[0]

    @property
    def num_env_agents(self):
        return 1 if len(self.observation_space.shape) == 1 else self.observation_space.shape[0]

    def set_agents(self, agents):
        r"""Sets the agents that will be used in this environment."""
        if not isinstance(agents, list):
            agents = [agents]
        if len(agents) > self.num_env_agents:
            raise ValueError("Cannot have more agents than the environment can handle.")
        self.agents = agents

    def train(self, num_episodes=100, max_t=None, add_noise=True, scores_window_size=100):
        widget = [
            "Episode: ", pb.Counter(), '/' , str(num_episodes), ' ',
            pb.Percentage(), ' ', pb.ETA(), ' ', pb.Bar(marker=pb.RotatingMarker()), ' ',
            'Rolling Average: ', pb.FormatLabel('')
        ]
        timer = pb.ProgressBar(widgets=widget, maxval=num_episodes).start()

        self.episode_scores = []

        for i_episode in range(1, num_episodes+1):
            # TODO: Check progress bar
            current_average = self.get_current_average_score(scores_window_size)
            widget[12] = pb.FormatLabel(str(current_average)[:6])
            timer.update(i_episode)

            observation = self.env.reset()
            scores = np.zeros(self.num_agents)
            self.reset_agents()

            t = 1
            while True:
                if max_t and t == max_t + 1:
                    break

                action = self.act(observation, add_noise=add_noise)
                next_observation, reward, done, _ = self.env.step(action)
                self.step(observation, action, reward, next_observation, done)

                observation = next_observation
                scores += reward
                t += 1

                if done:
                    break

            self.episode_scores.append(scores)

            # TODO: Add a save_every option

        return self.episode_scores

    def test(self, num_episodes=5, load_state_dicts=False):
        # TODO: Use load_state_dicts
        # TODO: Add game scores for competitive environment?

        for i in range(1, num_episodes+1):
            observation = self.env.reset()
            scores = np.zeros(self.num_env_agents)

            while True:
                self.env.render()
                action = self.act(observation)
                next_observation, reward, done, _ = self.env.step(action)
                scores += reward

                if np.any(done):
                    break

                observation = next_observation
