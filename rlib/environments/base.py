import numpy as np


class BaseEnvironment:
    @property
    def num_agents(self):
        return len(self.agents)

    def act(self, observations, add_noise=False):
        r"""Picks an action for each agent given their individual observations."""
        if self.num_env_agents > 1:
            actions = []
            for agent, observation in zip(self.agents, observations):
                action = agent.act(observation, add_noise=add_noise)
                actions.append(action)
            return np.array(actions)
        else:
            action = self.agents[0].act(observations, add_noise=add_noise)
            if self.action_type == list:
                return np.array([action])
            else:
                return action

    def step(self, observation, action, reward, next_observation, done):
        r"""Step helper for each agent.

        By default, every agent receives the full description of the environment
        available. It is up to each individual agent implementation to decide
        what to do with that information.
        """
        for agent in self.agents:
            agent.step(observation, action, reward, next_observation, done)

    def update(self, rewards):
        # TODO: Make sure the zip actually works
        if self.num_env_agents > 1:
            for agent, reward in zip(self.agents, rewards):
                agent.update(reward)
        else:
            self.agents[0].update(rewards)

    def reset_agents(self):
        r"""Resets each individual agent."""
        for agent in self.agents:
            agent.reset()

    def save_state_dicts(self):
        r"""Wrapper to save state dicts for each individual agent."""
        for agent in self.agents:
            agent.save_state_dicts()

    def load_state_dicts(self):
        r"""Wrapper to load state dicts for each individual agent."""
        for agent in self.agents:
            agent.load_state_dicts()

    def plot_scores(self, scores=None, env_solved_score=None):
        r"""Plots scores for each episode.

        Params
        ======
        scores (list): List of scores to plot (one per episode)
        env_solved_score (float): If provided, shows the score where the
                                  environment is considered solved
        """
        if not scores:
            scores = self.episode_scores

        fig = plt.figure()
        ax = fig.add_subplot(111)

        plt.plot(np.arange(1, len(scores) + 1), scores, label="Episode Score")

        if env_solved_score:
            # This line indicates the score at which the environment is considered solved
            plt.axhline(y=0.5, color="r", linestyle="-", label="Environment Solved")

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.ylabel("Score")
        plt.xlabel("Episode #")
        plt.show()

    def get_max_score_per_episode(self):
        return np.max(self.scores)

    def get_rolling_score_averages(self, window):
        r"""Helper to get mean score in a rolling window across episode
        scores.
        """
        rolling_score_averages = []
        for i in range(len(self.episode_scores)):
            if i <= window:
                s = self.episode_scores[:i+1]
                average_score = np.mean(s)
            else:
                s = self.episode_scores[i-window:i+1]
                average_score = np.mean(s)
            rolling_score_averages.append(average_score)

    def get_current_average_score(self, window):
        s = len(self.episode_scores)
        return np.mean(self.episode_scores) if s <= window else np.mean(self.episode_scores[s-window:])
