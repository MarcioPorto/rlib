class BaseEnvironment:
    def act(self, observations, add_noise=False):
        r"""Picks an action for each agent given their individual observations."""
        actions = []
        for agent, observation in zip(self.agents, observations):
            action = agent.act(observation, add_noise=add_noise)
            actions.append(action)
        return np.array(actions)

    def step(self, observation, action, reward, next_observation, done):
        r"""Step helper for each agent.

        By default, every agent receives the full description of the environment
        available. It is up to each individual agent implementation to decide
        what to do with that information.
        """
        for agent in self.agents:
            agent.step(observation, action, reward, next_observation, done)

    def reset_noise(self):
        r"""Resets each individual agent."""
        # TODO: Explore how to only do this if an agent has noise defined
        for agent in self.agents:
            agent.reset()

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
