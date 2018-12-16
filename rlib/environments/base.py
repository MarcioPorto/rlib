class BaseEnvironment:
    def plot_scores(self, scores=None, env_solved_score=None):
        r"""Plots scores for each episode.

        Params
        ======
        scores (list): List of scores to plot (one per episode)
        env_solved_score (float): If provided, shows the score where the environment is considered solved
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

    def get_rolling_score_averages(self):
        rolling_score_averages = []
        # TODO: Loop over a window of scores and get their mean
        average_score = np.mean(scores_deque)
        rolling_score_averages.append(average_score)
