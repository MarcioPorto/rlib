import numpy as np
from tensorboardX import SummaryWriter


class BaseEnvironment:
    def act(self, observations, add_noise=False, logger=None):
        """Picks an action for each agent given their individual observations."""
        action = self.algorithm.act(observations, add_noise=add_noise, logger=logger)

        # TODO: Fix this
        return action

        # if self.action_type == list and not isinstance(action, np.ndarray):
        #     return np.array([action])
        # else:
        #     return action

    def plot_scores(self, scores=None, env_solved_score=None):
        """Plots scores for each episode.

        Args:
            scores (list): List of scores to plot (one per episode).
            env_solved_score (float): If provided, shows the score where the
                                      environment is considered solved.
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
        """Get max score for the current episode."""
        return np.max(self.scores)

    def get_rolling_score_averages(self, window):
        """Helper to get mean score in a rolling window across episode scores."""
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
        """Get the current average score for this environment."""
        s = len(self.episode_scores)
        return np.mean(self.episode_scores) if s <= window else np.mean(self.episode_scores[s-window:])
