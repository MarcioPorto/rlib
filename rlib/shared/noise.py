import copy
import random

import numpy as np


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        """Initializes parameters and noise process.

        Args:
            size: Size.
            seed: Seed.
            mu (float): Mu.
            theta (float): Theta.
            sigma (float): Sigma.

        Returns:
            An instance of OUNoise.
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self) -> None:
        """Resets the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Updates internal state and return it as a noise sample.
        
        Returns:
            A noise sample.
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.normal(loc=0, scale=1) for _ in range(len(x))])
        self.state = x + dx
        return self.state
