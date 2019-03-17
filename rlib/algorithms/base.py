import os
from abc import ABC, abstractmethod

import torch


class Agent(ABC):
    REQUIRED_HYPERPARAMETERS = {}
    ALGORITHM = None

    def __init__(self, *args, **kwargs):
        if "new_hyperparameters" in kwargs:
            if isinstance(kwargs["new_hyperparameters"], dict):
                self._set_hyperparameters(kwargs["new_hyperparameters"])

        # Converts each hyperparameter into an attribute
        # This minimizes the code written to use the hyperparameters
        for key, value in self.REQUIRED_HYPERPARAMETERS.items():
            setattr(self, key.upper(), value)

    @abstractmethod
    def origin(self):
        pass
    
    @abstractmethod
    def description(self):
        pass

    def reset(self):
        if hasattr(self, "noise"):
            self.noise.reset()

    def act(self, state, add_noise=False, logger=None):
        """Default `act` implementation"""
        pass

    def step(self, state, action, reward, next_state, done, logger=None):
        """Default `step` implementation"""
        pass

    def learn(self, experiences, logger=None):
        """Default `learn` implementation"""
        pass

    def update(self, rewards, logger=None):
        """Default `update` implementation"""
        pass

    def get_hyperparameters(self):
        """Returns the current state of the required hyperparameters"""
        return self.REQUIRED_HYPERPARAMETERS

    def _set_hyperparameters(self, new_hyperparameters):
        """Adds user defined hyperparameter values to the list required
        hyperparameters.
        """
        for key, value in new_hyperparameters.items():
            if key in self.REQUIRED_HYPERPARAMETERS.keys():
                self.REQUIRED_HYPERPARAMETERS[key] = value

    def save_state_dicts(self):
        """Save state dicts to file."""
        if not self.model_output_dir:
            return

        for sd in self.state_dicts:
            torch.save(
                comb[0].state_dict(),
                os.path.join(self.model_output_dir, "{}.pth".format(sd[1]))
            )

    def load_state_dicts(self):
        """Load state dicts from file."""
        if not self.model_output_dir:
            raise Exception("You must provide an input directory to load state dict.")

        for sd in self.state_dicts:
            comb[0].load_state_dict(
                torch.load(os.path.join(self.model_output_dir, "{}.pth".format(sd[1])))
            )
