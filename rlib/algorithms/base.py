import os

import torch
from tensorboardX import SummaryWriter


class Agent:
    REQUIRED_HYPERPARAMETERS = {}
    ALGORITHM = None

    def __init__(self, *args, **kwargs):
        if "new_hyperparameters" in kwargs:
            if isinstance(kwargs["new_hyperparameters"], dict):
                self._set_hyperparameters(kwargs["new_hyperparameters"])

        if "enable_logger" in kwargs and kwargs["enable_logger"] == True:
            logger_path = kwargs["logger_path"] if "logger_path" in kwargs and kwargs["logger_path"] is not None else self.ALGORITHM
            logger_comment = kwargs["logger_comment"] if "logger_comment" in kwargs and kwargs["logger_comment"] is not None else ""
            self.logger = SummaryWriter(logger_path, logger_comment)
        else:
            self.logger = None

        # Converts each hyperparameter into an attribute
        # This minimizes the code written to use the hyperparameters
        for key, value in self.REQUIRED_HYPERPARAMETERS.items():
            setattr(self, key.upper(), value)

    def reset(self):
        if hasattr(self, "noise"):
            self.noise.reset()

    def tear_down(self):
        """Called at the end of training loop to close the SummaryWriter"""
        if self.logger:
            self.logger.close()

    def act(self, state, add_noise=False):
        """Default `act` implementation"""
        pass

    def step(self, state, action, reward, next_state, done):
        """Default `step` implementation"""
        pass

    def update(self, rewards):
        """Default `update` implementation"""
        pass

    def get_hyperparameters(self):
        r"""Returns the current state of the required hyperparameters"""
        return self.REQUIRED_HYPERPARAMETERS

    def _set_hyperparameters(self, new_hyperparameters):
        r"""Adds user defined hyperparameter values to the list required
        hyperparameters.
        """
        for key, value in new_hyperparameters.items():
            if key in self.REQUIRED_HYPERPARAMETERS.keys():
                self.REQUIRED_HYPERPARAMETERS[key] = value

    def save_state_dicts(self):
        r"""Save state dicts to file."""
        if not self.model_output_dir:
            return

        for sd in self.state_dicts:
            torch.save(
                comb[0].state_dict(),
                os.path.join(self.model_output_dir, "{}.pth".format(sd[1]))
            )

    def load_state_dicts(self):
        r"""Load state dicts from file."""
        if not self.model_output_dir:
            raise Exception("You must provide an input directory to load state dict.")

        for sd in self.state_dicts:
            comb[0].load_state_dict(
                torch.load(os.path.join(self.model_output_dir, "{}.pth".format(sd[1])))
            )
