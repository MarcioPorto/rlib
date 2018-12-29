import os

import torch


class Agent:
    # TODO: Move common agent initialization here
    # TODO: How to require anything inheriting from this to implement certain methods?

    def reset(self):
        if hasattr(self, "noise"):
            self.noise.reset()

    def act(self, state, add_noise=False):
        pass

    def step(self, state, action, reward, next_state, done):
        pass

    def update(self, rewards):
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
            raise Exception("You must provide an output directory to save state dict.")

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
