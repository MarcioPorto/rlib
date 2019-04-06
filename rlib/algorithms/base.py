import os
from abc import ABC, abstractmethod

import torch


class Agent(ABC):
    """Base implementation for all Agents.
    
    All other agents must inherit this class, regardless of framework being used.
    """

    REQUIRED_HYPERPARAMETERS = {}
    ALGORITHM = None

    def __init__(self, *args, **kwargs):
        """Shared Agent initialization.
        
        All agents must call this method as a first initialization step.
        """
        if "new_hyperparameters" in kwargs:
            if isinstance(kwargs["new_hyperparameters"], dict):
                self._set_hyperparameters(kwargs["new_hyperparameters"])

        if "logger" not in kwargs:
            raise ValueError('Make sure to pass a logger argument to this method.')

        self.logger = kwargs["logger"]

        # Converts each hyperparameter into an attribute
        # This minimizes the code written to use the hyperparameters
        for key, value in self.REQUIRED_HYPERPARAMETERS.items():
            setattr(self, key.upper(), value)

    @abstractmethod
    def origin(self):
        """Returns a string indicating the source of this algorithm."""
        pass
    
    @abstractmethod
    def description(self):
        """Returns a brief description of this algorithm."""
        pass

    def reset(self):
        """Resets the state of the agent at the beginning of each episode."""
        if hasattr(self, "noise"):
            self.noise.reset()

    def act(self, state, add_noise: bool = False):
        """Default `act` implementation.
        
        This method uses the current policy learned by the agent to pick the
        agent's next action in the environment. 
        """
        pass

    def step(self, state, action, reward, next_state, done):
        """Default `step` implementation.

        Called after an agent takes an action in the environment, this method is 
        used to perform actions on the most recent (s, a, r, s', d) tuple.
        """
        pass

    def learn(self, experiences):
        """Default `learn` implementation.

        Given a list of experiences (usually from a sample of the experiences in
        a ReplayBuffer), this method updates the agent's model of the environment.
        """
        pass

    def update(self, states, actions, rewards):
        """Default `update` implementation.

        Given a list of rewards collected over the course of an entire episode,
        this method updates the agent's model of the environment.
        """
        pass

    def get_hyperparameters(self):
        """Returns the current state of the required hyperparameters.
        
        Returns:
            A dictionary of hyperparameters.
        """
        return self.REQUIRED_HYPERPARAMETERS

    def _set_hyperparameters(self, new_hyperparameters):
        """Adds user defined hyperparameter values to the list required hyperparameters.

        Any keys not recognized by the algorithm will be ignored.

        Args:
            new_hyperparameters: A dictionary containing the new hyperparameter values.
        """
        for key, value in new_hyperparameters.items():
            if key in self.REQUIRED_HYPERPARAMETERS.keys():
                self.REQUIRED_HYPERPARAMETERS[key] = value

    def save_state_dicts(self):
        """Save state dictionaries to file."""
        # TODO: Add TensorFlow support
        if not self.model_output_dir:
            return

        for sd in self.state_dicts:
            torch.save(
                sd[0].state_dict(),
                os.path.join(self.model_output_dir, "{}.pth".format(sd[1]))
            )

    def load_state_dicts(self):
        """Load state dictionaries from file."""
        # TODO: Add TensorFlow support
        if not self.model_output_dir:
            raise Exception("You must provide an input directory to load state dict.")

        for sd in self.state_dicts:
            sd[0].load_state_dict(
                torch.load(os.path.join(self.model_output_dir, "{}.pth".format(sd[1])))
            )
