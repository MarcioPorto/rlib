import os

import imageio
from tensorboardX import SummaryWriter


class Logger:
    # TODO: Consider better names for these
    # TODO: Consider usage of these
    VERBOSITY = ['DEBUG', 'INFO']

    def __init__(self, path: str, comment: str = None, verbosity: str = 'DEBUG', 
                 experiment_name: str = None) -> None:
        """ Initializes a Logger for training statistics.

        Args:
            path (str): Location where events will be stored
            comment (str): Extra description of this experiment
        """
        self.path = path
        self.full_path = os.path.join(
            self.path, 'logs'
        )
        os.makedirs(self.full_path, exist_ok=True)

        self.comment = comment
        self.verbosity = verbosity
        self.experiment_name = experiment_name

        self.writer = SummaryWriter(self.full_path, self.comment)

    def close(self) -> None:
        """ Closes SummaryWriter. """
        self.writer.close()

    def add_scalar(self, name: str, data, index: int) -> None:
        self.writer.add_scalar("data/{}".format(name), data, index)

    def add_scalars(self, name: str, data, index: int) -> None:
        self.writer.add_scalars("data/{}".format(name), data, index)

    def add_video(self):
        raise NotImplementedError()


class GIFRecorder:
    """ Helper to record GIFs of the agent during training. """

    def __init__(self, path: str, duration: float = 0.5, 
                 experiment_name: str = None) -> None:
        """
        Args:
            path (str): Location where to store GIFs
            duration (float): Duration of GIFs
        """
        self.path = path
        self.full_path = os.path.join(
            self.path, 'gifs'
        )
        os.makedirs(self.full_path, exist_ok=True)

        self.duration = duration
        self.experiment_name = experiment_name

    def save_gif(self, filename: str, frames) -> None:
        """
        Args:
            filename (str): Name of file where GIF will be saved
            frames (list): Environment frames
        """
        imageio.mimsave(
            os.path.join(self.full_path, filename), 
            frames, 
            duration=self.duration
        )


def hard_update(local_model, target_model) -> None:
    r"""Hard updates model parameters.

    θ_target = θ_local

    Args:
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(local_param.data)


def soft_update(local_model, target_model, tau) -> None:
    r"""Soft updates model parameters.

    θ_target = τ * θ_local + (1 - τ) * θ_target

    Args:
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
