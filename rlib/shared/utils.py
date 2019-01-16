import os

import imageio
from tensorboardX import SummaryWriter


class Logger:
    def __init__(self, path, comment=None):
        """
        Args:
            path (str): Location where events will be stored
            comment (str): Extra description of this experiment
        """
        # TODO: Add an option for experiment name?

        self.path = path
        self.comment = comment
        self.writer = SummaryWriter(self.path, self.comment)

    def close(self):
        self.writer.close()

    def add_scalar(self, name, data, index):
        self.writer.add_scalar("data/{}".format(name), data, index)

    def add_scalars(self, name, data, index):
        self.writer.add_scalars("data/{}".format(name), data, index)


class GIFRecorder:
    """Helper to record GIFs of the environment during training."""

    def __init__(self, path, duration=0.4):
        """
        Args:
            path (str): Location where to store GIFs
            duration (float): Duration of GIFs
        """
        self.path = path
        self.duration = duration
        os.makedirs(self.path, exist_ok=True)

    def save_gif(self, filename, frames):
        """
        Args:
            filename (str): Name of file where GIF will be saved
            frames (list): Environment frames
        """
        # TODO: FIX
        # frames_vector = torch.from_numpy(np.array(frames))
        # self.logger.add_video("video/episode-{}".format(i_episode), frames_vector)

        imageio.mimsave(os.path.join(self.path, filename), frames, duration=self.duration)


def hard_update(local_model, target_model):
    r"""Hard updates model parameters.

    θ_target = θ_local

    Args:
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(local_param.data)


def soft_update(local_model, target_model, tau):
    r"""Soft updates model parameters.

    θ_target = τ * θ_local + (1 - τ) * θ_target

    Args:
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
