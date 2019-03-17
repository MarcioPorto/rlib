import random
from collections import namedtuple, deque

import numpy as np
import torch


class ReplayBuffer:
    """ Fixed-size buffer to store experience tuples. """

    def __init__(self, buffer_size, batch_size, device, seed=0):
        """ Initializes a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            device (str): PyTorch device
            seed (int): random seed
        """
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """ Adds a new experience to memory. """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """ Randomly samples a batch of experiences from memory. """
        experiences = random.sample(self.memory, k=self.batch_size)

        st = [e.state for e in experiences if e is not None]

        try:
            assert len(st[0].shape) > 1
            state = np.asarray(st)
            action = np.asarray([e.action for e in experiences if e is not None])
            reward = np.asarray([e.reward for e in experiences if e is not None])
            next_state = np.asarray([e.next_state for e in experiences if e is not None])
            done = np.asarray([e.done for e in experiences if e is not None])
        except (AttributeError, AssertionError):
            state = np.vstack(st)
            action = np.vstack([e.action for e in experiences if e is not None])
            reward = np.vstack([e.reward for e in experiences if e is not None])
            next_state = np.vstack([e.next_state for e in experiences if e is not None])
            done = np.vstack([e.done for e in experiences if e is not None])

        states = torch.from_numpy(state).float().to(self.device)
        actions = torch.from_numpy(action).float().to(self.device)
        rewards = torch.from_numpy(reward).float().to(self.device)
        next_states = torch.from_numpy(next_state).float().to(self.device)
        dones = torch.from_numpy(done.astype(np.uint8)).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """ Returns the current size of the memory deque. """
        return len(self.memory)
