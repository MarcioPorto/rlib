import random
from collections import namedtuple, deque

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size: int, batch_size: int, seed: int = 0):
        """ Initializes a ReplayBuffer object.

        Args:
            buffer_size (int): maximum size of buffer.
            batch_size (int): size of each training batch.
            seed (int): random seed.

        Returns:
            An instance of ReplayBuffer.
        """
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience", 
            field_names=["state", "action", "reward", "next_state", "done"]
        )

    def add(self, state, action, reward, next_state, done):
        """Adds a new experience to memory.
        
        Args:
            state: Environment state.
            action: Environment action.
            reward: Reward for the actions above.
            next_state: Next environment state.
            done (bool): Boolean indicating if the environment has terminated. 
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly samples a batch of experiences from memory. 
        
        Returns:
            A tuple of (states, actions, rewards, next_states, dones).
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        st = [e.state for e in experiences if e is not None]

        try:
            assert len(st[0].shape) > 1
            states = np.asarray(st)
            actions = np.asarray([e.action for e in experiences if e is not None])
            rewards = np.asarray([e.reward for e in experiences if e is not None])
            next_states = np.asarray([e.next_state for e in experiences if e is not None])
            dones = np.asarray([e.done for e in experiences if e is not None])
        except (AttributeError, AssertionError):
            states = np.vstack(st)
            actions = np.vstack([e.action for e in experiences if e is not None])
            rewards = np.vstack([e.reward for e in experiences if e is not None])
            next_states = np.vstack([e.next_state for e in experiences if e is not None])
            dones = np.vstack([e.done for e in experiences if e is not None])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Gets the current size of the memory deque.

        Returns: 
            Current size of the memory deque.
        """
        return len(self.memory)
