import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

from rlib.algorithms.base import Agent
from rlib.networks.pytorch.network import Policy


class PPOAgent(Agent):
    """PPO Agent implementation."""

    # TODO: Consider how to extend this to accept multiple agents?
    # TODO: Ensure that this cannot be changed in other ways
    # TODO: Look up original value for these params
    REQUIRED_HYPERPARAMETERS = {
        "gamma": 1.0,
        "learning_rate": 1e-2,
        "num_updates": 4,
        "epsilon": 0.1,
        "beta": 0.01
    }

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 policy=None,
                 optimizer=None,
                 new_hyperparameters=None,
                 seed: int = 0,
                 device: str = "cpu",
                 model_output_dir=None,
                 logger=None):
        """Initialize an PPOAgent object.

        Args:
            state_size (int): dimension of each state.
            action_size (int): dimension of each action.
            policy (torch.nn.Module): Policy model.
            optimizer (torch.optim): Model optimizer.
            new_hyperparameters (dict): New hyperparameter values.
            seed (int): Random seed.
            device (str): Identifier for device to be used by PyTorch.
            model_output_dir (str): Directory where state dicts will be saved to.
            logger (Logger): Tensorboard logger helper.

        Returns:
            An instance of PPOAgent.
        """
        super(PPOAgent, self).__init__(
            new_hyperparameters=new_hyperparameters,
            logger=logger
        )

        random.seed(seed)

        self.seed = seed
        self.time_step = 0

        self.device = device

        if policy:
            self.policy = policy.to(self.device)
        else:
            self.policy = Policy(
                s_size=state_size,
                a_size=action_size
            ).to(self.device)

        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(
                self.policy.parameters(),
                lr=self.LEARNING_RATE
            )

        self.state_dicts = [
            (self.policy, "policy_params"),
            (self.optimizer, "optimizer_params"),
        ]

        self.model_output_dir = model_output_dir

    def __str__(self) -> str:
        """Helper to output network architecture for the agent.
        
        Returns:
            A string representation of this algorithm.
        """
        return ("{}\n{}".format(
            "Policy:",
            self.policy,
        ))

    def origin(self) -> str:
        """Helper to get the original paper for this algorithm.

        Returns: 
            The original paper for this algorithm.
        """
        return 'https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf'
    
    def description(self) -> str:
        """Helper to get a brief description of this algorithm.

        Returns:
            A brief description of this algorithm.
        """
        description = (
            'The key idea underlying policy gradients is to push up the '
            'probabilities of actions that lead to higher return, and '
            'push down the probabilities of actions that lead to lower '
            'return, until you arrive at the optimal policy.'
        )
        return description

    def initialize(self, num_agents: int, num_workers: int) -> None:
        self.num_agents = num_agents
        self.num_workers = num_workers

        self.saved_log_probs = [[] for _ in range(self.num_workers)]
        self.saved_probs = [[] for _ in range(self.num_workers)]

    def reset(self) -> None:
        """Reset PPOAgent."""
        self.saved_log_probs = [[] for _ in range(self.num_workers)]
        self.saved_probs = [[] for _ in range(self.num_workers)]

    def step(self, state, action, reward, next_state, done) -> None:
        """Increment step count."""
        self.time_step += 1

    def act(self, states, add_noise: bool = False):
        """Chooses an action for the current state based on the current policy.

        Args:
            state: The current state of the environment for each worker.
            add_noise (bool): Controls addition of noise.

        Returns: 
            Action for given state as per current policy.
        """
        actions = []

        for worker_id, state in enumerate(states):
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            probs = self.policy.forward(state).cpu()
            m = Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)

            self.saved_log_probs[worker_id].append(log_prob)
            self.saved_probs[worker_id].append(probs[0][action.item()])

            actions.append(action.item())
            
        return actions

    def update(self, states, actions, rewards) -> None:
        """Updates policy.

        Args:
            states: Environment states trajectories.
            actions: Environment rewards trajectories.
            rewards: Environment rewards trajectories.
        """
        for _ in range(self.NUM_UPDATES):
            policy_loss = []

            for worker_id in range(self.num_workers):
                states_l = states[worker_id]
                actions_l = actions[worker_id]
                rewards_l = rewards[worker_id]

                # Future rewards entry for each timestep
                R_future = []

                for t in range(len(states_l)):
                    future_rewards = rewards_l[t:]
                    discounts = self.GAMMA ** np.arange(len(future_rewards) + 1)
                    discounted_future_rewards = sum([a * b for a, b in zip(discounts, future_rewards)])
                    R_future.append(discounted_future_rewards)

                # TODO: Normalize rewards using mean and std

                # print('Worker {}'.format(worker_id))
                policy_loss_l = []

                new_probs = []
                old_probs = []
                
                # TODO: Change this enumeration
                for t, log_prob in enumerate(self.saved_log_probs[worker_id]):
                    # old_prob = self.saved_probs[worker_id][t]
                    old_prob = log_prob

                    # TODO: Should this be using no_grad?
                    state = torch.from_numpy(states_l[t]).float().unsqueeze(0).to(self.device)
                    probs = self.policy.forward(state).cpu()

                    # new_prob = probs[0][actions_l[t]]
                    m = Categorical(probs)
                    a = torch.Tensor([actions_l[t]])
                    new_prob = m.log_prob(a)

                    new_probs.append(new_prob)
                    old_probs.append(old_prob)

                    prob_ratio = new_prob / old_prob
                    
                    clip = torch.clamp(prob_ratio, 1 - self.EPSILON, 1 + self.EPSILON)
                    clipped_prob_ratio = torch.min(prob_ratio * R_future[t], clip * R_future[t]).view(1)

                    policy_loss_l.append(-clipped_prob_ratio)

                new_probs = torch.Tensor(new_probs)
                old_probs = torch.Tensor(old_probs)

                entropy = - (
                    new_probs * torch.log(old_probs + 1.e-10) + \
                    (1.0 - new_probs) * torch.log(1.0 - old_probs + 1.e-10)
                )

                policy_loss.append(
                    torch.cat(policy_loss_l) + self.BETA * entropy
                )

            policy_loss = torch.cat(policy_loss).sum() / self.num_workers

            self.optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            self.optimizer.step()

            if self.logger:
                policy_loss = policy_loss.cpu().detach().item()
                self.logger.add_scalar(
                    'loss', policy_loss, self.time_step
                )

            del policy_loss
        
        self.EPSILON *= 0.999
