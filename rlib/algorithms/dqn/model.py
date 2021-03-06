import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """DQN Network Model."""

    def __init__(self, state_size: int, action_size: int, 
                 hidden_layer_input_size: int = 64,
                 hidden_layer_output_size: int = 64, 
                 seed: int = 0, softmax_output: bool = False):
        """Initializes parameters and build model.

        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_layer_input_size (int): Dimension of hidden layer input
            hidden_layer_output_size (int): Dimension of hidden layer output
            seed (int): Random seed
            softmax_output (bool): Decides use of softmax activation in the output layer
        
        Returns:
            An instance of QNetwork.
        """
        super(QNetwork, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.softmax_output = softmax_output

        self.fc1 = nn.Linear(state_size, hidden_layer_input_size)
        self.fc2 = nn.Linear(hidden_layer_input_size, hidden_layer_output_size)
        self.fc3 = nn.Linear(hidden_layer_output_size, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values.
        
        Args:
            state: A reading of the environment state.

        Returns:
            Probabily distribution over actions.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.softmax_output:
            x = F.softmax(x, dim=0)
        return x
