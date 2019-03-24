# rlib

[![Build Status](https://travis-ci.org/MarcioPorto/rlib.svg?branch=master)](https://travis-ci.org/MarcioPorto/rlib)

`rlib` is a small deep reinforcement learning library with implementations of popular deep RL algorithms. Each algorithm is highly modular and customizable, making this library a great choice for anyone who wants to test the performance of different algorithms in the same environment. `rlib` uses PyTorch as the library of choice for its initial version, but support for TensorFlow is on the roadmap.

## Installation

```bash
pip install rlib
```

## Usage

Using `rlib` is this simple:

```python
from rlib.algorithms.pytorch.dqn import DQNAgent
from rlib.environments.gym import GymEnvironment


e = gym.make('CartPole-v0')

observation_size = 4
action_size = 2

dqn = DQNAgent(observation_size, action_size)
env = GymEnvironment(e, dqn)
env.train()
env.test()
```

## Supported Algorithms

|        | PyTorch  | TensorFlow |
|--------|----------|------------|
| A2C    |          |            |
| A3C    |          |            |
| DPPG   | &#10004; |            |
| DQN    | &#10004; |            |
| MADDPG |          |            |
| PPO    |          |            |
| SAC    |          |            |
| TD3    |          |            |
| TRPO   |          |            |
| VPG    | &#10004; |            |

## Advanced

### TensorBoard and GIFRecorder

1. Initialize `Logger` and/or `GIFRecorder` objects. 

```python
os.makedirs('your/log/dir', exist_ok=True)

logger = Logger(output_dir)
gifs_recorder = GIFRecorder(output_dir, duration=3.0)
```

2. Initialize a new environment using these objects.

```python
env = GymEnvironment(e, dqn, logger=logger, gifs_recorder=gifs_recorder)
```

3. To check Tensorboard logs, run:

```bash
tensorboard --logdir=your/log/dir
```

### Custom models

1. Define your own custom model.

```python
class NeuralNet(torch.nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(4, 8) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
```

2. Check the documentation for the algorithm you are using for the appropriate argument name. For DQN:

```python
dqn = DQNAgent(
    observation_size, action_size,
    qnetwork_local=NeuralNet(),
    qnetwork_target=NeuralNet(),
)
```

### Saving model weights

1. Set the `model_output_dir` argument when creating a new instance of an algorithm to the directory where you want your model to be saved.

## Testing

To run all tests:

```bash
python -m unittest discover test/
```

## Contributing

### Installation

Feel free to open issues with any bugs found or any feature requests. Pull requests are always welcome for new functionality.

```bash
virtualenv -p python3 venv
cd rlib/
source venv/bin/activate
pip install -r requirements.txt
```

To make sure your installation worked, run one of the algorithms in the test folder:

```
python test/algorithms/dqn_test.py
```

## License

`rlib` is released under the [MIT License](https://github.com/MarcioPorto/rlib/blob/master/LICENSE.md).

