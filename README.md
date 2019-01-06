# rlib

`rlib` is a small deep reinforcement learning library with implementations of popular deep RL algorithms. Each algorithm is highly modular and customizable, making this library a great choice for anyone who wants to test the performance of different algorithms in the same environment. `rlib` uses PyTorch as the library of choice for its initial version, but support for other libraries like TensorFlow are on the roadmap for the future.

## Installation

Coming soon.

## Usage

Using `rlib` is this simple:

```python
from rlib.algorithms.dqn.agent import DQNAgent
from rlib.environments.gym import GymEnvironment

env = GymEnvironment("CartPole-v0")
dqn = DQNAgent(env.observation_size, env.action_size, seed=seed)
env.set_algorithm(dqn)
env.train()
env.test()
```

## Testing

To run all tests:

```
python -m unittest
```

## Contributing

Coming soon.

## Credit

Some of the code in this repository is based on Udacity's [deep-reinforcement-learning](https://github.com/udacity/deep-reinforcement-learning) repository.

## License

`rlib` is released under the [MIT License](https://github.com/MarcioPorto/rlib/blob/master/LICENSE.md).