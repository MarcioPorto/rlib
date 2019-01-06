import gym

from rlib.algorithms.dqn.agent import DQNAgent
from rlib.environments.gym import GymEnvironment


def main(seed=0):
    env = GymEnvironment("CartPole-v0")
    dqn = DQNAgent(env.observation_size, env.action_size, seed=seed)
    env.set_algorithm(dqn)
    env.train()
    env.test()


if __name__ == "__main__":
    main()
