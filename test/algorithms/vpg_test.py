import gym

from rlib.algorithms.vpg.agent import VPGAgent
from rlib.environments.gym import GymEnvironment


def main(seed=0):
    env = GymEnvironment("CartPole-v0")
    vpg = VPGAgent(env.observation_size, env.action_size, seed=seed)
    env.set_algorithm(vpg)
    env.train()
    env.test()


if __name__ == "__main__":
    main()
