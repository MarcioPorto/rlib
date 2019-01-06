import gym

from rlib.algorithms.vpg import VPG
from rlib.environments.gym import GymEnvironment


def main(seed=0):
    env = GymEnvironment("CartPole-v0")
    vpg = VPG(env.observation_size, env.action_size, seed=seed)
    env.set_algorithm(vpg)
    env.train()
    env.test()


if __name__ == "__main__":
    main()
