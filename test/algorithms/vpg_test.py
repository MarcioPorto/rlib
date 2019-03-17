import gym

from rlib.algorithms.vpg import VPG
from rlib.environments.gym import GymEnvironment


def main(seed=0):
    e = gym.make('CartPole-v0')
    e.seed(seed)

    observation_size = 4
    action_size = 2

    vpg = VPG(observation_size, action_size, seed=seed)
    env = GymEnvironment(e, vpg)
    env.train()
    env.test()


if __name__ == "__main__":
    main()
