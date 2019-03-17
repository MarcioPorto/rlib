import gym

from rlib.algorithms.ddpg import DDPG
from rlib.environments.gym import GymEnvironment


def main(seed=0):
    e = gym.make('Pendulum-v0')
    e.seed(seed)

    observation_size = 3
    action_size = 1

    ddpg = DDPG(observation_size, action_size, num_agents=1, seed=seed)
    env = GymEnvironment(e, ddpg)
    env.train()
    env.test()


if __name__ == "__main__":
    main()
