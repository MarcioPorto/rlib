import gym

from rlib.algorithms.ddpg import DDPG
from rlib.environments.gym import GymEnvironment


def main(seed=0):
    env = GymEnvironment("Pendulum-v0")
    ddpg = DDPG(env.observation_size, env.action_size, num_agents=1, seed=seed)
    env.set_algorithm(ddpg)
    env.train()
    env.test()


if __name__ == "__main__":
    main()
