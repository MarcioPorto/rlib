import gym

from rlib.algorithms.maddpg import MADDPG
from rlib.environments.gym import GymEnvironment


def main(seed=0):
    env = GymEnvironment("Pendulum-v0")
    maddpg = MADDPG(env.observation_size, env.action_size, num_agents=1, seed=seed)
    env.set_algorithm(maddpg)
    env.train()
    env.test()


if __name__ == "__main__":
    main()
