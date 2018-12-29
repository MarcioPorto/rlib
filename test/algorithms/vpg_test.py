import gym

from rlib.algorithms.vpg.agent import VPGAgent
from rlib.environments.gym import GymEnvironment


def main(seed=0):
    gym_env = GymEnvironment("CartPole-v0")
    vpg_agent = VPGAgent(gym_env.observation_size, gym_env.action_size, seed=seed)
    gym_env.set_agents([vpg_agent])
    gym_env.train()
    gym_env.test()


if __name__ == "__main__":
    main()
