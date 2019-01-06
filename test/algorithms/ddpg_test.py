import gym

from rlib.algorithms.ddpg.agent import DDPGAgent
from rlib.environments.gym import GymEnvironment


def main(seed=0):
    gym_env = GymEnvironment("Pendulum-v0")
    ddpg_agent = DDPGAgent(gym_env.observation_size, gym_env.action_size, num_agents=1, seed=seed)
    gym_env.set_agents([ddpg_agent])
    gym_env.train()
    # gym_env.test()


if __name__ == "__main__":
    main()
