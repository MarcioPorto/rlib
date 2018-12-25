import gym

from rlib.algorithms.dqn.agent import DQNAgent
from rlib.environments.gym import GymEnvironment


def main(seed=0):
    gym_env = GymEnvironment("Pendulum-v0")
    dqn_agent = DQNAgent(gym_env.observation_size, gym_env.action_size, seed=seed)
    gym_env.set_agents([dqn_agent])
    gym_env.train()
    gym_env.test()


if __name__ == "__main__":
    main()
