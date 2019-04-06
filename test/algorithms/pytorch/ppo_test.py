import gym

from rlib.algorithms.pytorch.ppo import PPOAgent
from rlib.environments.gym import GymEnvironment


def main(seed=0):
    e = gym.make('CartPole-v0')
    e.seed(seed)

    observation_size = 4
    action_size = 2

    ppo = PPOAgent(observation_size, action_size, seed=seed, new_hyperparameters={'num_updates': 2})
    env = GymEnvironment(e, ppo)
    env.train(num_episodes=500)
    env.test()


if __name__ == "__main__":
    main()
