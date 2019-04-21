import gym

from rlib.algorithms.pytorch.ppo import PPOAgent
from rlib.environments.gym import ParallelGymEnvironment


def main(seed=0):
    # e = gym.make('CartPole-v0') # discrete actions
    e = gym.make('Pendulum-v0') # continuous actions
    # e = gym.make('LunarLander-v2')

    e.seed(seed)

    # TODO: Figure out how to get this from the environment
    # observation_size = 4
    # action_size = 2
    observation_size = 3
    action_size = 1
    # observation_size = 8
    # action_size = 4

    ppo = PPOAgent(observation_size, action_size, seed=seed, new_hyperparameters={
        'num_updates': 2,
        'epsilon': 0.2
    })
    env = ParallelGymEnvironment(e, ppo)
    env.train(num_episodes=500, num_workers=4)
    # env.test()


if __name__ == "__main__":
    main()
