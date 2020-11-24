import numpy as np
import random

state_dim = 2
action_dim = 3
action_dim_p = action_dim
action_dim_a1 = action_dim
action_dim_a2 = action_dim
solved_reward = 230  # stop training if avg_reward > solved_reward
max_timesteps = 1000  # max timesteps in one episode


class GridEnv():
    def __init__(self):
        self.reset()

    def reset(self):
        self.counter = 0
        state = np.random.random((state_dim, ))
        return state

    def step(self, actions):
        state = np.random.random((state_dim, ))
        rewards = self.payoffs[tuple(actions)]
        done = self.counter > 50
        self.counter += 1
        return state, rewards, done

    def seed(self, seed_num):
        np.random.seed(seed_num)
        random.seed(seed_num)
        self.payoffs = np.random.random(
            (action_dim, action_dim, action_dim, 3))
