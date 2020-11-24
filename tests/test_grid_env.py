import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from ermas.envs.grid_env import GridEnv

env = GridEnv()
env.seed(0)
print(env.reset())
print(env.step([1, 0, 2]))
print(env.step([2, 0, 1]))
print(env.step([1, 0, 1]))
print(env.reset())
print(env.step([1, 0, 0]))
