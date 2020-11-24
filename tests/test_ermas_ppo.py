import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from ermas.envs.trading_env import state_dim, action_dim_p
from ermas.ermas_ppo import ERMAS_PPO

n_latent_var = 32
lr = 0.01
betas = (0.9, 0.999)
gamma = 0.995
K_epochs = 4
eps_clip = 0.2


p_ppo = ERMAS_PPO(state_dim, action_dim_p, n_latent_var, lr, betas, gamma, K_epochs,
            eps_clip, 0)
