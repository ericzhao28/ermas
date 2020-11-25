"""Argument parsing utilites"""

import argparse
import torch


def get_args():
    """Argparse specs"""

    use_cuda = torch.cuda.is_available()
    device = 0 if use_cuda else "cpu"
    print("Default device", device)

    parser = argparse.ArgumentParser()

    # Add main values
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--exp_id', type=str, default="default")
    parser.add_argument('--resume_exp_id', type=str, default="")
    parser.add_argument('--perturb', type=float, default=0)

    # PPO Params
    parser.add_argument('--n_latent_var_supplier', type=int, default=32)
    parser.add_argument('--n_latent_var_consumer', type=int, default=16)
    parser.add_argument('--n_latent_var_shipping', type=int, default=8)
    parser.add_argument('--update_timestep', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--K_epochs', type=int, default=3)
    parser.add_argument('--eps_clip', type=float, default=0.2)
    parser.add_argument('--main_episodes', type=int, default=4)
    parser.add_argument('--num_loops', type=int, default=100)

    # Baselines
    parser.add_argument('--crra_sigma', type=float, default=-1)
    parser.add_argument('--da_random', type=float, default=0)
    parser.add_argument('--worst_action_prob', type=float, default=0)

    # ERMAS Params
    parser.add_argument('--lambda_lr', type=float, default=0.01)
    parser.add_argument('--initial_lambda', type=float, default=2)
    parser.add_argument('--beta', type=float, default=0.3)
    parser.add_argument('--ermas_eps', type=float, default=10.0)
    parser.add_argument('--eval_episodes', type=int, default=1)
    parser.add_argument('--unilateral_episodes', type=int, default=2)

    # Parse args
    args = parser.parse_args()

    return args
