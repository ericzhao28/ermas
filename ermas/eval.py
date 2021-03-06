from comet_ml import Experiment, api
from ermas.ppo import PPO, Memory
from ermas.envs.trading_env import TradingEnv, action_dim_p, action_dim_a1, action_dim_a2, state_dim, max_timesteps
from ermas.config import comet_ml_key
from ermas.args import get_args
from ermas.utils import crra_concavity
import random
import torch

betas = (0.9, 0.999)


def main():
    args = get_args()
    exp_id = args.exp_id + "_" + str(args.random_seed)

    # Initialize Comet.ml
    logger = Experiment(comet_ml_key, project_name="new_ermas")
    logger.set_name("test_" + exp_id + "_" + str(args.perturb))
    logger.log_parameters(vars(args))

    # Load training environment
    consumer_happiness = 10 - 10 * args.perturb
    env = TradingEnv(consumer_happiness)
    if args.random_seed:
        torch.manual_seed(args.random_seed)
        env.seed(args.random_seed)

    # Initialize PPO agents
    memory = Memory()
    p_ppo = PPO(state_dim, action_dim_p, args.n_latent_var_supplier, args.planner_lr, betas,
                args.gamma, args.K_epochs, args.eps_clip, 0)
    a1_ppo = PPO(state_dim, action_dim_a1, args.n_latent_var_shipping, args.planner_lr, betas,
                 args.gamma, args.K_epochs, args.eps_clip, 1)
    a2_ppo = PPO(state_dim, action_dim_a2, args.n_latent_var_consumer, args.lr, betas,
                 args.gamma, args.K_epochs, args.eps_clip, 2)

    # Load planner
    fname = "altsaves/p_ppo_save_{}.dict".format(exp_id)
    print(fname)
    p_ppo.policy.load_state_dict(torch.load(fname, map_location="cpu"))

    # Main training loop
    timestep = 0

    for i_episode in range(args.num_loops):
        history = {
            "p_running_reward": 0,
            "a1_running_reward": 0,
            "a2_running_reward": 0
        }

        for _ in range(args.main_episodes):
            state = env.reset()
            for t in range(max_timesteps):
                timestep += 1

                # Running policy_old:
                p_action = p_ppo.policy_old.act(state, memory)
                a1_action = a1_ppo.policy_old.act(state, memory)
                a2_action = a2_ppo.policy_old.act(state, memory)

                state, rewards, done = env.step(
                    [p_action, a1_action, a2_action])
                p_reward, a1_reward, a2_reward = rewards

                # Saving reward and is_terminal:
                memory.rewards[1].append(a1_reward)
                memory.rewards[2].append(a2_reward)
                memory.is_terminals.append(done)

                # Update if its time
                if timestep % args.update_timestep == 0:
                    a1_ppo.update(memory)
                    a2_ppo.update(memory)
                    memory.clear_memory()
                    timestep = 0

                # Log data
                for k, v in env.get_stats().items():
                    k = "Env " + k
                    if k not in history:
                        history[k] = 0
                    history[k] += v
                history["p_running_reward"] += p_reward
                history["a1_running_reward"] += a1_reward
                history["a2_running_reward"] += a2_reward

                # End episode
                if done:
                    break

        # Log main loop results
        print('Episode {} \t Planner reward: {}'.format(
            i_episode, history["p_running_reward"]))
        for k, v in history.items():
            v = float(v / (max_timesteps * args.main_episodes))
            logger.log_metric("Main_" + k, v, step=i_episode)

        ##### Save model files
        fname = "altsaves/a1_ppo_save_test_{}.dict".format(exp_id)
        torch.save(a1_ppo.policy.state_dict(), fname)
        logger.log_asset(fname, overwrite=True, step=i_episode)

        fname = "altsaves/a2_ppo_save_test_{}.dict".format(exp_id)
        torch.save(a2_ppo.policy.state_dict(), fname)
        logger.log_asset(fname, overwrite=True, step=i_episode)


if __name__ == "__main__":
    main()
