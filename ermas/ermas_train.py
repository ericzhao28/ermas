from comet_ml import Experiment, api
from ermas.ppo import PPO, Memory
from ermas.ermas_ppo import ERMAS_PPO
from ermas.envs.trading_env import TradingEnv, action_dim_p, action_dim_a1, action_dim_a2, state_dim, max_timesteps
from ermas.config import comet_ml_key
from ermas.args import get_args
import torch

betas = (0.9, 0.999)


def main():
    args = get_args()
    exp_id = args.exp_id + "_" + str(args.random_seed)

    # Initialize Comet.ml
    logger = Experiment(comet_ml_key, project_name="ermas")
    logger.set_name(exp_id + "_" + str(args.random_seed))
    logger.log_parameters(vars(args))

    # Load training environment
    env = TradingEnv()
    if args.random_seed:
        torch.manual_seed(args.random_seed)
        env.seed(args.random_seed)

    # Initialize PPO agents
    memory = Memory()
    memory1 = Memory()
    memory2 = Memory()
    dmemory = Memory()
    p_ppo = PPO(state_dim, action_dim_p, args.n_latent_var, args.lr, betas,
                args.gamma, args.K_epochs, args.eps_clip, 0)
    a1_ppo = ERMAS_PPO(state_dim, action_dim_a1, args.n_latent_var, args.lr,
                       betas, args.gamma, args.K_epochs, args.eps_clip, 1)
    a2_ppo = ERMAS_PPO(state_dim, action_dim_a2, args.n_latent_var, args.lr,
                       betas, args.gamma, args.K_epochs, args.eps_clip, 2)
    a1_mppo = PPO(state_dim, action_dim_a1, args.n_latent_var, args.lr, betas,
                  args.gamma, args.K_epochs, args.eps_clip, 1)
    a2_mppo = PPO(state_dim, action_dim_a2, args.n_latent_var, args.lr, betas,
                  args.gamma, args.K_epochs, args.eps_clip, 2)

    # Initialize lagrange multipliers
    a1_lambda = args.initial_lambda
    a2_lambda = args.initial_lambda

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
                memory.rewards[0].append(p_reward)
                memory.rewards[1].append(a1_reward)
                memory.rewards[2].append(a2_reward)
                memory.is_terminals.append(done)

                # Update if its time
                if timestep % args.update_timestep == 0:
                    adv = p_ppo.update(memory)
                    a1_ppo.update(memory, a1_lambda, adv)
                    a2_ppo.update(memory, a2_lambda, adv)
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

        # Early evaluation eval
        a1_running_reward = 0
        a2_running_reward = 0
        dmemory.clear_memory()
        for _ in range(args.eval_episodes):
            state = env.reset()
            for t in range(max_timesteps):
                p_action = p_ppo.policy_old.act(state, dmemory)
                a1_action = a1_ppo.policy_old.act(state, dmemory)
                a2_action = a2_ppo.policy_old.act(state, dmemory)
                state, rewards, done = env.step(
                    [p_action, a1_action, a2_action])
                p_reward, a1_reward, a2_reward = rewards
                a1_running_reward += a1_reward
                a2_running_reward += a2_reward
                if done:
                    break

        # A1 loop
        a1_mppo.policy_old.load_state_dict(a1_ppo.policy.state_dict())
        a1_mppo.policy.load_state_dict(a1_ppo.policy.state_dict())
        for _ in range(args.unilateral_episodes):
            state = env.reset()
            for t in range(max_timesteps):
                timestep += 1
                p_action = p_ppo.policy_old.act(state, memory1)
                a1_action = a1_mppo.policy_old.act(state, memory1)
                a2_action = a2_ppo.policy_old.act(state, memory1)
                state, rewards, done = env.step(
                    [p_action, a1_action, a2_action])
                p_reward, a1_reward, a2_reward = rewards
                memory1.rewards[1].append(a1_reward)
                memory1.is_terminals.append(done)
                if timestep % args.update_timestep == 0:
                    a1_mppo.update(memory1)
                    memory1.clear_memory()
                    timestep = 0
                if done:
                    break

        # A2 loop
        a2_mppo.policy_old.load_state_dict(a2_ppo.policy.state_dict())
        a2_mppo.policy.load_state_dict(a2_ppo.policy.state_dict())
        for _ in range(args.unilateral_episodes):
            state = env.reset()
            for t in range(max_timesteps):
                timestep += 1
                p_action = p_ppo.policy_old.act(state, memory2)
                a1_action = a1_ppo.policy_old.act(state, memory2)
                a2_action = a2_mppo.policy_old.act(state, memory2)
                state, rewards, done = env.step(
                    [p_action, a1_action, a2_action])
                p_reward, a1_reward, a2_reward = rewards
                memory2.rewards[2].append(a2_reward)
                memory2.is_terminals.append(done)
                if timestep % args.update_timestep == 0:
                    a2_mppo.update(memory2)
                    memory2.clear_memory()
                    timestep = 0
                if done:
                    break

        # A1 eval
        a1_new_running_reward = 0
        dmemory.clear_memory()
        for _ in range(args.eval_episodes):
            state = env.reset()
            for t in range(max_timesteps):
                timestep += 1
                p_action = p_ppo.policy_old.act(state, dmemory)
                a1_action = a1_mppo.policy_old.act(state, dmemory)
                a2_action = a2_ppo.policy_old.act(state, dmemory)
                state, rewards, done = env.step(
                    [p_action, a1_action, a2_action])
                p_reward, a1_reward, a2_reward = rewards
                if done:
                    break
                a1_new_running_reward += a1_reward

        # A2 eval
        a2_new_running_reward = 0
        dmemory.clear_memory()
        for _ in range(args.eval_episodes):
            state = env.reset()
            for t in range(max_timesteps):
                timestep += 1
                p_action = p_ppo.policy_old.act(state, dmemory)
                a1_action = a1_ppo.policy_old.act(state, dmemory)
                a2_action = a2_mppo.policy_old.act(state, dmemory)
                state, rewards, done = env.step(
                    [p_action, a1_action, a2_action])
                p_reward, a1_reward, a2_reward = rewards
                if done:
                    break
                a2_new_running_reward += a2_reward

        ###### Apply meta learning updates
        new_state_dict = a1_mppo.policy.state_dict()
        old_state_dict = a1_ppo.policy.state_dict()
        for key in new_state_dict:
            old_state_dict[key] = (old_state_dict[key] -
                                   args.beta * a1_lambda * new_state_dict[key]
                                   ) / (1 + args.beta * a1_lambda)
        a1_ppo.policy.load_state_dict(old_state_dict)
        a1_ppo.policy_old.load_state_dict(old_state_dict)

        new_state_dict = a2_mppo.policy.state_dict()
        old_state_dict = a2_ppo.policy.state_dict()
        for key in new_state_dict:
            old_state_dict[key] = (old_state_dict[key] -
                                   args.beta * a2_lambda * new_state_dict[key]
                                   ) / (1 + args.beta * a2_lambda)
        a2_ppo.policy.load_state_dict(old_state_dict)
        a2_ppo.policy_old.load_state_dict(old_state_dict)

        ##### Update lambda estimates
        a1_lambda = a1_lambda + (args.lambda_lr / max_timesteps) * (
            a1_new_running_reward - args.ermas_eps - a1_running_reward)
        a2_lambda = a2_lambda + (args.lambda_lr / max_timesteps) * (
            a2_new_running_reward - args.ermas_eps - a2_running_reward)

        a1_lambda = max(0, a1_lambda)
        a2_lambda = max(0, a2_lambda)

        ##### Log ERMAS stats
        a1_running_reward = float(
            (a1_running_reward / (max_timesteps * args.eval_episodes)))
        a1_new_running_reward = float(
            (a1_new_running_reward / (max_timesteps * args.eval_episodes)))
        print('Episode {} \t a1 reward: {}, {}'.format(i_episode,
                                                       a1_running_reward,
                                                       a1_new_running_reward))
        a2_running_reward = float(
            (a2_running_reward / (max_timesteps * args.eval_episodes)))
        a2_new_running_reward = float(
            (a2_new_running_reward / (max_timesteps * args.eval_episodes)))
        print('Episode {} \t a2 reward: {}, {}'.format(i_episode,
                                                       a2_running_reward,
                                                       a2_new_running_reward))
        print("Episode {} \t Lambdas".format(i_episode), a1_lambda, a2_lambda)

        logger.log_metric("Lambda 1", a1_lambda, step=i_episode)
        logger.log_metric("Lambda 2", a2_lambda, step=i_episode)
        logger.log_metric("Delta 1",
                          a1_new_running_reward - a1_running_reward,
                          step=i_episode)
        logger.log_metric("Delta 2",
                          a2_new_running_reward - a2_running_reward,
                          step=i_episode)

        ##### Save model files
        fname = "saves/p_ppo_save_{}.dict".format(exp_id)
        torch.save(p_ppo.policy.state_dict(), fname)
        logger.log_asset(fname, overwrite=True, step=i_episode)

        fname = "saves/a1_ppo_save_{}.dict".format(exp_id)
        torch.save(a1_ppo.policy.state_dict(), fname)
        logger.log_asset(fname, overwrite=True, step=i_episode)

        fname = "saves/a2_ppo_save_{}.dict".format(exp_id)
        torch.save(a2_ppo.policy.state_dict(), fname)
        logger.log_asset(fname, overwrite=True, step=i_episode)


if __name__ == "__main__":
    main()
