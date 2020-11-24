from ermas.ppo import PPO, Memory
from ermas.ermas_ppo import ERMAS_PPO
from ermas.envs.trading_env import TradingEnv, action_dim_p, action_dim_a1, action_dim_a2, state_dim, max_timesteps

############## Hyperparameters ##############
n_latent_var = 64  # number of variables in hidden layer
update_timestep = 2000  # update policy every n timesteps
lr = 0.001
betas = (0.9, 0.999)
gamma = 0.995  # discount factor
K_epochs = 4  # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO
random_seed = 1
lambda_lr = 20
beta = 0.3
ermas_eps = -0.01
eval_episodes = 2
unilateral_episodes = 3
main_episodes = 5
num_loops = 10


def main():
    env = TradingEnv()

    history = {
        "a1": [],
        "a2": [],
        "delta1": [],
        "delta2": [],
        "rewardp": [],
        "reward1": [],
        "reward2": [],
    }

    #############################################

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    memory = Memory()
    memory1 = Memory()
    memory2 = Memory()
    dmemory = Memory()
    p_ppo = PPO(state_dim, action_dim_p, n_latent_var, lr, betas, gamma,
                K_epochs, eps_clip, 0)
    a1_ppo = ERMAS_PPO(state_dim, action_dim_a1, n_latent_var, lr, betas,
                       gamma, K_epochs, eps_clip, 1)
    a2_ppo = ERMAS_PPO(state_dim, action_dim_a2, n_latent_var, lr, betas,
                       gamma, K_epochs, eps_clip, 2)
    a1_mppo = PPO(state_dim, action_dim_a1, n_latent_var, lr, betas, gamma,
                  K_epochs, eps_clip, 1)
    a2_mppo = PPO(state_dim, action_dim_a2, n_latent_var, lr, betas, gamma,
                  K_epochs, eps_clip, 2)

    a1_lambda = 5
    a2_lambda = 5

    # logging variables
    avg_length = 0
    timestep = 0

    # training loop
    i_episode = 0
    for i_episode in range(num_loops):
        p_running_reward = 0.0
        a1_running_reward = 0.0
        a2_running_reward = 0.0
        # Main loop
        for _ in range(main_episodes):
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

                # update if its time
                if timestep % update_timestep == 0:
                    adv = p_ppo.update(memory)
                    a1_ppo.update(memory, a1_lambda, adv)
                    a2_ppo.update(memory, a2_lambda, adv)
                    memory.clear_memory()
                    timestep = 0

                p_running_reward += p_reward
                a1_running_reward += a1_reward
                a2_running_reward += a2_reward
                if done:
                    break

                for k, v in env.get_stats().items():
                    k = "Env " + k
                    if k not in history:
                        history[k] = []
                    history[k].append(v)

            avg_length += t

        avg_length = int(avg_length / (max_timesteps * main_episodes))
        p_running_reward = float(
            (p_running_reward / (max_timesteps * main_episodes)))
        a1_running_reward = float(
            (a1_running_reward / (max_timesteps * main_episodes)))
        a2_running_reward = float(
            (a2_running_reward / (max_timesteps * main_episodes)))
        print('Episode {} \t avg length: {} \t p reward: {}'.format(
            i_episode, avg_length, p_running_reward))
        history["rewardp"].append(p_running_reward)
        history["reward1"].append(a1_running_reward)
        history["reward2"].append(a2_running_reward)

        # Before eval
        a1_running_reward = 0
        a2_running_reward = 0
        dmemory.clear_memory()
        for _ in range(eval_episodes):
            state = env.reset()
            for t in range(max_timesteps):
                timestep += 1
                p_action = p_ppo.policy_old.act(state, dmemory)
                a1_action = a1_ppo.policy_old.act(state, dmemory)
                a2_action = a2_ppo.policy_old.act(state, dmemory)
                state, rewards, done = env.step(
                    [p_action, a1_action, a2_action])
                p_reward, a1_reward, a2_reward = rewards
                if done:
                    break
                a1_running_reward += a1_reward
                a2_running_reward += a2_reward

        # A1 loop
        a1_mppo.policy_old.load_state_dict(a1_ppo.policy.state_dict())
        a1_mppo.policy.load_state_dict(a1_ppo.policy.state_dict())
        for _ in range(unilateral_episodes):
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
                if timestep % update_timestep == 0:
                    a1_mppo.update(memory1)
                    memory1.clear_memory()
                    timestep = 0
                if done:
                    break

        # A2 loop
        a2_mppo.policy_old.load_state_dict(a2_ppo.policy.state_dict())
        a2_mppo.policy.load_state_dict(a2_ppo.policy.state_dict())
        for _ in range(unilateral_episodes):
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
                if timestep % update_timestep == 0:
                    a2_mppo.update(memory2)
                    memory2.clear_memory()
                    timestep = 0
                if done:
                    break

        # A1 eval
        a1_new_running_reward = 0
        dmemory.clear_memory()
        for _ in range(eval_episodes):
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
        for _ in range(eval_episodes):
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

        # Apply meta learning updates
        new_state_dict = a1_mppo.policy.state_dict()
        old_state_dict = a1_ppo.policy.state_dict()
        for key in new_state_dict:
            old_state_dict[key] = (old_state_dict[key] -
                                   beta * a1_lambda * new_state_dict[key]) / (
                                       1 + beta * a1_lambda)

        new_state_dict = a2_mppo.policy.state_dict()
        old_state_dict = a2_ppo.policy.state_dict()
        for key in new_state_dict:
            old_state_dict[key] = (old_state_dict[key] -
                                   beta * a2_lambda * new_state_dict[key]) / (
                                       1 + beta * a2_lambda)

        # Update lambda estimates
        a1_lambda = a1_lambda + (lambda_lr / max_timesteps) * (
            a1_new_running_reward - ermas_eps - a1_running_reward)
        a2_lambda = a2_lambda + (lambda_lr / max_timesteps) * (
            a2_new_running_reward - ermas_eps - a2_running_reward)

        a1_lambda = max(0, a1_lambda)
        a2_lambda = max(0, a2_lambda)

        history["delta1"].append(a1_new_running_reward - a1_running_reward)
        history["delta2"].append(a2_new_running_reward - a2_running_reward)
        history["a1"].append(a1_lambda)
        history["a2"].append(a2_lambda)
        for k, v in env.get_stats().items():
            k = "Env " + k
            if k not in history:
                history[k] = []
            history[k].append(v)

        # Log
        a1_running_reward = float(
            (a1_running_reward / (max_timesteps * eval_episodes)))
        a1_new_running_reward = float(
            (a1_new_running_reward / (max_timesteps * eval_episodes)))
        print('Episode {} \t a1 reward: {}, {}'.format(i_episode,
                                                       a1_running_reward,
                                                       a1_new_running_reward))

        a2_running_reward = float(
            (a2_running_reward / (max_timesteps * eval_episodes)))
        a2_new_running_reward = float(
            (a2_new_running_reward / (max_timesteps * eval_episodes)))
        print('Episode {} \t a2 reward: {}, {}'.format(i_episode,
                                                       a2_running_reward,
                                                       a2_new_running_reward))

        print("Episode {} \t Lambdas".format(i_episode), a1_lambda, a2_lambda)


if __name__ == "__main__":
    main()
