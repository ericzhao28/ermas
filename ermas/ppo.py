import torch
import torch.nn as nn
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self):
        self.actions = [[], [], []]
        self.logprobs = [[], [], []]
        self.rewards = [[], [], []]
        self.states = []
        self.is_terminals = []

    def clear_memory(self):
        for c in self.actions:
            del c[:]
        for c in self.rewards:
            del c[:]
        for c in self.logprobs:
            del c[:]
        del self.states[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, index):
        super(ActorCritic, self).__init__()

        self.index = index

        # actor
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var), nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var), nn.Tanh(),
            nn.Linear(n_latent_var, action_dim), nn.Softmax(dim=-1))

        # critic
        self.value_layer = nn.Sequential(nn.Linear(state_dim, n_latent_var),
                                         nn.Tanh(),
                                         nn.Linear(n_latent_var, n_latent_var),
                                         nn.Tanh(), nn.Linear(n_latent_var, 1))

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        if self.index == 0:
            memory.states.append(state)
        memory.actions[self.index].append(action)
        memory.logprobs[self.index].append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma,
                 K_epochs, eps_clip, index):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.index = index

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var,
                                  index).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=lr,
                                          betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var,
                                      index).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards[self.index]),
                                       reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(
            memory.actions[self.index]).to(device).detach()
        old_logprobs = torch.stack(
            memory.logprobs[self.index]).to(device).detach()

        first_advantage = None

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values:
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            if first_advantage is None:
                first_advantage = advantages
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip,
                                1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(
                state_values, rewards) - 0.05 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        return first_advantage
