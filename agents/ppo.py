import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, q_max):
        super().__init__()
        self.q_max = q_max

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim)
        )

        self.log_std = nn.Parameter(torch.zeros(action_dim), requires_grad=False)
        self.min_log_std = -3.0
        self.max_log_std = 0.0

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def act(self, obs):
        mean = self.actor(obs)
        std = self.log_std.exp()
        dist = Normal(mean, std)

        z = dist.rsample()
        action = torch.tanh(z)
        action = (action + 1.0) / 2.0 * self.q_max

        log_prob = dist.log_prob(z).sum(-1)
        log_prob -= torch.log(1 - torch.tanh(z) ** 2 + 1e-6).sum(-1)

        value = self.critic(obs).squeeze(-1)
        return action, log_prob, value

    def evaluate(self, obs, action):
        mean = self.actor(obs)
        std = self.log_std.exp()
        dist = Normal(mean, std)

        z = 2.0 * action / self.q_max - 1.0
        z = torch.clamp(z, -0.999, 0.999)
        z = 0.5 * torch.log((1 + z) / (1 - z))

        log_prob = dist.log_prob(z).sum(-1)
        log_prob -= torch.log(1 - torch.tanh(z) ** 2 + 1e-6).sum(-1)

        entropy = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)

        return log_prob, entropy, value

    def set_log_std(self, progress):
        log_std = self.max_log_std + progress * (self.min_log_std - self.max_log_std)
        self.log_std.data.fill_(log_std)


class PPOAgent:
    def __init__(
        self,
        n_actions: int,
        q_max: float,
        obs_dim: int,
        lr=3e-4,
        gamma=0.99,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01
    ):
        self.n_actions = n_actions
        self.q_max = q_max
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

        self.policy = ActorCritic(obs_dim, n_actions, q_max)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []

    def select_action(self, state: np.ndarray) -> np.ndarray:
        state_t = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action, log_prob, value = self.policy.act(state_t)

        self.states.append(state_t.squeeze(0))
        self.actions.append(action.squeeze(0))
        self.log_probs.append(log_prob.squeeze(0))
        self.values.append(value.squeeze(0))

        return action.cpu().numpy().flatten()

    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        self.rewards.append(reward)

        if not done:
            return

        returns = []
        G = 0.0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs).detach()
        values = torch.stack(self.values).detach()

        advantages = returns - values

        log_probs, entropy, new_values = self.policy.evaluate(states, actions)

        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = (returns - new_values).pow(2).mean()
        entropy_loss = -entropy.mean()

        loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.reset()
