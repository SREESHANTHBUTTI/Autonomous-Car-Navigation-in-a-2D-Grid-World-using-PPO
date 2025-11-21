# ppo_agent.py  (fixed)
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt

# Use gymnasium (maintained); if you still have 'gym' installed only, install gymnasium or
# change back to gym but you'll get the deprecation message.
import gymnasium as gym

# Custom environment import (assumes grid_env.py implements a Gymnasium env)
from grid_env import GridWorldEnv
print(">>> grid_env.py loaded successfully <<<")

# ======================
# Actor-Critic Network
# ======================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        """
        state: Tensor of shape (batch, state_dim) or (state_dim,)
        returns: Categorical distr (batch aware) and values (batch,1)
        """
        probs = self.actor(state)
        dist = Categorical(probs)
        value = self.critic(state)
        return dist, value


# ======================
# PPO Agent
# ======================
class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, epochs=4):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs

        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def update(self, memory):
        if len(memory['states']) == 0:
            return

        # Stack states (T, state_dim)
        states = torch.stack(memory['states']).float()
        # Actions should be long for log_prob
        actions = torch.stack(memory['actions']).long().detach()
        rewards = memory['rewards']
        old_log_probs = torch.stack(memory['log_probs']).detach()

        # Compute discounted returns
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        # Normalize returns (safe)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for _ in range(self.epochs):
            dist, values = self.policy(states)
            # new_log_probs: shape (T,)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            values = values.squeeze()

            # Advantage = Return - Value
            advantage = returns - values.detach()

            # Ratio (pi_theta / pi_theta_old)
            ratio = torch.exp(new_log_probs - old_log_probs)

            # PPO clipped objective
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            loss = -torch.min(surr1, surr2).mean() + 0.5 * (returns - values).pow(2).mean() - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


# ======================
# Utilities
# ======================
def obs_to_tensor(obs):
    """Convert observation to 1D torch.FloatTensor."""
    # If observation is (obs, info) accidentally passed, handle it upstream.
    if isinstance(obs, dict):
        # flatten dict values (common in some envs)
        # pick numeric arrays and concatenate
        parts = []
        for v in obs.values():
            arr = np.asarray(v)
            parts.append(arr.ravel())
        arr = np.concatenate(parts).astype(np.float32)
    else:
        arr = np.asarray(obs).astype(np.float32).ravel()
    return torch.from_numpy(arr)


# ======================
# Training Function
# ======================
def train_ppo(num_episodes=2000, grid_size=5):
    env = GridWorldEnv(grid_size=grid_size)

    # Derive state_dim from observation_space (handle Box)
    obs_space = env.observation_space
    if hasattr(obs_space, "shape") and obs_space.shape is not None:
        state_dim = int(np.prod(obs_space.shape))
    else:
        # fallback to 2 if not available, but better to inspect your env
        state_dim = 2

    action_dim = env.action_space.n
    ppo = PPO(state_dim, action_dim)

    all_rewards = []
    best_avg_reward = -float("inf")

    for episode in range(1, num_episodes + 1):
        # Gymnasium reset returns (obs, info)
        obs, info = env.reset()
        state = obs
        memory = {'states': [], 'actions': [], 'rewards': [], 'log_probs': []}
        total_reward = 0.0

        terminated = False
        truncated = False
        while not (terminated or truncated):
            state_t = obs_to_tensor(state)                     # 1D tensor
            # If single sample, actor expects shape (state_dim,) which works with Linear
            dist, value = ppo.policy(state_t)
            action = dist.sample()                             # 0-dim tensor (long)
            action_int = int(action.item())                    # pass python int to env

            next_obs, reward, terminated, truncated, info = env.step(action_int)

            # store tensors (detach to avoid graph retention)
            memory['states'].append(state_t.detach())
            memory['actions'].append(action.detach())
            memory['rewards'].append(float(reward))
            memory['log_probs'].append(dist.log_prob(action).detach())

            state = next_obs
            obs = next_obs
            total_reward += float(reward)

        all_rewards.append(total_reward)
        ppo.update(memory)

        if episode % 50 == 0:
            avg_reward = np.mean(all_rewards[-50:])
            print(f"Episode {episode}/{num_episodes} | Avg Reward (last 50): {avg_reward:.2f}")
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(ppo.policy.state_dict(), "ppo_trained_model.pth")
                print(f"Saved best model (Avg Reward = {avg_reward:.2f})")

    # Save final model
    torch.save(ppo.policy.state_dict(), "ppo_trained_model_final.pth")
    print(" Training completed! Model saved as 'ppo_trained_model_final.pth'")

    # ======================
    # Plot learning curve
    # ======================
    if len(all_rewards) > 0:
        episodes = np.arange(1, len(all_rewards) + 1)

        plt.figure(figsize=(10, 5))
        plt.plot(episodes, all_rewards, alpha=0.3, label="Reward per Episode")

        window = 50
        if len(all_rewards) >= window:
            kernel = np.ones(window) / window
            smooth = np.convolve(all_rewards, kernel, mode='valid')
            smooth_episodes = np.arange(window, len(all_rewards) + 1)
            plt.plot(smooth_episodes, smooth, linewidth=2, label=f"{window}-Episode Moving Average")
        else:
            plt.plot(episodes, all_rewards, linewidth=2, label="Reward per Episode")

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("PPO Training Reward vs Episode")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return ppo.policy


if __name__ == "__main__":
    train_ppo(num_episodes=2000, grid_size=5)
