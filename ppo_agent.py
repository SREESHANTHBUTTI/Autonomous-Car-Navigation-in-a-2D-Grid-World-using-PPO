import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import gym

# Custom environment import
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
        states = torch.stack(memory['states'])
        actions = torch.stack(memory['actions']).detach()
        rewards = memory['rewards']
        old_log_probs = torch.stack(memory['log_probs']).detach()

        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for _ in range(self.epochs):
            dist, values = self.policy(states)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            values = values.squeeze()

            # Advantage = Return - Value
            advantage = returns - values.detach()

            # Ratio (pi_theta / pi_theta_old)
            ratio = torch.exp(new_log_probs - old_log_probs)

            # PPO Loss
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            loss = -torch.min(surr1, surr2).mean() + 0.5 * (returns - values).pow(2).mean() - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


# ======================
# Training Function
# ======================
def train_ppo(num_episodes=2000):
    env = GridWorldEnv(grid_size=5)
    state_dim = 2
    action_dim = env.action_space.n
    ppo = PPO(state_dim, action_dim)

    all_rewards = []
    best_avg_reward = -float("inf")

    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        memory = {'states': [], 'actions': [], 'rewards': [], 'log_probs': []}
        total_reward = 0

        done = False
        while not done:
            state_t = torch.FloatTensor(state)
            dist, value = ppo.policy(state_t)
            action = dist.sample()

            nnext_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated


            memory['states'].append(state_t)
            memory['actions'].append(action)
            memory['rewards'].append(reward)
            memory['log_probs'].append(dist.log_prob(action))

            state = nnext_state
            total_reward += reward

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

    # Plot reward trend
    plt.plot(all_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO Training Reward Curve")
    plt.grid(True)
    plt.show()

    return ppo.policy


if __name__ == "__main__":
    train_ppo(num_episodes=2000)

