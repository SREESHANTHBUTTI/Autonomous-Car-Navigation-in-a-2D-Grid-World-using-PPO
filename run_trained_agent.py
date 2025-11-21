import torch
import numpy as np
from grid_env import GridWorldEnv
from ppo_agent import ActorCritic

# ------------------------------
# Load Environment
# ------------------------------
env = GridWorldEnv(grid_size=5)

state_dim = 2
action_dim = env.action_space.n

# ------------------------------
# Load Model
# ------------------------------
model = ActorCritic(state_dim, action_dim)
model.load_state_dict(torch.load("ppo_trained_model.pth"))
model.eval()

# ------------------------------
# Reset Environment
# ------------------------------
state, info = env.reset()
state = np.array(state, dtype=np.float32)

done = False
total_reward = 0
step = 0

print("\n Starting Trained Agent Run...\n")
env.render()

# ------------------------------
# Run the trained PPO agent
# ------------------------------
while not done and step < 50:

    state_tensor = torch.tensor(state, dtype=torch.float32)

    with torch.no_grad():
        dist, value = model(state_tensor)
        action = dist.sample().item()          # <<< IMPORTANT FIX

    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    next_state = np.array(next_state, dtype=np.float32)

    state = next_state
    total_reward += reward
    step += 1

    print(f"Step {step}: Action={action}, Reward={reward}")
    env.render()

print(f"\n Episode finished in {step} steps | Total Reward: {total_reward}")
env.close()
