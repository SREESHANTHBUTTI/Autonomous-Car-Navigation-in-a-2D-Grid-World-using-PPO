import importlib
import sys
import os

# Make sure Python can see the current folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Reload module in case of caching issues
import grid_env
importlib.reload(grid_env)
from grid_env import GridWorldEnv

# Test environment
env = GridWorldEnv(grid_size=5)
print(" GridWorldEnv initialized successfully!")

obs = env.reset()
print("Starting state:", obs)

for step in range(5):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(f"Step {step+1}: Action={action}, State={obs}, Reward={reward}, Done={done}")
    env.render()
    if done:
        print(" Goal reached!")
        break
