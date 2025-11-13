# Autonomous-Car-Navigation-in-a-2D-Grid-World-using-PPO
This project implements a Proximal Policy Optimization (PPO) agent trained inside a fully custom GridWorld environment built using Gymnasium. The agent learns to navigate from a start position to a goal position while avoiding obstacles, optimizing its movements through trial-and-error interaction.

The environment uses a simple reward model:
+1.0 for reaching the goal
–0.01 for each step
Obstacles block movement

The PPO agent is built using PyTorch, and training includes:
Advantage computation
PPO clipped objective
Actor-Critic neural networks
Automatic model checkpointing
Reward curve visualization

After training, a separate script tests the model and demonstrates the agent taking the optimal path.
This project shows how reinforcement learning can solve navigation problems efficiently using a single GPU or even CPU.
