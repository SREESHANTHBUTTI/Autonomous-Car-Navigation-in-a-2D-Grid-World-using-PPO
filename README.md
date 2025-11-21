# Autonomous Car Navigation in a 2D Grid World using PPO

This project implements an Autonomous Car navigation system in a custom 2D Grid-World environment using Proximal Policy Optimization (PPO) Reinforcement Learning. The goal of the agent is to navigate from a start position to a goal position while avoiding obstacles.

------------------------------------------------------------

## Features
- Custom GridWorld environment using Gymnasium
- PPO reinforcement learning implementation with Actor-Critic network
- Saves best model during training
- Training rewards plotted with moving average
- Console-based grid visualization of trained agent navigation

------------------------------------------------------------

## Project Structure

Autonomous-Car-Navigation-in-a-2D-Grid-World-using-PPO/
│
├── grid_env.py                (Custom environment)
├── ppo_agent.py               (PPO training script)
├── run_trained_agent.py       (Run trained agent)
├── ppo_trained_model.pth      (Saved model generated after training)
└── README.md

------------------------------------------------------------

## Training Configuration

Algorithm: Proximal Policy Optimization (PPO)  
Model: Actor-Critic Neural Network  
Learning Rate: 3e-4  
Discount Factor (gamma): 0.99  
Reward: +1 on reaching goal, -0.01 per step  

A moving average (50 episodes) reward curve is plotted to show learning stability.

------------------------------------------------------------

## How to Run

### Step 1: Install Dependencies
Command:
pip install torch gymnasium numpy matplotlib

### Step 2: Train PPO Agent
Command:
python ppo_agent.py

This will:
- Train the model
- Display reward training graph
- Save best model as "ppo_trained_model.pth"

### Step 3: Run Trained Agent
Command:
python run_trained_agent.py

You will see a grid printed like:
. . X . G
A . X . .
. X . . .
. . X . .
X . . . .

Legend:
A = Agent  
G = Goal  
X = Obstacle  
. = Free path  

The run continues until the agent reaches the goal or hits step limits.

------------------------------------------------------------

## Customization

You can change grid size, start, goal and obstacles in grid_env.py.

Example:
env = GridWorldEnv(
    grid_size=7,
    start_pos=(6,0),
    goal_pos=(0,6),
    obstacles={(1,2), (3,3), (4,1), (2,5)}
)

------------------------------------------------------------

## Training Output Graph

The script generates a reward curve for evaluating convergence:
- Episode reward plot
- 50-episode moving average

------------------------------------------------------------

## Future Enhancements
- GUI visualization (e.g., using Pygame)
- Larger grid with complex paths
- Dynamic obstacle movement
- Better reward shaping

------------------------------------------------------------

## Author
Sreeshanth Butti  
GitHub Repository:  
github.com/SREESHANTHBUTTI/Autonomous-Car-Navigation-in-a-2D-Grid-World-using-PPO

------------------------------------------------------------
