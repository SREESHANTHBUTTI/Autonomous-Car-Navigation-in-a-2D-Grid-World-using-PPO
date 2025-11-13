import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size=5, start_pos=(2, 0), goal_pos=(3, 4), obstacles=None):
        super(GridWorldEnv, self).__init__()

        # Save parameters
        self.grid_size = grid_size
        self.start_pos = start_pos
        self.goal_pos = goal_pos

        # Default obstacles
        if obstacles is None:
            self.obstacles = {(1, 3), (2, 1), (3, 3), (4,0), (0,4)}
        else:
            self.obstacles = set(obstacles)

        # Agent position
        self.agent_pos = list(self.start_pos)

        # Action space: Up, Down, Left, Right
        self.action_space = spaces.Discrete(4)

        # Observation space: agent (x, y)
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = list(self.start_pos)
        return np.array(self.agent_pos, dtype=np.int32), {}

    def step(self, action):
        x, y = self.agent_pos

        if action == 0:   # Up
            new_pos = (x - 1, y)
        elif action == 1: # Down
            new_pos = (x + 1, y)
        elif action == 2: # Left
            new_pos = (x, y - 1)
        else:             # Right
            new_pos = (x, y + 1)

        # Check boundaries
        if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
            # Check obstacle
            if new_pos not in self.obstacles:
                self.agent_pos = list(new_pos)

        # Compute reward
        if tuple(self.agent_pos) == self.goal_pos:
            reward = 1.0
            terminated = True
        else:
            reward = -0.01
            terminated = False

        return (
            np.array(self.agent_pos, dtype=np.int32),
            reward,
            terminated,
            False,  # truncated
            {},
        )

    def render(self):
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Draw obstacles safely
        for ox, oy in self.obstacles:
            if 0 <= ox < self.grid_size and 0 <= oy < self.grid_size:
                grid[ox][oy] = "X"

        # Draw goal
        gx, gy = self.goal_pos
        if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
            grid[gx][gy] = "G"

        # Draw agent
        ax, ay = self.agent_pos
        if 0 <= ax < self.grid_size and 0 <= ay < self.grid_size:
            grid[ax][ay] = "A"

        # Print grid
        for row in grid:
            print(" ".join(row))
        print()
