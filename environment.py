import numpy as np


class GridEnvironment:
    def __init__(self):
        self.n_states = 25  # 5x5 Grid
        self.n_actions = 4  # Up, Down, Left, Right
        self.goal_state = 24  # Bottom-right corner
        self.pit_state = 12  # Center trap
        self.state = 0

    def reset(self):
        # Start at a random state (excluding goal/pit)
        while True:
            self.state = np.random.randint(0, self.n_states)
            if self.state != self.goal_state and self.state != self.pit_state:
                break
        return self.state

    def step(self, action):
        # Grid dimensions
        row = self.state // 5
        col = self.state % 5

        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        if action == 0:
            row = max(0, row - 1)
        elif action == 1:
            row = min(4, row + 1)
        elif action == 2:
            col = max(0, col - 1)
        elif action == 3:
            col = min(4, col + 1)

        # Convert back to state index
        new_state = row * 5 + col
        self.state = new_state

        # Calculate Reward & Done flag
        reward = -1  # Step cost (default)
        done = False

        if new_state == self.goal_state:
            reward = 10
            done = True
        elif new_state == self.pit_state:
            reward = -10
            done = True

        return new_state, reward, done