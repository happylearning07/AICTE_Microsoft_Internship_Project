import numpy as np
import matplotlib.pyplot as plt
import random


class FarmingEnvironment:
    def __init__(self, grid_size=5, initial_money=10, seed_cost=1, crop_sell_price=5):
        self.grid_size = grid_size
        self.grid = [['empty' for _ in range(grid_size)] for _ in range(grid_size)]
        self.money = initial_money
        self.seed_cost = seed_cost
        self.crop_sell_price = crop_sell_price
        self.total_crops_harvested = 0

    def reset(self):
        self.grid = [['empty' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.money = 10
        self.total_crops_harvested = 0
        return self._get_state()

    def _get_state(self):
        # Flatten the grid + money as the state representation
        flat_grid = []
        for row in self.grid:
            for cell in row:
                if cell == 'empty':
                    flat_grid.append(0)
                elif cell == 'planted':
                    flat_grid.append(1)
                elif cell == 'ready':
                    flat_grid.append(2)
        flat_grid.append(self.money)
        return tuple(flat_grid)

    def step(self, action, x, y):
        reward = 0
        done = False

        if action == 'plant':
            if self.grid[x][y] == 'empty' and self.money >= self.seed_cost:
                self.grid[x][y] = 'planted'
                self.money -= self.seed_cost
            else:
                reward -= 1  # waste of action

        elif action == 'water':
            if self.grid[x][y] == 'planted':
                self.grid[x][y] = 'ready'
            else:
                reward -= 1

        elif action == 'harvest':
            if self.grid[x][y] == 'ready':
                self.grid[x][y] = 'empty'
                self.money += self.crop_sell_price
                self.total_crops_harvested += 1
                reward += 10
            else:
                reward -= 1

        # Optional penalty for dead crops (if you want to expand)
        state = self._get_state()

        return state, reward, done





class QLearningAgent:
    def __init__(self, actions, grid_size, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = {}
        self.actions = actions
        self.grid_size = grid_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_state_key(self, state):
        return str(state)

    def choose_action(self, state):
        state_key = self.get_state_key(state)
        
        # Initialize Q-table entry if state not present
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros((len(self.actions), self.grid_size, self.grid_size))

        # Epsilon-greedy exploration
        if np.random.uniform(0, 1) < self.epsilon:
            action_idx = random.randint(0, len(self.actions) - 1)
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
        else:
            q_values = self.q_table[state_key]
            action_idx, x, y = np.unravel_index(np.argmax(q_values, axis=None), q_values.shape)

        return action_idx, x, y

    def learn(self, state, action_idx, x, y, reward, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        # Initialize next state Q-table entry if not present
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros((len(self.actions), self.grid_size, self.grid_size))

        predict = self.q_table[state_key][action_idx][x][y]
        target = reward + self.gamma * np.max(self.q_table[next_state_key])

        self.q_table[state_key][action_idx][x][y] += self.alpha * (target - predict)







env = FarmingEnvironment(grid_size=5)
actions = np.array(['plant', 'water', 'harvest'])
agent = QLearningAgent(actions, grid_size=env.grid_size)

# Training loop
episodes = 100
rewards_per_episode = []

for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    for step in range(50):  # limit number of steps per episode
        action_idx, x, y = agent.choose_action(state)
        action = actions[action_idx]

        next_state, reward, done = env.step(action, x, y)
        agent.learn(state, action_idx, x, y, reward, next_state)

        state = next_state
        total_reward += reward

    rewards_per_episode.append(total_reward)
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Farm Agent Performance Over Time')
plt.show()
