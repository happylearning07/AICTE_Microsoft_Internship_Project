# 🌾 Farming Environment with Q-Learning Agent

This project demonstrates how a reinforcement learning agent (using **Q-learning**) can be trained to make intelligent farming decisions such as **planting**, **watering**, and **harvesting crops** in a grid-based farming environment.

---

## 🧠 Problem Statement

Traditional farming involves many manual decisions like when and where to plant, water, or harvest. These decisions are often inefficient, leading to wasted resources and reduced productivity. This project explores how **AI can automate and optimize farming operations** using reinforcement learning.

---

## 💡 Proposed Solution

I built a custom grid-based **farming environment** where:
- Each cell can be in one of three states: `empty`, `planted`, or `ready`.
- An agent interacts with this environment using the actions: `'plant'`, `'water'`, and `'harvest'`.
- The agent is trained using **tabular Q-learning** to learn optimal farming strategies over time.

---

## 🔧 Technologies Used

- **Python 3.x**
- **NumPy** – for Q-table and math operations
- **Matplotlib** – to visualize training performance
- **Random** – for stochastic action selection

---

## 📊 Q-Learning Algorithm

Q-learning is used to estimate the optimal action-value function:

![image](https://github.com/user-attachments/assets/af9b9a30-02db-42ed-bc4b-8fc12e7199a1)

- **α (alpha):** Learning rate
- **γ (gamma):** Discount factor
- **ε (epsilon):** Exploration rate (used in ε-greedy strategy)

---

## 🚜 Farming Environment

- The farm is modeled as a `grid_size × grid_size` matrix (default 5×5).
- Actions are:
  - `plant` → if cell is `empty` and agent has money
  - `water` → if cell is `planted`, it becomes `ready`
  - `harvest` → if cell is `ready`, earn profit and reset cell to `empty`
- Rewards are given based on correct or incorrect actions.
- State is represented as a flattened grid + current money.

---

## 📈 Training & Results

- The agent is trained for multiple episodes (e.g., 100).
- Performance improves over time as seen in the increasing total rewards per episode.

- ![image](https://github.com/user-attachments/assets/98b7959f-09c4-4aec-8745-814d662c471e)



