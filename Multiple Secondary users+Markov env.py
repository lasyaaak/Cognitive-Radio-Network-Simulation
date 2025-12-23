import numpy as np
import random
import matplotlib.pyplot as plt
import time
import pickle
import os

# Parameters
num_channels = 10
steps = 1000
visual_steps = 20
alpha = 0.1
epsilon = 0.1
gamma = 0.9
num_users = 3

# Markov transition probabilities
transition_probs = {
    0: [0.8, 0.2],
    1: [0.2, 0.8]
}

# Initialize Q-tables
Q_tables = [{} for _ in range(num_users)]
Q_files = [f"q_table_user{i+1}.pkl" for i in range(num_users)]

# Load Q-tables if they exist
for i, file in enumerate(Q_files):
    if os.path.exists(file):
        with open(file, 'rb') as f:
            loaded_q = pickle.load(f)
            Q_tables[i] = {tuple(k): v for k, v in loaded_q.items()}
        print(f"\n Q-table for User {i+1} loaded from file.")

# Functions
def get_next_channel_states(current_states):
    return [np.random.choice([0, 1], p=transition_probs[state]) for state in current_states]

def get_action(Q, state, epsilon):
    if state not in Q:
        Q[state] = [0.0] * num_channels
    return random.randint(0, num_channels - 1) if random.random() < epsilon else int(np.argmax(Q[state]))

def update_q(Q, state, action, reward, next_state):
    if state not in Q:
        Q[state] = [0.0] * num_channels
    if next_state not in Q:
        Q[next_state] = [0.0] * num_channels
    Q[state][action] += alpha * (reward + gamma * max(Q[next_state]) - Q[state][action])

# -----------------------
# Training Phase
# -----------------------
channel_states = [random.randint(0, 1) for _ in range(num_channels)]
train_rewards = [0] * num_users

for step in range(steps):
    state = tuple(channel_states)
    actions = [get_action(Q_tables[i], state, epsilon) for i in range(num_users)]

    # Detect collisions
    collisions = {a: actions.count(a) for a in actions}

    rewards = []
    for i, action in enumerate(actions):
        if channel_states[action] == 1 and collisions[action] == 1:
            reward = 1
        else:
            reward = -1  # Penalize for busy channel or collision
        rewards.append(reward)
        train_rewards[i] += int(reward == 1)

    next_channel_states = get_next_channel_states(channel_states)
    next_state = tuple(next_channel_states)

    for i in range(num_users):
        update_q(Q_tables[i], state, actions[i], rewards[i], next_state)

    channel_states = next_channel_states

print("\n Training complete")
for i in range(num_users):
    print(f"User {i+1} Training Success Rate: {train_rewards[i]/steps*100:.2f}%")

# Save Q-tables
for i, file in enumerate(Q_files):
    with open(file, 'wb') as f:
        pickle.dump(Q_tables[i], f)

# -----------------------
# Visualization Phase
# -----------------------
# -----------------------
# Visualization Phase (Updated: Only Final Heatmap)
# -----------------------
channel_states = [random.randint(0, 1) for _ in range(num_channels)]
visual_grid = np.zeros((visual_steps, num_channels))
selected_actions = [[] for _ in range(num_users)]
success_counts = [0] * num_users

for step in range(visual_steps):
    state = tuple(channel_states)
    actions = [get_action(Q_tables[i], state, epsilon=0.01) for i in range(num_users)]
    for i in range(num_users):
        selected_actions[i].append(actions[i])

    collisions = {a: actions.count(a) for a in actions}

    for i, action in enumerate(actions):
        if channel_states[action] == 1 and collisions[action] == 1:
            success_counts[i] += 1

    visual_grid[step] = state
    channel_states = get_next_channel_states(channel_states)

# Plot only the final heatmap after all steps
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(1 - visual_grid, cmap='gray', aspect='auto', vmin=0, vmax=1)

colors = ['red', 'blue', 'green']
for user_id in range(num_users):
    for t in range(visual_steps):
        rect = plt.Rectangle((selected_actions[user_id][t] - 0.5, t - 0.5), 1, 1,
                             linewidth=1.5, edgecolor=colors[user_id], facecolor='none')
        ax.add_patch(rect)

ax.set_title(f"Multi-User Q-Learning: Final Heatmap ({visual_steps} Steps)")
ax.set_xlabel("Channel")
ax.set_ylabel("Time Step")
ax.set_xticks(range(num_channels))

# Add success rate annotation
for i in range(num_users):
    ax.text(num_channels + 0.5, i * 1.5,
            f"User {i+1} Success: {success_counts[i]/visual_steps*100:.1f}%",
            fontsize=10, color=colors[i])

plt.tight_layout()
plt.savefig("final_heatmap.png")
plt.show()
