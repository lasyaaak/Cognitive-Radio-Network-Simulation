import numpy as np
import random
import matplotlib.pyplot as plt
import time
import json
import os

# Parameters
num_channels = 10
steps = 1000
visual_steps = 20
alpha = 0.1
epsilon = 0.1
gamma = 0.9
q_table_file = "q_table.json"

# Markov transition probabilities
transition_probs = {
    0: [0.8, 0.2],  # Busy -> Busy / Free
    1: [0.2, 0.8]   # Free -> Busy / Free
}

# Initialize Q-table
Q = {}

# Load existing Q-table
if os.path.exists(q_table_file):
    with open(q_table_file, "r") as f:
        raw_q = json.load(f)
        Q = {eval(k): v for k, v in raw_q.items()}
    print("📂 Q-table loaded from file.")
else:
    print("🆕 Starting with empty Q-table.")

# Initialize channel states randomly
channel_states = [random.randint(0, 1) for _ in range(num_channels)]

def get_next_channel_states(current_states):
    return [np.random.choice([0, 1], p=transition_probs[s]) for s in current_states]

def get_action(state, epsilon):
    if state not in Q:
        Q[state] = [0.0] * num_channels
    if random.random() < epsilon:
        return random.randint(0, num_channels - 1)
    return int(np.argmax(Q[state]))

def update_q_table(state, action, reward, next_state):
    if state not in Q:
        Q[state] = [0.0] * num_channels
    if next_state not in Q:
        Q[next_state] = [0.0] * num_channels
    Q[state][action] += alpha * (reward + gamma * max(Q[next_state]) - Q[state][action])

# -----------------------
# Training Phase
# -----------------------
train_rewards = []
for step in range(steps):
    state = tuple(channel_states)
    action = get_action(state, epsilon)
    reward = 1 if channel_states[action] == 1 else 0
    train_rewards.append(reward)

    next_channel_states = get_next_channel_states(channel_states)
    next_state = tuple(next_channel_states)

    update_q_table(state, action, reward, next_state)
    channel_states = next_channel_states

print(f"\n✅ Training complete\nTraining Success Rate: {sum(train_rewards)/steps*100:.2f}%")

# Save Q-table to file
with open(q_table_file, "w") as f:
    json.dump({str(k): v for k, v in Q.items()}, f)
print("💾 Q-table saved to file.")

# -----------------------
# Visualization Phase
# -----------------------
channel_states = [random.randint(0, 1) for _ in range(num_channels)]
visual_grid = np.zeros((visual_steps, num_channels))
selected_actions = []

success_count = 0

for step in range(visual_steps):
    state = tuple(channel_states)
    action = get_action(state, epsilon=0.01)
    selected_actions.append(action)
    reward = 1 if channel_states[action] == 1 else 0
    success_count += reward
    visual_grid[step] = state
    channel_states = get_next_channel_states(channel_states)

# Final Heatmap Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.imshow(1 - visual_grid, cmap='gray', aspect='auto', vmin=0, vmax=1)

for t in range(visual_steps):
    rect = plt.Rectangle((selected_actions[t] - 0.5, t - 0.5), 1, 1,
                         linewidth=1.5, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

ax.set_title(f"Final Visualization | Success Rate: {success_count/visual_steps*100:.2f}%")
ax.set_xlabel("Channel")
ax.set_ylabel("Time Step")
ax.set_xticks(range(num_channels))
plt.tight_layout()

# 💾 Save the heatmap as PNG
plt.savefig("final_heatmap.png")
plt.show()


print(f"\n🎯 Visual Test Success Rate: {success_count/visual_steps*100:.2f}%")

