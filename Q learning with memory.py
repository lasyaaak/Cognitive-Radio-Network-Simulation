import random
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# --- Parameters ---
num_channels = 10
num_steps = 1000
alpha = 0.1
epsilon = 0.1
channel_probabilities = [0.8, 0.6, 0.4, 0.2, 0.9, 0.5, 0.3, 0.7, 0.85, 0.65]  # Fixed probabilities

# --- Load or initialize Q-table ---
if os.path.exists('q_table.json'):
    with open('q_table.json', 'r') as f:
        Q = json.load(f)
    print("✅ Loaded existing Q-table.")
else:
    Q = [0.0] * num_channels
    print("⚙️ Initialized new Q-table.")

# --- Simulation ---
success_history = []
learning_curve = []

for step in range(num_steps):
    # Generate channel states: 1 = free (black), 0 = busy (white)
    channel_states = [1 if random.random() < p else 0 for p in channel_probabilities]

    # ε-greedy action selection
    if random.random() < epsilon:
        action = random.randint(0, num_channels - 1)
    else:
        action = Q.index(max(Q))

    # Reward and Q update
    reward = channel_states[action]
    Q[action] = Q[action] + alpha * (reward - Q[action])

    # Record performance
    success_history.append(reward)
    learning_curve.append(Q[action])

# --- Save Q-table ---
with open('q_table.json', 'w') as f:
    json.dump(Q, f)
print("💾 Q-table saved.")

# --- Final Q-values and Success ---
print("\nFinal Q-values:")
for i, q in enumerate(Q):
    print(f"Channel {i}: {q:.3f}")

success_rate = sum(success_history) / num_steps * 100
print(f"\n✅ Overall Success Rate: {success_rate:.2f}%")

# --- Moving Average Plot (Success Rate) ---
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

plt.figure(figsize=(10, 4))
plt.plot(moving_average(success_history, 50), label='Success Rate (Moving Avg)', color='green')
plt.title("Learning Trend (Success Rate over Time)")
plt.xlabel("Step")
plt.ylabel("Success Rate")
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Learning Curve (Q-value of selected actions) ---
plt.figure(figsize=(10, 4))
plt.plot(learning_curve, label='Q-value (Selected Action)', color='blue', alpha=0.6)
plt.title("Learning Curve (Q-values of Selected Actions)")
plt.xlabel("Step")
plt.ylabel("Q-value")
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
