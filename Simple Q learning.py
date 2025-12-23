import random
import matplotlib.pyplot as plt

# Number of channels and time slots
num_channels = 10
time_slots = 100

# Simulate PU activity: 1 = occupied, 0 = free
primary_occupancy_prob = 0.3
channel_states = []
for _ in range(time_slots):
    channels = [1 if random.random() < primary_occupancy_prob else 0 for _ in range(num_channels)]
    channel_states.append(channels)

# Initialize Q-table for SU
Q = [0.0] * num_channels
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.2  # Exploration rate

su_choices = []
su_results = []

# Q-learning loop
for t in range(time_slots):
    if random.random() < epsilon:
        action = random.randint(0, num_channels - 1)  # Explore
    else:
        max_q = max(Q)
        best_actions = [i for i, q in enumerate(Q) if q == max_q]
        action = random.choice(best_actions)  # Exploit

    su_choices.append(action)

    # Reward: 1 if channel is free, 0 if occupied
    reward = 1 if channel_states[t][action] == 0 else 0
    su_results.append(reward)

    # Q-value update
    Q[action] = Q[action] + alpha * (reward - Q[action])

# Evaluate success rate
success_rate = sum(su_results) / time_slots * 100
print(f"SU Success Rate: {success_rate:.2f}%")

# Visualize
plt.figure(figsize=(12, 6))
plt.imshow(channel_states, aspect='auto', cmap='gray_r')
plt.scatter(su_choices, range(time_slots), c='red', s=10, label='SU Choice')
plt.xlabel('Channel')
plt.ylabel('Time Slot')
plt.title('Primary User Occupancy and SU Choices (Q-learning)')
plt.legend()
plt.colorbar(label='PU: 1 = occupied, 0 = free')
plt.show()
