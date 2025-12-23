import random
import matplotlib.pyplot as plt

# Parameters
num_channels = 10            # Number of channels in the CRN
time_slots = 100             # Total time steps
primary_occupancy_prob = 0.3 # Probability a channel is occupied by a primary user

# Simulate primary user occupancy
channel_states = []

for _ in range(time_slots):
    channels = [1 if random.random() < primary_occupancy_prob else 0 for _ in range(num_channels)]
    channel_states.append(channels)

# Plot the occupancy heatmap
plt.figure(figsize=(12, 6))
plt.imshow(channel_states, aspect='auto', cmap='gray_r')
plt.xlabel('Channel Number')
plt.ylabel('Time Slot')
plt.title('Primary User Channel Occupancy Over Time')
plt.colorbar(label='Occupied (1) / Free (0)')
plt.show()

