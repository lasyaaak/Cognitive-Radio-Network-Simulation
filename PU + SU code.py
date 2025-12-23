import random
import matplotlib.pyplot as plt

# Parameters
num_channels = 10
time_slots = 100
primary_occupancy_prob = 0.3

channel_states = []
su_choices = []
su_results = []

# Generate PU occupancy
for _ in range(time_slots):
    channels = [1 if random.random() < primary_occupancy_prob else 0 for _ in range(num_channels)]
    channel_states.append(channels)

# Simulate SU
for t in range(time_slots):
    chosen_channel = random.randint(0, num_channels - 1)  # SU picks a random channel
    su_choices.append(chosen_channel)

    if channel_states[t][chosen_channel] == 0:
        su_results.append(1)  # Success
    else:
        su_results.append(0)  # Collision

# Success rate
success_rate = sum(su_results) / time_slots * 100
print(f"SU Success Rate (Random Strategy): {success_rate:.2f}%")

# Plot PU occupancy
plt.figure(figsize=(12, 6))
plt.imshow(channel_states, aspect='auto', cmap='gray_r')
plt.scatter(su_choices, range(time_slots), c='red', s=10, label='SU Choice')
plt.xlabel('Channel Number')
plt.ylabel('Time Slot')
plt.title('Primary User Occupancy & Secondary User Decisions')
plt.legend()
plt.colorbar(label='Occupied (1) / Free (0)')
plt.show()
