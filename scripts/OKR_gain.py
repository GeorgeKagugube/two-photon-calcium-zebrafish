import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('eye_mov.csv')

time = df['time'].values  # Time in seconds
left_eye = df['left_eye'].values  # Left eye position
right_eye = df['right_eye'].values  # Right eye position

# Define stimulus velocity (constant)
stimulus_velocity = 0.86  # deg/s (example value, adjust as needed)

# Compute Eye Velocity
def compute_velocity(position, time):
    return np.gradient(position, time)

vel_left = compute_velocity(left_eye, time)
vel_right = compute_velocity(right_eye, time)

# Compute OKR Gain
def compute_okr_gain(eye_velocity, stimulus_velocity):
    return np.mean(eye_velocity) / stimulus_velocity

okr_gain_left = compute_okr_gain(vel_left, stimulus_velocity)
okr_gain_right = compute_okr_gain(vel_right, stimulus_velocity)

# Display results
print(f"OKR Gain (Left Eye): {okr_gain_left:.2f}")
print(f"OKR Gain (Right Eye): {okr_gain_right:.2f}")

# Plot eye velocity and stimulus velocity
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(time[:-1], vel_left[:-1], label='Left Eye Velocity', color='blue')
ax.plot(time[:-1], vel_right[:-1], label='Right Eye Velocity', color='red')
ax.axhline(y=stimulus_velocity, color='green', linestyle='dashed', label='Stimulus Velocity')
ax.set_title('Eye Velocity and Stimulus Velocity')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Velocity (deg/s)')
ax.legend()

plt.show()
