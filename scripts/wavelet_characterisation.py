import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load the dataset
file_path = "/mnt/data/eye_mov.csv"
df = pd.read_csv('eye_mov.csv')

time = df['time'].values  # Time in seconds
left_eye = df['left_eye'].values  # Left eye position
right_eye = df['right_eye'].values  # Right eye position

# Define temporal frequency (Hz) and spatial frequency (cycles/degree)
temporal_frequency = 0.0167  # Example value in Hz
spatial_frequency = 0.4  # Example value in cycles/degree

# Estimate Stimulus Speed (deg/s)
stimulus_speed = temporal_frequency / spatial_frequency

# Estimate Stimulus Amplitude using A = Stimulus Speed / (2 * pi * Temporal Frequency)
def estimate_stimulus_amplitude(stimulus_speed, temporal_frequency):
    return stimulus_speed / (2 * np.pi * temporal_frequency)

stimulus_amplitude = estimate_stimulus_amplitude(stimulus_speed, temporal_frequency)

# Compute Amplitude (Peak-to-Peak)
def compute_amplitude(signal):
    return np.max(signal) - np.min(signal)

amp_left = compute_amplitude(left_eye)
amp_right = compute_amplitude(right_eye)

# Compute Velocity (First derivative of position)
def compute_velocity(position, time):
    return np.gradient(position, time)

vel_left = compute_velocity(left_eye, time)
vel_right = compute_velocity(right_eye, time)

# Compute Number of Saccades (Peaks in velocity exceeding a threshold)
def detect_saccades(velocity, threshold=20):
    peaks, _ = find_peaks(np.abs(velocity), height=threshold)
    return len(peaks)

num_saccades_left = detect_saccades(vel_left)
num_saccades_right = detect_saccades(vel_right)

# Display results
print(f"Estimated Stimulus Speed: {stimulus_speed:.2f} deg/s")
print(f"Estimated Stimulus Amplitude: {stimulus_amplitude:.2f} degrees")
print(f"Left Eye Amplitude: {amp_left:.2f} degrees")
print(f"Right Eye Amplitude: {amp_right:.2f} degrees")
print(f"Left Eye Velocity Mean: {np.mean(vel_left):.2f} deg/s")
print(f"Right Eye Velocity Mean: {np.mean(vel_right):.2f} deg/s")
print(f"Number of Saccades (Left Eye): {num_saccades_left}")
print(f"Number of Saccades (Right Eye): {num_saccades_right}")

# Plot eye movement and velocity
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
ax[0].plot(time, left_eye, label='Left Eye', color='blue')
ax[0].plot(time, right_eye, label='Right Eye', color='red')
ax[0].set_title('Eye Movement Position')
ax[0].legend()

ax[1].plot(time[:-1], vel_left[:-1], label='Left Eye Velocity', color='cyan')
ax[1].plot(time[:-1], vel_right[:-1], label='Right Eye Velocity', color='magenta')
ax[1].set_title('Eye Movement Velocity')
ax[1].legend()

plt.tight_layout()
plt.show()