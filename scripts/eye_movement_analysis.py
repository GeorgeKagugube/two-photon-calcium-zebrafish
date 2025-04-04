import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, correlate
from scipy.fftpack import fft, fftfreq

# Load sample data (replace with actual file path)
# The data should be a CSV with 'time', 'left_eye', 'right_eye' columns
data = pd.read_csv('eye_mov.csv')

time = data['time'].values  # Time in seconds
left_eye = data['left_eye'].values  # Left eye position (degrees)
right_eye = data['right_eye'].values  # Right eye position (degrees)

# 1. Preprocessing: Apply Butterworth low-pass filter
def butter_lowpass_filter(data, cutoff=10, fs=100, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

filtered_left_eye = butter_lowpass_filter(left_eye)
filtered_right_eye = butter_lowpass_filter(right_eye)

# 2. Saccade Detection based on velocity threshold
def detect_saccades(eye_data, threshold=20):
    velocity = np.diff(eye_data) / np.diff(time)  # Compute velocity
    peaks, _ = find_peaks(np.abs(velocity), height=threshold)  # Identify saccades
    return peaks, velocity

left_saccades, left_velocity = detect_saccades(filtered_left_eye)
right_saccades, right_velocity = detect_saccades(filtered_right_eye)

# 3. Cross-Correlation Analysis (synchrony between left and right eye movements)
def cross_correlation(x, y):
    correlation = correlate(x - np.mean(x), y - np.mean(y), mode='full')
    lag = np.arange(-len(x) + 1, len(x)) * np.mean(np.diff(time))  # Compute lag time
    return lag, correlation

lag, cross_corr = cross_correlation(filtered_left_eye, filtered_right_eye)

# 4. Frequency-Domain Analysis (FFT)
def compute_fft(signal, fs=100):
    n = len(signal)
    freq = fftfreq(n, d=1/fs)
    fft_values = np.abs(fft(signal))
    return freq[:n//2], fft_values[:n//2]

freqs_left, fft_left = compute_fft(filtered_left_eye)
freqs_right, fft_right = compute_fft(filtered_right_eye)
print (f'Peak: {left_saccades} and Velocity: {left_velocity}')
print (f'Peak: {right_saccades} and Velocity: {right_velocity}')

# Plot Results
fig, ax = plt.subplots(3, 1, figsize=(10, 8))

# Plot eye movement traces
ax[0].plot(time, filtered_left_eye, label='Left Eye', color='blue')
ax[0].plot(time, filtered_right_eye, label='Right Eye', color='red')
ax[0].scatter(time[left_saccades], filtered_left_eye[left_saccades], color='cyan', label='Left Saccades')
ax[0].scatter(time[right_saccades], filtered_right_eye[right_saccades], color='magenta', label='Right Saccades')
ax[0].set_title('Filtered Eye Movement Traces')
ax[0].legend()

# Plot cross-correlation
ax[1].plot(lag, cross_corr, color='green')
ax[1].set_title('Cross-Correlation between Left and Right Eye Movements')
ax[1].set_xlabel('Lag (s)')

# Plot FFT
ax[2].plot(freqs_left, fft_left, label='Left Eye FFT', color='blue')
ax[2].plot(freqs_right, fft_right, label='Right Eye FFT', color='red')
ax[2].set_title('Frequency Spectrum of Eye Movements')
ax[2].set_xlabel('Frequency (Hz)')
ax[2].legend()

plt.tight_layout()
plt.show()