import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, deconvolve
from scipy.stats import linregress

# Load the dataset
#file_path = "/mnt/data/eye_mov.csv"
#df = pd.read_csv(file_path)
df = pd.read_csv('eye_mov.csv')

time = df['time'].values  # Time in seconds
left_eye = df['left_eye'].values  # Left eye position
right_eye = df['right_eye'].values  # Right eye position

# Deconvolution step
# Assume an estimated impulse response function (kernel) for deconvolution
kernel = np.ones(5) / 5  # Example moving average kernel

def apply_deconvolution(signal, kernel):
    deconv_result, _ = deconvolve(signal, kernel)
    return deconv_result

left_eye_deconv = apply_deconvolution(left_eye, kernel)
right_eye_deconv = apply_deconvolution(right_eye, kernel)

# Detect Saccades (Peaks in velocity exceeding a threshold)
def detect_saccades(position, time, threshold=5):
    velocity = np.gradient(position, time)
    peaks, _ = find_peaks(np.abs(velocity), height=threshold)
    return peaks, velocity

left_saccades, left_velocity = detect_saccades(left_eye_deconv, time)
right_saccades, right_velocity = detect_saccades(right_eye_deconv, time)

# Function to extract linear sections and estimate slopes
def estimate_slopes(position, time, saccades):
    slopes = []
    start_idx = 0
    
    for saccade in saccades:
        if start_idx < saccade - 1:
            x_section = time[start_idx:saccade]
            y_section = position[start_idx:saccade]
            if len(x_section) > 1:  # Avoid empty segments
                slope, _, _, _, _ = linregress(x_section, y_section)
                slopes.append(slope)
        start_idx = saccade + 1
    
    # Final segment after last saccade
    if start_idx < len(position) - 1:
        x_section = time[start_idx:]
        y_section = position[start_idx:]
        if len(x_section) > 1:
            slope, _, _, _, _ = linregress(x_section, y_section)
            slopes.append(slope)
    
    return np.mean(slopes) if slopes else 0

# Compute mean slopes for linear sections
mean_slope_left = estimate_slopes(left_eye_deconv, time, left_saccades)
mean_slope_right = estimate_slopes(right_eye_deconv, time, right_saccades)

# Save results to a file
output_file = "linear_section_slopes_deconv.csv"
slope_data = pd.DataFrame({
    "Eye": ["Left", "Right"],
    "Mean Slope": [mean_slope_left, mean_slope_right]
})
slope_data.to_csv(output_file, index=False)

print(f"Slope estimates saved to {output_file}")

# Plot the deconvolved eye movement trace and detected saccades
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
ax[0].plot(time[:len(left_eye_deconv)], left_eye_deconv, label='Left Eye Position (Deconvolved)', color='blue')
ax[0].scatter(time[left_saccades], left_eye_deconv[left_saccades], color='red', label='Saccades', zorder=3)
ax[0].set_title('Left Eye Movement with Saccades (Deconvolved)')
ax[0].legend()

ax[1].plot(time[:len(right_eye_deconv)], right_eye_deconv, label='Right Eye Position (Deconvolved)', color='green')
ax[1].scatter(time[right_saccades], right_eye_deconv[right_saccades], color='red', label='Saccades', zorder=3)
ax[1].set_title('Right Eye Movement with Saccades (Deconvolved)')
ax[1].legend()

plt.tight_layout()
plt.show()
