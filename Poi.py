import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# === CONFIGURATION ===
file_path = "D:\\Projects\\DPA_contest2_public_base_diff_vcc_a128_2009_12_23\\wave_aist-aes-agilent_2009-12-30_22-53-19_n=0_k=0000000000000003243f6a8885a308d3_m=0000000000000002b7e151628aed2a6a_c=e6a636e30c85f35e980f3546a04daff7.csv"
skip_header_rows = 6
x_increment = 2e-10  # 200 ps = 2e-10 seconds per sample

# === Time Range (in units of 1e-7 seconds) ===
x_start_1e7 = 0.773
x_end_1e7 = 0.897

# === STEP 1: Load CSV and Clean ===
df = pd.read_csv(file_path, skiprows=skip_header_rows, header=None)
trace = pd.to_numeric(df[0], errors='coerce').dropna().values.flatten()

# === STEP 2: Create Time Axis ===
time_axis = np.arange(len(trace)) * x_increment
time_axis_scaled = time_axis * 1e7  # To plot in 1e-7 s

# === STEP 3: Convert Time to Sample Indices ===
x_start = int((x_start_1e7 * 1e-7) / x_increment)
x_end = int((x_end_1e7 * 1e-7) / x_increment)

# Clamp to valid range
x_start = max(0, x_start)
x_end = min(len(trace), x_end)

# Debug check
print(f"x_start = {x_start}, x_end = {x_end}, length = {x_end - x_start}")

# Final length validation
if x_end - x_start < 5:
    raise ValueError("Selected time range is too short to analyze!")


# === STEP 4: Extract Segment and Derivatives ===
trace_segment = trace[x_start:x_end]
time_segment = time_axis_scaled[x_start:x_end]

first_derivative = np.gradient(trace_segment)
second_derivative = np.gradient(first_derivative)

# === STEP 5: Find Points of Interest ===
zero_crossings = np.where(np.diff(np.sign(first_derivative)))[0]
inflection_points = np.where(np.diff(np.sign(second_derivative)))[0]
sharp_transitions, _ = find_peaks(np.abs(first_derivative),
                                  height=np.max(np.abs(first_derivative)) * 0.3)

# === STEP 6: Plot Full Trace and Overlay Segment ===
plt.figure(figsize=(15, 6))
plt.plot(time_axis_scaled, trace, label="Full Trace", color='gray', linewidth=0.8)
plt.axvspan(x_start_1e7, x_end_1e7, color='yellow', alpha=0.3, label="Selected Segment")

# === STEP 7: Overlay Segment with POIs ===
plt.plot(time_segment, trace_segment, color='black', label="Selected Trace Segment")

plt.scatter(time_segment[zero_crossings], trace_segment[zero_crossings], color='blue', s=20, label="Zero Crossings")
plt.scatter(time_segment[inflection_points], trace_segment[inflection_points], color='orange', s=20, label="Inflection Points")
plt.scatter(time_segment[sharp_transitions], trace_segment[sharp_transitions], color='red', s=20, label="Sharp Transitions")

plt.title("Power Trace with Points of Interest (Time in ×1e-7 seconds)")
plt.xlabel("Time (×1e-7 s)")
plt.ylabel("Power")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === STEP 8: Print Summary of POIs ===
print(f"\n[INFO] Time range: {x_start_1e7} to {x_end_1e7} (×1e-7 s)")
print(f"Sample range: {x_start} to {x_end} — total: {x_end - x_start} points")
print("\nFirst few POIs (sharp transitions):")
for i in range(min(10, len(sharp_transitions))):
    idx = sharp_transitions[i]
    print(f"  Time = {time_segment[idx]:.5f} ×1e-7 s, Power = {trace_segment[idx]:.5f}")
