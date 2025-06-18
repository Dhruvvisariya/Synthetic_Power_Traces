import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline, lagrange
from numpy.polynomial.chebyshev import Chebyshev

# === Newton Interpolation Functions ===
def newton_divided_diff(x, y):
    n = len(x)
    coef = np.copy(y)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j - 1:n - 1]) / (x[j:n] - x[0:n - j])
    return coef

def newton_poly(coef, x_data, x):
    n = len(coef)
    result = coef[-1]
    for i in range(n - 2, -1, -1):
        result = result * (x - x_data[i]) + coef[i]
    return result

# === Load and clean CSV data ===
file_path = "D:\\Projects\\DPA_contest2_public_base_diff_vcc_a128_2009_12_23\\wave_aist-aes-agilent_2009-12-30_22-53-19_n=0_k=0000000000000003243f6a8885a308d3_m=0000000000000002b7e151628aed2a6a_c=e6a636e30c85f35e980f3546a04daff7.csv"
df = pd.read_csv(file_path, skiprows=6, header=None)
trace = pd.to_numeric(df[0], errors='coerce').dropna().values

# === Time axis (200ps step) ===
x_increment = 2e-10
time = np.arange(len(trace)) * x_increment

# === Focused time window in original units (1e-7 s) ===
start_1e7 = 0.665
end_1e7 = 0.773
start_idx = int((start_1e7 * 1e-7) / x_increment)
end_idx = int((end_1e7 * 1e-7) / x_increment)

start_idx = max(0, start_idx)
end_idx = min(len(trace), end_idx)

# === Extract the segment ===
segment = trace[start_idx:end_idx]
segment_time = time[start_idx:end_idx]

# === Derivatives and POIs ===
d1 = np.gradient(segment)
d2 = np.gradient(d1)
zc = np.where(np.diff(np.sign(d1)))[0]
ip = np.where(np.diff(np.sign(d2)))[0]
peaks, _ = find_peaks(np.abs(d1), height=np.max(np.abs(d1)) * 0.3)

poi_indices = np.unique(np.concatenate([zc, ip, peaks]))
poi_x = segment_time[poi_indices]
poi_y = segment[poi_indices]

# === Normalize segment and POIs ===
segment_time -= segment_time[0]
segment -= np.min(segment)

poi_x -= poi_x[0]
poi_y -= np.min(poi_y)

# === Sort POIs ===
sort_idx = np.argsort(poi_x)
poi_x = poi_x[sort_idx]
poi_y = poi_y[sort_idx]
poi_x, unique_idx = np.unique(poi_x, return_index=True)
poi_y = poi_y[unique_idx]

# === Interpolation domain ===
x_smooth = np.linspace(poi_x[0], poi_x[-1], len(segment_time))

# === Lagrange ===
lagrange_poly = lagrange(poi_x, poi_y)
y_lagrange = lagrange_poly(x_smooth)

# === Spline ===
spline = CubicSpline(poi_x, poi_y)
y_spline = spline(x_smooth)

# === Newton ===
newton_coef = newton_divided_diff(poi_x, poi_y)
y_newton = newton_poly(newton_coef, poi_x, x_smooth)

# === Chebyshev ===
cheb_nodes = np.cos((2*np.arange(1, len(poi_x)+1)-1)/(2*len(poi_x)) * np.pi)
cheb_nodes = 0.5 * (poi_x[-1] - poi_x[0]) * (cheb_nodes + 1) + poi_x[0]
cheb_values = np.interp(cheb_nodes, poi_x, poi_y)
cheb_fit = Chebyshev.fit(cheb_nodes, cheb_values, deg=len(cheb_nodes)-1, domain=[poi_x[0], poi_x[-1]])
y_chebyshev = cheb_fit(x_smooth)

# === Real Segment ===
real_segment_x = segment_time
real_segment_y = segment

# === Error functions ===
def compute_errors(pred, real):
    mse = np.mean((pred - real) ** 2)
    mae = np.mean(np.abs(pred - real))
    rmse = np.sqrt(mse)
    return mse, mae, rmse

# === Calculate Errors ===
errors = {
    "Lagrange": compute_errors(y_lagrange, real_segment_y),
    "Spline": compute_errors(y_spline, real_segment_y),
    "Newton": compute_errors(y_newton, real_segment_y),
    "Chebyshev": compute_errors(y_chebyshev, real_segment_y),
}

# === Plot ===
plt.figure(figsize=(14, 8))
plt.plot(real_segment_x, real_segment_y, label="Real Trace", color='black', linewidth=2)
#plt.plot(x_smooth, y_lagrange, label="Lagrange", linestyle='--')
plt.plot(x_smooth, y_spline, label="Cubic Spline", linestyle='-.')
#plt.plot(x_smooth, y_newton, label="Newton", linestyle=':')
plt.plot(x_smooth, y_chebyshev, label="Chebyshev", linestyle='-')
plt.scatter(poi_x, poi_y, color='red', label="POIs", zorder=5, s=30)
plt.xlabel("Normalized Time (s)")
plt.ylabel("Normalized Amplitude")
plt.title("Interpolations vs Real Trace")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === Print Errors ===
print("\n=== Interpolation Errors (compared to real trace) ===")
for name, (mse, mae, rmse) in errors.items():
    print(f"{name:10} | MSE: {mse:.6f} | MAE: {mae:.6f} | RMSE: {rmse:.6f}")
