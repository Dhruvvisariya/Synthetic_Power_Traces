import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr

# === AES SBOX ===
SBOX = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b,
    0xfe, 0xd7, 0xab, 0x76, 0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
    0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0, 0xb7, 0xfd, 0x93, 0x26,
    0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2,
    0xeb, 0x27, 0xb2, 0x75, 0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
    0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84, 0x53, 0xd1, 0x00, 0xed,
    0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f,
    0x50, 0x3c, 0x9f, 0xa8, 0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
    0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2, 0xcd, 0x0c, 0x13, 0xec,
    0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14,
    0xde, 0x5e, 0x0b, 0xdb, 0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
    0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79, 0xe7, 0xc8, 0x37, 0x6d,
    0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f,
    0x4b, 0xbd, 0x8b, 0x8a, 0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
    0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e, 0xe1, 0xf8, 0x98, 0x11,
    0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f,
    0xb0, 0x54, 0xbb, 0x16
]

def sbox(byte): return SBOX[byte]
def hamming_weight(byte): return bin(byte).count("1")
def hamming_distance(a, b): return bin(a ^ b).count("1")

def compute_subbytes_hd_hw(msg_hex, key_hex):
    msg_bytes = bytes.fromhex(msg_hex)
    key_bytes = bytes.fromhex(key_hex)
    xor_bytes = bytes([m ^ k for m, k in zip(msg_bytes, key_bytes)])
    sbox_outputs = [sbox(b) for b in xor_bytes]
    hd = sum(hamming_distance(a, b) for a, b in zip(xor_bytes, sbox_outputs))
    hw = sum(hamming_weight(b) for b in sbox_outputs)
    return hd, hw

# === POI EXTRACTOR ===
def extract_trace_segment(file_path, x_increment, x_start_1e7, x_end_1e7):
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
        trace = np.array([float(l.strip()) for l in lines[6:] if l.strip() and not l.startswith("#")])
        x_start = int((x_start_1e7 * 1e-7) / x_increment)
        x_end = int((x_end_1e7 * 1e-7) / x_increment)
        segment = trace[x_start:x_end]
        return segment.tolist()
    except Exception as e:
        print(f"⚠️ {file_path}: {e}")
        return []

# === CONFIGURATION ===
folder_path = "D:/Projects/DPA_contest2_public_base_diff_vcc_a128_2009_12_23"
key = "0000000000000003243f6a8885a308d3"
x_increment = 2e-10
x_start_1e7 = 0.773
x_end_1e7 = 0.897
max_traces = 200

# === DATA COLLECTION ===
records = []
for fname in os.listdir(folder_path)[:max_traces]:
    if "_k=" not in fname or "_m=" not in fname: continue
    k_hex = fname.split("_k=")[1].split("_")[0]
    m_hex = fname.split("_m=")[1].split("_")[0]
    if k_hex != key: continue
    full_path = os.path.join(folder_path, fname)
    pois = extract_trace_segment(full_path, x_increment, x_start_1e7, x_end_1e7)
    if pois:
        hd, hw = compute_subbytes_hd_hw(m_hex, k_hex)
        records.append({"HD": hd, "HW": hw, "POI": pois, "File": fname})

df = pd.DataFrame(records)

# === FILTER: keep only groups with 2 to 4 traces ===
hd_groups = df.groupby("HD").filter(lambda x: 2 <= len(x) <= 4)
hw_groups = df.groupby("HW").filter(lambda x: 2 <= len(x) <= 4)

# === INTRA-GROUP SIMILARITY CHECK ===

def print_poi_stats_and_intra_corr(df_grouped, group_col):
    groups = df_grouped.groupby(group_col)
    
    for group_val, group_df in groups:
        print(f"\n=== Group {group_col} = {group_val} ===")
        pois_list = group_df["POI"].tolist()
        
        n_traces = len(pois_list)
        print(f"Number of traces: {n_traces}")
        if n_traces < 2:
            print("Not enough traces for correlation.")
            continue
        
        poi_lengths = [len(poi) for poi in pois_list]
        print(f"POI segment lengths: min={min(poi_lengths)}, max={max(poi_lengths)}")
        if max(poi_lengths) != min(poi_lengths):
            print("Warning: POI lengths vary within the group, alignment may be off.")
        
        corrs = []
        for i in range(n_traces):
            for j in range(i+1, n_traces):
                corr, _ = pearsonr(pois_list[i], pois_list[j])
                corrs.append(corr)
        
        if corrs:
            mean_corr = np.mean(corrs)
            std_corr = np.std(corrs)
            print(f"Mean intra-group Pearson correlation: {mean_corr:.4f} ± {std_corr:.4f}")
        else:
            print("Not enough pairs for correlation.")
        
        # Plot first 3 traces for visual check
        plt.figure(figsize=(10,4))
        for idx, trace in enumerate(pois_list[:3]):
            plt.plot(trace, label=f"Trace {idx+1}")
        plt.title(f"Group {group_col}={group_val}: Sample POI Traces")
        plt.xlabel("POI Index")
        plt.ylabel("Power")
        plt.legend()
        plt.grid(True)
        plt.show()

print("=== Intra-group similarity for HD groups ===")
print_poi_stats_and_intra_corr(hd_groups, "HD")

print("=== Intra-group similarity for HW groups ===")
print_poi_stats_and_intra_corr(hw_groups, "HW")
#ABOVE WORKS WELL TO PROVE INTR GRP SIMILARITIES


#NEW APPEND
from itertools import combinations

def cohen_d(x, y):
    """
    Compute Cohen's d vector between two sets of POIs (2D arrays: n_samples x POI_length).
    Returns an array of effect sizes per POI.
    """
    nx, ny = x.shape[0], y.shape[0]
    dof = nx + ny - 2
    
    mean_x = np.mean(x, axis=0)
    mean_y = np.mean(y, axis=0)
    std_x = np.std(x, axis=0, ddof=1)
    std_y = np.std(y, axis=0, ddof=1)
    
    pooled_std = np.sqrt(((nx-1)*std_x**2 + (ny-1)*std_y**2) / dof)
    # Avoid division by zero
    pooled_std[pooled_std == 0] = 1e-10
    
    d = (mean_x - mean_y) / pooled_std
    return d

def compute_all_cohen_d(df, group_col):
    groups = df.groupby(group_col)
    group_keys = list(groups.groups.keys())
    results = {}
    
    for g1, g2 in combinations(group_keys, 2):
        data1 = np.vstack(groups.get_group(g1)["POI"].values)
        data2 = np.vstack(groups.get_group(g2)["POI"].values)
        d_vec = cohen_d(data1, data2)
        results[(g1, g2)] = d_vec
    return results

def has_significant_diff(d_values, threshold=0.8, min_consecutive=5):
    """
    Check if |d| > threshold for at least min_consecutive consecutive POIs.
    """
    above_thresh = np.abs(d_values) > threshold
    max_run = 0
    current_run = 0
    
    for val in above_thresh:
        if val:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    
    return max_run >= min_consecutive

def summarize_all_diffs(cohen_d_results, threshold=0.8, min_consecutive=5):
    conclusions = {}
    for pair, d_vec in cohen_d_results.items():
        conclusions[pair] = has_significant_diff(d_vec, threshold, min_consecutive)
    return conclusions

def print_summary(conclusions, group_name):
    print(f"\n=== Summary of inter-group differences for {group_name} ===")
    total_pairs = len(conclusions)
    significant_pairs = sum(conclusions.values())
    print(f"Total pairs compared: {total_pairs}")
    print(f"Pairs with significant difference (|d| > 0.8 over {5} consecutive POIs): {significant_pairs}")
    
    if significant_pairs == 0:
        print("No strong differences detected between any groups.")
    elif significant_pairs == total_pairs:
        print("All group pairs show strong differences in power traces.")
    else:
        print("Some pairs show strong differences, others do not.")

# --- Run on your filtered groups ---
hd_diffs = compute_all_cohen_d(hd_groups, "HD")
hd_conclusions = summarize_all_diffs(hd_diffs)
print_summary(hd_conclusions, "HD groups")

hw_diffs = compute_all_cohen_d(hw_groups, "HW")
hw_conclusions = summarize_all_diffs(hw_diffs)
print_summary(hw_conclusions, "HW groups")

#NEW APPEND
# Print detailed stats before summary, for each pair
def print_detailed_cohen_d(cohen_d_results, threshold=0.8):
    for (g1, g2), d_vec in cohen_d_results.items():
        max_abs_d = np.max(np.abs(d_vec))
        mean_abs_d = np.mean(np.abs(d_vec))
        num_significant = np.sum(np.abs(d_vec) > threshold)
        
        print(f"\nPair {g1} vs {g2}:")
        print(f" Max |d| = {max_abs_d:.3f}")
        print(f" Mean |d| = {mean_abs_d:.3f}")
        print(f" Number of POIs with |d| > {threshold} = {num_significant}")
        print(f" Sample d values (first 10): {d_vec[:10]}")

# Compute intra-group mean correlations and standard deviations
def intra_group_corr_stats(df, group_col):
    groups = df.groupby(group_col)
    stats = {}
    for group_val, group_df in groups:
        pois_list = group_df["POI"].tolist()
        n_traces = len(pois_list)
        if n_traces < 2:
            stats[group_val] = (np.nan, np.nan)
            continue
        
        corrs = []
        for i in range(n_traces):
            for j in range(i+1, n_traces):
                corr, _ = pearsonr(pois_list[i], pois_list[j])
                corrs.append(corr)
        stats[group_val] = (np.mean(corrs), np.std(corrs))
    return stats

def print_intra_group_stats(stats, group_name):
    print(f"\n=== Intra-group correlation stats for {group_name} ===")
    for group_val, (mean_corr, std_corr) in stats.items():
        if np.isnan(mean_corr):
            print(f"Group {group_val}: Not enough traces for correlation")
        else:
            print(f"Group {group_val}: Mean intra-group Pearson r = {mean_corr:.3f} ± {std_corr:.3f}")

# Run and print detailed Cohen's d for HD groups
print("\n=== Detailed Cohen's d for HD groups ===")
print_detailed_cohen_d(hd_diffs)

# Run and print detailed Cohen's d for HW groups
print("\n=== Detailed Cohen's d for HW groups ===")
print_detailed_cohen_d(hw_diffs)

# Compute and print intra-group stats
hd_intra_stats = intra_group_corr_stats(hd_groups, "HD")
print_intra_group_stats(hd_intra_stats, "HD")

hw_intra_stats = intra_group_corr_stats(hw_groups, "HW")
print_intra_group_stats(hw_intra_stats, "HW")

# Then print your summary conclusions again
print_summary(hd_conclusions, "HD groups")
print_summary(hw_conclusions, "HW groups")

#WORKS FINE SIMILARITY AND DIFFERENCE ARE CLEAR

#VIZUALS
import matplotlib.pyplot as plt
import numpy as np

# === FIXED: INTRA-GROUP BAR PLOT ===
def plot_intra_group_correlation_bar(stats_dict, title):
    groups = list(stats_dict.keys())
    means = [stats_dict[g][0] for g in groups]
    stds = [stats_dict[g][1] for g in groups]

    x = np.arange(len(groups))

    plt.figure(figsize=(10, 5))
    bars = plt.bar(x, means, yerr=stds, capsize=5, color='skyblue', edgecolor='black')
    avg_line = np.mean(means)
    plt.axhline(avg_line, color='red', linestyle='--', label=f"Avg = {avg_line:.2f}")

    plt.xticks(x, groups)
    plt.ylabel("Mean Intra-group Correlation")
    plt.xlabel("Group")
    plt.title(title)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

# === FIXED: INTER-GROUP COHEN'S D LINE PLOT ===
def plot_cohen_d_vectors(d_vector_dict, title, threshold=0.8):
    plt.figure(figsize=(12, 6))

    for (g1, g2), d_vec in d_vector_dict.items():
        label = f"{g1} vs {g2}"
        plt.plot(range(len(d_vec)), d_vec, label=label, alpha=0.7)

    plt.axhline(threshold, color='red', linestyle='--', linewidth=1)
    plt.axhline(-threshold, color='red', linestyle='--', linewidth=1)
    plt.title(title)
    plt.xlabel("POI Index")
    plt.ylabel("Cohen's d")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize='small', ncol=2)
    plt.tight_layout()
    plt.show()


# === PLOT ALL FOUR ===
plot_intra_group_correlation_bar(hd_intra_stats, "Intra-group Similarity (HD): Pearson Correlation")
plot_cohen_d_vectors(hd_diffs, "Inter-group Differences (HD): Cohen's d per POI")

plot_intra_group_correlation_bar(hw_intra_stats, "Intra-group Similarity (HW): Pearson Correlation")
plot_cohen_d_vectors(hw_diffs, "Inter-group Differences (HW): Cohen's d per POI")

