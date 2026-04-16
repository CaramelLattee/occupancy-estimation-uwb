"""
view_data.py — Dataset overview and box plot generator
EEM 499 FYP — Occupancy Estimation Using UWB Radar and Machine Learning
Muhammad Hafizul Bin Ahmad Husni | 161115 | USM

Usage:
    python view_data.py [data_folder]
    python view_data.py             # defaults to ./data
    python view_data.py D:/test_1   # custom folder
"""

import h5py
import numpy as np
import os
import glob
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_FOLDER = sys.argv[1] if len(sys.argv) > 1 else "data"
LABEL_NAMES = {0: "Empty", 1: "1 Person", 2: "2 People", 3: "3 People"}
COLORS      = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
# ─────────────────────────────────────────────────────────────────────────────

# Find all HDF5 files
files = sorted(glob.glob(os.path.join(DATA_FOLDER, "*.h5")))

if not files:
    print(f"No .h5 files found in '{DATA_FOLDER}'")
    print(f"Make sure you have run capture.py and saved some sessions.")
    sys.exit(1)

print(f"\nFound {len(files)} session file(s) in '{DATA_FOLDER}'\n")

# Print session table
print(f"{'File':<35} {'Label':>6} {'B0 Frames':>10} {'B1 Frames':>10} {'Mean Dist(cm)':>14} {'Mean RSSI(dBm)':>15}")
print("-" * 95)

all_data = []

for fpath in files:
    fname = os.path.basename(fpath)
    try:
        with h5py.File(fpath, "r") as f:
            label   = int(f.attrs.get("label", -1))
            b0_dist = f["board_0/distance_cm"][:]
            b0_rssi = f["board_0/rssi_dbm"][:]
            b1_dist = f["board_1/distance_cm"][:]
            b1_rssi = f["board_1/rssi_dbm"][:]

            mean_dist = (b0_dist.mean() + b1_dist.mean()) / 2
            mean_rssi = (b0_rssi.mean() + b1_rssi.mean()) / 2

            print(f"{fname:<35} {label:>6} {len(b0_dist):>10} {len(b1_dist):>10} "
                  f"{mean_dist:>14.1f} {mean_rssi:>15.1f}")

            all_data.append({
                "label":   label,
                "b0_dist": b0_dist,
                "b0_rssi": b0_rssi,
                "b1_dist": b1_dist,
                "b1_rssi": b1_rssi,
                "fname":   fname,
            })
    except Exception as e:
        print(f"{fname:<35} ERROR: {e}")

# Summary by label
print(f"\n{'─'*65}")
print("Summary by Label:")
print(f"{'─'*65}")
print(f"{'Label':<10} {'Description':<12} {'Sessions':>9} {'Total Frames':>13} {'Mean Dist':>10} {'Mean RSSI':>10}")
print("-" * 65)

labels_present = sorted(set(d["label"] for d in all_data))

for lbl in labels_present:
    subset    = [d for d in all_data if d["label"] == lbl]
    total_frm = sum(len(d["b0_dist"]) for d in subset)
    mean_dist = np.mean([d["b0_dist"].mean() for d in subset])
    mean_rssi = np.mean([d["b0_rssi"].mean() for d in subset])
    desc      = LABEL_NAMES.get(lbl, f"Label {lbl}")
    print(f"{lbl:<10} {desc:<12} {len(subset):>9} {total_frm:>13} {mean_dist:>10.1f} {mean_rssi:>10.1f}")

print(f"{'─'*65}")
total_sessions = len(all_data)
total_frames   = sum(len(d["b0_dist"]) for d in all_data)
print(f"{'TOTAL':<10} {'':<12} {total_sessions:>9} {total_frames:>13}")

# ── BOX PLOTS ─────────────────────────────────────────────────────────────────
if len(labels_present) == 0:
    print("\nNo data to plot.")
    sys.exit(0)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("DWM3001CDK — Occupancy Data Overview", fontsize=14, fontweight="bold")

box_labels = [LABEL_NAMES.get(lbl, f"Label {lbl}") for lbl in labels_present]
colors_used = [COLORS[lbl % len(COLORS)] for lbl in labels_present]

# Plot 1: Distance distribution
ax1 = axes[0]
box_dist = [np.concatenate([d["b0_dist"] for d in all_data if d["label"]==lbl])
            for lbl in labels_present]
bp1 = ax1.boxplot(box_dist, patch_artist=True, labels=box_labels, notch=False)
for patch, color in zip(bp1["boxes"], colors_used):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax1.set_title("Distance Distribution per Label")
ax1.set_ylabel("Distance (cm)")
ax1.set_xlabel("Occupancy")
ax1.grid(axis="y", alpha=0.3)

# Plot 2: RSSI distribution
ax2 = axes[1]
box_rssi = [np.concatenate([d["b0_rssi"] for d in all_data if d["label"]==lbl])
            for lbl in labels_present]
bp2 = ax2.boxplot(box_rssi, patch_artist=True, labels=box_labels, notch=False)
for patch, color in zip(bp2["boxes"], colors_used):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_title("RSSI Distribution per Label")
ax2.set_ylabel("RSSI (dBm)")
ax2.set_xlabel("Occupancy")
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
out_path = os.path.join(DATA_FOLDER, "data_overview.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nPlot saved: {out_path}")
