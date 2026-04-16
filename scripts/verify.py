"""
verify.py — HDF5 session file verifier
EEM 499 FYP — Occupancy Estimation Using UWB Radar and Machine Learning
Muhammad Hafizul Bin Ahmad Husni | 161115 | USM

Usage:
    python verify.py <path_to_h5_file>
    python verify.py data/session_001_label0.h5
"""

import h5py
import numpy as np
import sys
import os

def verify(fname):
    if not os.path.exists(fname):
        print(f"File not found: {fname}")
        return

    size_kb = os.path.getsize(fname) / 1024
    print(f"\n{'='*55}")
    print(f"  File:     {os.path.basename(fname)}")
    print(f"  Size:     {size_kb:.1f} KB")
    print(f"{'='*55}")

    with h5py.File(fname, "r") as f:
        label    = f.attrs.get("label", "N/A")
        session  = f.attrs.get("session", "N/A")
        duration = f.attrs.get("duration", "N/A")

        label_desc = {0:"Empty room", 1:"1 person", 2:"2 people", 3:"3 people"}
        print(f"  Label:    {label} — {label_desc.get(int(label), 'Unknown')}")
        print(f"  Session:  {session}")
        print(f"  Duration: {duration}s")
        print()

        for idx in [0, 1]:
            key = f"board_{idx}"
            if key not in f:
                print(f"  Board {idx}: NOT FOUND in file")
                continue

            dist = f[f"{key}/distance_cm"][:]
            rssi = f[f"{key}/rssi_dbm"][:]
            ts   = f[f"{key}/timestamps"][:]
            seq  = f[f"{key}/seq_number"][:]

            duration_actual = ts[-1] - ts[0] if len(ts) > 1 else 0
            fps = len(dist) / duration_actual if duration_actual > 0 else 0
            outlier_rate = np.sum(dist > 200) / len(dist) * 100

            print(f"  Board {idx} ({'Initiator/COM6' if idx == 0 else 'Responder/COM8'}):")
            print(f"    Frames collected : {len(dist)}")
            print(f"    Frame rate       : {fps:.1f} fps")
            print(f"    Distance range   : {dist.min():.0f} – {dist.max():.0f} cm")
            print(f"    Mean distance    : {dist.mean():.1f} cm")
            print(f"    Dist std dev     : {dist.std():.1f} cm")
            print(f"    Outlier rate     : {outlier_rate:.1f}% (dist > 200 cm)")
            print(f"    Mean RSSI        : {rssi.mean():.1f} dBm")
            print(f"    RSSI range       : {rssi.min():.1f} – {rssi.max():.1f} dBm")
            print(f"    Duration logged  : {duration_actual:.1f}s")
            print()

        # Quality check
        d0 = f["board_0/distance_cm"][:]
        d1 = f["board_1/distance_cm"][:]
        r0 = f["board_0/rssi_dbm"][:]
        r1 = f["board_1/rssi_dbm"][:]

        print("  Quality Check:")
        checks = [
            ("Frame count Board A", len(d0) >= 400, f"{len(d0)} frames"),
            ("Frame count Board B", len(d1) >= 400, f"{len(d1)} frames"),
            ("Distance reasonable", 100 <= np.median(d0) <= 600, f"median {np.median(d0):.0f} cm"),
            ("RSSI reasonable",     -90 <= r0.mean() <= -40,     f"mean {r0.mean():.1f} dBm"),
            ("Both boards agree",   abs(np.median(d0)-np.median(d1)) < 20, f"diff {abs(np.median(d0)-np.median(d1)):.0f} cm"),
        ]
        all_pass = True
        for name, passed, detail in checks:
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
            print(f"    [{status}] {name}: {detail}")

        print()
        if all_pass:
            print(f"  Overall: GOOD — session is valid")
        else:
            print(f"  Overall: WARNING — check failed items above")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify.py <path_to_h5_file>")
        print("Example: python verify.py data/session_001_label0.h5")
        sys.exit(1)
    verify(sys.argv[1])
