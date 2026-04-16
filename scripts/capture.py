"""
capture.py — Dual-board DWM3001CDK data logger
EEM 499 FYP — Occupancy Estimation Using UWB Radar and Machine Learning
Muhammad Hafizul Bin Ahmad Husni | 161115 | USM

Usage:
    python capture.py <session_name> <label>
    python capture.py session_001 0    # empty room
    python capture.py session_011 1    # 1 person
    python capture.py session_021 2    # 2 people
    python capture.py session_031 3    # 3 people
"""

import serial
import threading
import time
import numpy as np
import h5py
import re
import sys
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
BOARD_A_PORT  = "COM6"       # Initiator board — change if different
BOARD_B_PORT  = "COM8"       # Responder board — change if different
BAUD          = 115200
SESSION_NAME  = sys.argv[1] if len(sys.argv) > 1 else "session_001"
LABEL         = int(sys.argv[2]) if len(sys.argv) > 2 else 0
DURATION_SEC  = 120          # Recording duration (seconds)
STARTUP_DELAY = 15           # Countdown before recording starts (seconds)
INIT_WAIT     = 5            # Wait for boards to initialise after commands
OUTPUT_DIR    = "data"       # Output folder for HDF5 files
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

buffers = {0: [], 1: []}
lock    = threading.Lock()
running = True


def parse_line(line):
    """
    Parse distance and RSSI from the second line of SESSION_INFO_NTF output.
    Format: [mac_address=0x0000, status="SUCCESS", distance[cm]=185, RSSI[dBm]=-62.0]}
    """
    if "distance[cm]" not in line:
        return None
    dist = re.search(r"distance\[cm\]=(\d+)", line)
    rssi = re.search(r"RSSI\[dBm\]=([-\d.]+)", line)
    if dist and rssi:
        return {
            "dist": int(dist.group(1)),
            "rssi": float(rssi.group(1))
        }
    return None


def read_board(board_idx, port, role_cmd):
    """Connect to a board, send CLI commands, and stream ranging data."""
    try:
        ser = serial.Serial(port, BAUD, timeout=1)
        print(f"  [Board {board_idx}] Connected on {port}")

        # Wake up and send configuration commands
        time.sleep(1.0)
        ser.write(b"\r\n")
        time.sleep(0.5)
        ser.write(b"DIAG 1\r\n")
        time.sleep(0.5)
        ser.write(f"{role_cmd}\r\n".encode())
        time.sleep(0.5)
        ser.reset_input_buffer()
        print(f"  [Board {board_idx}] Commands sent: DIAG 1 + {role_cmd}")

        # Two-line state machine parser
        # Line 1: SESSION_INFO_NTF: {..., sequence_number=X, ...}
        # Line 2: [mac_address=..., distance[cm]=X, RSSI[dBm]=X]}
        waiting_for_data = False
        seq_num = None

        while running:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            if "SESSION_INFO_NTF" in line:
                seq_match = re.search(r"sequence_number=(\d+)", line)
                if seq_match:
                    seq_num = int(seq_match.group(1))
                    waiting_for_data = True

            elif waiting_for_data and "distance[cm]" in line:
                result = parse_line(line)
                if result:
                    ts = time.time()
                    with lock:
                        buffers[board_idx].append((ts, seq_num, result["dist"], result["rssi"]))
                waiting_for_data = False

        ser.close()

    except serial.SerialException as e:
        print(f"\n  [Board {board_idx}] ERROR: {e}")
        print(f"  Check that {port} is correct and not in use by PuTTY or another program.")


# Start Board B (Responder) first, then Board A (Initiator)
print("\nConnecting to boards...")
t1 = threading.Thread(target=read_board, args=(1, BOARD_B_PORT, "RESPF"), daemon=True)
t0 = threading.Thread(target=read_board, args=(0, BOARD_A_PORT, "INITF"), daemon=True)
t1.start()
time.sleep(1.5)
t0.start()

# Wait for boards to initialise and confirm ranging is active
print(f"\nWaiting {INIT_WAIT}s for boards to initialise...")
for i in range(INIT_WAIT, 0, -1):
    print(f"  Initialising... {i}s", end="\r")
    time.sleep(1)

# Check boards are sending data
with lock:
    n0, n1 = len(buffers[0]), len(buffers[1])

if n0 == 0 or n1 == 0:
    print(f"\n  WARNING: Board A={n0} frames, Board B={n1} frames after init.")
    print("  Waiting 5 more seconds...")
    time.sleep(5)
    with lock:
        n0, n1 = len(buffers[0]), len(buffers[1])

    if n0 == 0 or n1 == 0:
        print("\n  ERROR: No frames received. Check:")
        print("  - Both boards are powered and connected via J20 USB port")
        print("  - COM port numbers are correct (check Device Manager)")
        print("  - PuTTY is closed (COM port cannot be shared)")
        running = False
        sys.exit(1)

print(f"\n  Boards confirmed ranging — Board A: {n0}, Board B: {n1} frames")

# Clear buffer to discard frames from initialisation period
with lock:
    buffers[0].clear()
    buffers[1].clear()

# Countdown before recording
label_desc = {0: "Empty room — leave the room now!", 1: "1 person — enter and get settled", 2: "2 people — enter and get settled", 3: "3 people — enter and get settled"}
desc = label_desc.get(LABEL, f"{LABEL} people — get ready")
print(f"\nLabel {LABEL}: {desc}")
print(f"Recording starts in {STARTUP_DELAY} seconds...")

for i in range(STARTUP_DELAY, 0, -1):
    print(f"  {i:2d}s remaining...", end="\r")
    time.sleep(1)

# Clear buffer again to discard countdown frames
with lock:
    buffers[0].clear()
    buffers[1].clear()

print("\n" + "=" * 50)
print(f"  RECORDING STARTED")
print(f"  Session: {SESSION_NAME} | Label: {LABEL} | Duration: {DURATION_SEC}s")
print("=" * 50)

# Record
for i in range(DURATION_SEC):
    time.sleep(1)
    with lock:
        n0, n1 = len(buffers[0]), len(buffers[1])
    bar = "█" * int(30 * (i+1) / DURATION_SEC) + "░" * (30 - int(30 * (i+1) / DURATION_SEC))
    print(f"  [{bar}] t={i+1:03d}s  A:{n0} frames  B:{n1} frames", end="\r")

running = False
time.sleep(0.5)
print("\n")

# Check we have data
with lock:
    n0, n1 = len(buffers[0]), len(buffers[1])

if n0 == 0 or n1 == 0:
    print(f"  ERROR: No frames collected (Board A={n0}, Board B={n1}). File NOT saved.")
    sys.exit(1)

# Save to HDF5
fname = os.path.join(OUTPUT_DIR, f"{SESSION_NAME}_label{LABEL}.h5")
with h5py.File(fname, "w") as f:
    f.attrs["label"]    = LABEL
    f.attrs["session"]  = SESSION_NAME
    f.attrs["duration"] = DURATION_SEC

    for idx in [0, 1]:
        data     = buffers[idx]
        ts_arr   = np.array([x[0] for x in data])
        seq_arr  = np.array([x[1] for x in data])
        dist_arr = np.array([x[2] for x in data])
        rssi_arr = np.array([x[3] for x in data])

        grp = f.create_group(f"board_{idx}")
        grp.create_dataset("timestamps",  data=ts_arr)
        grp.create_dataset("seq_number",  data=seq_arr)
        grp.create_dataset("distance_cm", data=dist_arr)
        grp.create_dataset("rssi_dbm",    data=rssi_arr)

# Print summary
print(f"  Saved: {fname}")
with h5py.File(fname, "r") as f:
    for idx in [0, 1]:
        d = f[f"board_{idx}/distance_cm"][:]
        r = f[f"board_{idx}/rssi_dbm"][:]
        t = f[f"board_{idx}/timestamps"][:]
        print(f"  Board {idx}: {len(d):4d} frames | "
              f"dist {d.min():.0f}–{d.max():.0f} cm | "
              f"mean RSSI {r.mean():.1f} dBm | "
              f"duration {t[-1]-t[0]:.1f}s")
print(f"\n  Done! Next: python capture.py {SESSION_NAME[:-3]}{int(SESSION_NAME[-3:])+1:03d} {LABEL}")
