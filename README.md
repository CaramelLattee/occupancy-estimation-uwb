# Occupancy Estimation Using Ultra-Wide Band (UWB) Radar and Machine Learning

> **EEM 499 Final Year Project**  
> Muhammad Hafizul Bin Ahmad Husni | Matrix No.: 161115  
> Mechatronic Engineering, School of Electrical and Electronic Engineering  
> Universiti Sains Malaysia (USM)  
> Supervisor: Dr. Suhardi Azliy Junoh  
> Examiner: Assoc. Prof. Dr. Dzati Athiar Ramli

---

## Overview

This project develops a **non-invasive, privacy-preserving indoor occupancy estimation system** using two **Qorvo DWM3001CDK** Ultra-Wideband (UWB) radar development boards in a dual-node bistatic configuration, combined with machine learning classifiers.

The system estimates the number of occupants (0–3 persons) in a room by processing UWB ranging signals — without requiring cameras, wearable devices, or any contact with occupants.

A key research contribution is the **first empirical comparison of DIAG output vs raw CIR data** from the DWM3001CDK platform for multi-class occupancy estimation.

---

## Hardware

| Component | Details |
|-----------|---------|
| UWB Board | Qorvo DWM3001CDK × 2 |
| UWB Chip | DW3110 (IEEE 802.15.4z HRP UWB) |
| Host MCU | Nordic nRF52833 (ARM Cortex-M4F, 64 MHz) |
| Channel | Channel 5 — 6.5 GHz, 499.2 MHz bandwidth |
| Board A | COM6 — Initiator (INITF) |
| Board B | COM8 — Responder (RESPF) |
| Separation | ~185 cm, facing each other |
| Frame rate | ~5.3 frames per second per board |
| Baud rate | 115200 bps (J20 USB port) |

### Setup Diagram

```
┌─────────────────────────────────────────────────────────┐
│                  Laboratory Room (~5m × 6m)             │
│                                                         │
│  [Board A]  ◄────── UWB TWR (~185 cm) ──────► [Board B]│
│  Initiator                                   Responder  │
│  COM6                                         COM8      │
│     │                                           │       │
│     └──── USB-UART ──── [Host PC] ──── USB-UART ┘       │
│                      capture.py                         │
│                                                         │
│          P1 ●       P2 ●       P3 ●                     │
│              (Participants in various positions)         │
└─────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
occupancy-estimation-uwb/
│
├── data/                          # HDF5 dataset files
│   ├── session_001_label0.h5      # Label 0: Empty room
│   ├── session_011_label1.h5      # Label 1: 1 person
│   ├── session_021_label2.h5      # Label 2: 2 people
│   └── ...
│
├── scripts/                       # Data acquisition scripts
│   ├── capture.py                 # Main dual-board data logger
│   ├── verify.py                  # Single session HDF5 verifier
│   └── view_data.py               # Multi-session summary and box plot viewer
│
├── ml/                            # Machine learning pipeline (coming soon)
│   ├── feature_extraction.py      # DIAG sliding-window feature extractor
│   ├── train_baseline.py          # Random Forest and SVM baseline training
│   ├── train_cnn.py               # CNN training on range-Doppler images
│   └── evaluate.py                # Model evaluation and comparison
│
├── figures/                       # Report figures (PNG)
│   ├── figure_3_1_system_block_diagram.png
│   ├── figure_3_2_data_collection_flowchart.png
│   ├── figure_3_3_ml_pipeline.png
│   └── figure_3_4_testbed_environment.png
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/occupancy-estimation-uwb.git
cd occupancy-estimation-uwb
```

### 2. Create and activate virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Connect both DWM3001CDK boards

- Plug both boards via **J20 USB port** to your PC
- Check COM ports in Device Manager (Windows) or `/dev/tty*` (Linux)
- Update `BOARD_A_PORT` and `BOARD_B_PORT` in `capture.py`

### 5. Run a data collection session

```bash
# Label 0 = empty room, Label 1 = 1 person, etc.
python scripts/capture.py session_001 0

# Arguments: <session_name> <label>
python scripts/capture.py session_011 1
python scripts/capture.py session_021 2
python scripts/capture.py session_031 3
```

### 6. Verify saved data

```bash
python scripts/verify.py data/session_001_label0.h5
```

### 7. View dataset overview

```bash
python scripts/view_data.py
```

---

## Data Collection Scripts

### `capture.py` — Dual-board data logger

Automatically:
- Connects to both boards via pyserial
- Sends `DIAG 1`, `RESPF`, and `INITF` commands
- Waits 5 seconds for boards to initialise (verifies frames > 0)
- Counts down 15 seconds (get into position / leave room)
- Records for 120 seconds (~630 frames per board)
- Saves to HDF5 format

```bash
python scripts/capture.py <session_name> <label>

# Examples:
python scripts/capture.py session_001 0   # Empty room
python scripts/capture.py session_011 1   # 1 person
python scripts/capture.py session_021 2   # 2 people
python scripts/capture.py session_031 3   # 3 people
```

**Configuration** (edit at top of `capture.py`):

```python
BOARD_A_PORT  = "COM6"     # Initiator board port
BOARD_B_PORT  = "COM8"     # Responder board port
BAUD          = 115200
DURATION_SEC  = 120        # Recording duration in seconds
STARTUP_DELAY = 15         # Countdown before recording
INIT_WAIT     = 5          # Board initialisation wait
```

### `verify.py` — Session verifier

```bash
python scripts/verify.py data/session_001_label0.h5
```

Output example:
```
Label:    0 person(s)
Session:  session_001
Duration: 120s

Board 0:
  Frames collected : 631
  Distance range   : 179 – 192 cm
  Mean distance    : 185.7 cm
  Mean RSSI        : -61.4 dBm
  Duration logged  : 119.4 s
```

### `view_data.py` — Dataset overview

```bash
python scripts/view_data.py
```

Generates:
- Summary table of all sessions (file name, label, frame count, mean distance, mean RSSI)
- Box plot of distance and RSSI distributions per occupancy label
- Saves `data_overview.png`

---

## HDF5 Data Format

Each `.h5` session file contains:

```
session_001_label0.h5
├── attrs/
│   ├── label       → int (0, 1, 2, or 3)
│   ├── session     → str (e.g. "session_001")
│   └── duration    → int (120 seconds)
├── board_0/
│   ├── distance_cm  → float array [N]  — distance per frame (cm)
│   ├── rssi_dbm     → float array [N]  — RSSI per frame (dBm)
│   ├── timestamps   → float array [N]  — POSIX timestamps
│   └── seq_number   → int array [N]    — sequence numbers
└── board_1/
    ├── distance_cm
    ├── rssi_dbm
    ├── timestamps
    └── seq_number
```

Read with Python:
```python
import h5py
import numpy as np

with h5py.File("data/session_001_label0.h5", "r") as f:
    label = f.attrs["label"]
    dist  = f["board_0/distance_cm"][:]   # shape (N,)
    rssi  = f["board_0/rssi_dbm"][:]      # shape (N,)
    ts    = f["board_0/timestamps"][:]    # shape (N,)
    print(f"Label: {label}, Frames: {len(dist)}")
```

---

## Dataset Summary

| Label | Description | Sessions | Frames/Board | Status |
|-------|-------------|----------|-------------|--------|
| 0 | Empty room | 10 | ~6,300 | ✅ Complete |
| 1 | 1 person | 10 | ~6,300 | ✅ Complete |
| 2 | 2 people | 10 | ~6,300 | ✅ Complete |
| 3 | 3 people | 10 | ~6,300 | ⏳ Pending |
| **Total** | | **30–40** | **~25,000** | **75% Complete** |

### Preliminary Analysis (Labels 0–2)

| Metric | Label 0 (Empty) | Label 1 (1 Person) | Label 2 (2 People) |
|--------|----------------|-------------------|-------------------|
| Median distance | ~184 cm | ~185 cm | ~177 cm |
| Distance max outlier | ~193 cm | ~335 cm | ~310 cm |
| RSSI median | ~-62 dBm | ~-64 dBm | ~-65 dBm |
| RSSI IQR | ±2 dBm | ±17 dBm | ±8 dBm |

The **progressive RSSI drop** and **increasing distance variance** with each additional occupant provide strong discriminative features for ML classification.

---

## Machine Learning Approach

### Approach A — DIAG Baseline

Features extracted from a sliding window of W = 64 frames (50% overlap):

| Feature | Per Board | Both Boards |
|---------|-----------|-------------|
| Mean distance, mean RSSI | 2 | 4 |
| Std deviation (distance, RSSI) | 2 | 4 |
| Variance (distance, RSSI) | 2 | 4 |
| Distance range (max − min) | 1 | 2 |
| Rate of change | 1 | 2 |
| Outlier rate (dist > 200 cm) | 1 | 2 |
| RSSI minimum | 1 | 2 |
| Skewness + kurtosis | 2 | 4 |
| **Total** | **20** | **40** |

Classifiers: **Random Forest** (300 trees) and **SVM** (RBF kernel)

Expected accuracy: **78–88%** macro F1-score

### Approach B — CIR Enhanced (Planned)

- Firmware modification to call `dwt_readaccdata()` on Board B
- 256 CIR taps per frame at 921600 baud
- 2D-FFT → 64×64 range-Doppler images → CNN input
- Classifier: **CNN** (3 conv blocks + GAP + dense + softmax)

Expected accuracy: **88–96%** macro F1-score

### Evaluation Protocol

- Session-stratified 70 / 15 / 15 split (train / val / test)
- Primary metric: **Macro F1-score**
- Secondary: per-class precision, recall, confusion matrix
- Statistical test: McNemar test (p < 0.05)

---

## Requirements

```txt
pyserial>=3.5
numpy>=1.24
h5py>=3.8
matplotlib>=3.7
scikit-learn>=1.3
tensorflow>=2.14
scipy>=1.11
```

Install all:
```bash
pip install -r requirements.txt
```

---

## Board Setup (Step by Step)

### 1. Flash firmware

Use **J-Flash Lite** (included with SEGGER J-Link):
- Connect board via **J9 port** (lower USB)
- Device: `NRF52833_XXAA`
- Interface: `SWD` at `4000 kHz`
- File: `DW3_QM33_SDK\Binaries\DWM3001CDK-DW3_QM33_SDK_CLI-FreeRTOS.hex`
- Click **Program Device**
- Repeat for both boards

### 2. Configure via PuTTY (optional manual test)

Connect via **J20 port** (upper USB), baud rate `115200`:

```
# On Board B (Responder) first:
DIAG 1
RESPF

# On Board A (Initiator):
DIAG 1
INITF
```

You should see ranging output:
```
SESSION_INFO_NTF: {session_handle=1, sequence_number=1, ...
  [mac_address=0x0000, status="SUCCESS", distance[cm]=185, RSSI[dBm]=-62.0]}
```

### 3. Run capture.py

Close PuTTY first (COM port cannot be shared), then run:
```bash
python scripts/capture.py session_001 0
```

---

## Project Timeline

| Milestone | Target | Status |
|-----------|--------|--------|
| Hardware setup and characterisation | Oct 2025 | ✅ Done |
| Python data acquisition pipeline | Dec 2025 | ✅ Done |
| Literature review | Dec 2025 | ✅ Done |
| Data collection — Label 0 | Apr 2026 | ✅ Done |
| Data collection — Label 1 | Apr 2026 | ✅ Done |
| Data collection — Label 2 | Apr 2026 | ✅ Done |
| Data collection — Label 3 | May 2026 | ⏳ Pending |
| DIAG ML baseline training | May 2026 | ⏳ Pending |
| CIR firmware modification | May 2026 | ⏳ Pending |
| CIR ML model training | Jun 2026 | ⏳ Pending |
| Final report submission | Jun 2026 | ⏳ Pending |
| FYP presentation and viva | Jul 2026 | ⏳ Pending |

---

## References

1. Antolinos, E. et al. (2021). Occupancy estimation via UWB radar using machine learning. *IEEE Transactions on Instrumentation and Measurement*, 70, 1–10.
2. Huo, Z. et al. (2021). Multi-anchor UWB radar for human counting and localization. *IEEE Sensors Journal*, 21(12), 13566–13575.
3. Zhang, Z. et al. (2021). Human activity recognition based on motion sensor using u-net. *IEEE Access*, 7, 75478–75488.
4. Shrestha, A. et al. (2020). Continuous human activity classification from FMCW radar with Bi-LSTM networks. *IEEE Sensors Journal*, 20(22), 13607–13619.
5. Tekler, Z. D. et al. (2020). A scalable BLE approach to identify occupancy patterns. *Building and Environment*, 171, 106681.
6. Qorvo Inc. (2023). DWM3001CDK Development Kit. https://www.qorvo.com/products/p/DWM3001CDK

---

## License

This project is developed for academic purposes under EEM 499, Universiti Sains Malaysia.  
© 2026 Muhammad Hafizul Bin Ahmad Husni. All rights reserved.

---

## Contact

**Muhammad Hafizul Bin Ahmad Husni**  
School of Electrical and Electronic Engineering  
Universiti Sains Malaysia, Engineering Campus  
14300 Nibong Tebal, Pulau Pinang, Malaysia

Supervisor: Dr. Suhardi Azliy Junoh | Universiti Sains Malaysia
