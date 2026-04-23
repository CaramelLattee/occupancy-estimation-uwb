"""
train_baseline.py — DIAG Baseline ML Training
EEM 499 FYP — Occupancy Estimation Using UWB Radar and Machine Learning
Muhammad Hafizul Bin Ahmad Husni | 161115 | USM

What this script does:
─────────────────────────────────────────────────────────────────────
1. Loads all HDF5 sessions from data/ folder
2. Applies sliding window (W=64, 50% overlap) to extract 40 features
   per window from both boards (20 features × 2 boards)
3. Splits by SESSION (not by frame) to prevent data leakage
   → 70% train / 15% validation / 15% test
4. Trains Random Forest and SVM classifiers
5. Evaluates on test set — accuracy, macro F1, confusion matrix
6. Saves results to ml_results.json (read by web dashboard)
7. Saves trained models to models/ folder

Run:
    python scripts/train_baseline.py
    python scripts/train_baseline.py --data data/ --window 64 --overlap 0.5
─────────────────────────────────────────────────────────────────────
"""

import os, sys, json, glob, time, argparse
import numpy as np
import h5py
from collections import defaultdict

# ── IMPORTS ───────────────────────────────────────────────────────────────────
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (accuracy_score, f1_score,
                                 confusion_matrix, classification_report)
    from sklearn.model_selection import RandomizedSearchCV
    import joblib
except ImportError:
    print("ERROR: scikit-learn not installed.")
    print("Run: pip install scikit-learn joblib")
    sys.exit(1)

# ── CONFIG ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data",    default="data",  help="HDF5 data folder")
parser.add_argument("--window",  type=int, default=64, help="Sliding window size (frames)")
parser.add_argument("--overlap", type=float, default=0.5, help="Window overlap fraction")
parser.add_argument("--models",  default="models", help="Output folder for trained models")
parser.add_argument("--results", default="ml_results.json", help="Output JSON for web dashboard")
args = parser.parse_args()

WINDOW   = args.window
STEP     = max(1, int(WINDOW * (1 - args.overlap)))
DATA_DIR = args.data
MODEL_DIR= args.models
LABEL_NAMES = {0:"Empty", 1:"1 Person", 2:"2 People", 3:"3 People"}
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)

print()
print("=" * 60)
print("  DIAG Baseline ML Training — EEM 499 FYP")
print("=" * 60)
print(f"  Data folder : {DATA_DIR}")
print(f"  Window size : {WINDOW} frames (~{WINDOW/5.3:.1f}s)")
print(f"  Step size   : {STEP} frames ({int((1-args.overlap)*100)}% overlap)")
print(f"  Models out  : {MODEL_DIR}/")
print(f"  Results out : {args.results}")
print("=" * 60)
print()

# ── FEATURE EXTRACTION ────────────────────────────────────────────────────────

def extract_features_window(dist, rssi):
    """
    Extract 20 statistical features from one sliding window of one board.
    Input:  dist [W,]  — distance values in cm
            rssi [W,]  — RSSI values in dBm
    Output: feature vector [20,]
    """
    feats = []

    # Distance features (10)
    feats.append(np.mean(dist))                                    # 1. mean distance
    feats.append(np.std(dist))                                     # 2. std distance
    feats.append(np.var(dist))                                     # 3. variance distance
    feats.append(np.max(dist) - np.min(dist))                      # 4. range (max-min)
    feats.append(np.mean(np.abs(np.diff(dist))))                   # 5. mean rate of change
    feats.append(np.sum(dist > 200) / len(dist))                   # 6. outlier rate (>200cm)
    feats.append(np.percentile(dist, 75) - np.percentile(dist, 25))# 7. IQR distance
    feats.append(np.max(dist))                                     # 8. max distance
    feats.append(_skewness(dist))                                  # 9. skewness
    feats.append(_kurtosis(dist))                                  # 10. kurtosis

    # RSSI features (10)
    feats.append(np.mean(rssi))                                    # 11. mean RSSI
    feats.append(np.std(rssi))                                     # 12. std RSSI
    feats.append(np.var(rssi))                                     # 13. variance RSSI
    feats.append(np.max(rssi) - np.min(rssi))                      # 14. range RSSI
    feats.append(np.mean(np.abs(np.diff(rssi))))                   # 15. mean rate of change
    feats.append(np.min(rssi))                                     # 16. minimum RSSI
    feats.append(np.percentile(rssi, 75) - np.percentile(rssi, 25))# 17. IQR RSSI
    feats.append(np.percentile(rssi, 10))                          # 18. 10th percentile RSSI
    feats.append(_skewness(rssi))                                  # 19. skewness RSSI
    feats.append(_kurtosis(rssi))                                  # 20. kurtosis RSSI

    return np.array(feats, dtype=np.float32)

def _skewness(x):
    mu = np.mean(x); s = np.std(x)
    if s < 1e-8: return 0.0
    return float(np.mean(((x - mu) / s) ** 3))

def _kurtosis(x):
    mu = np.mean(x); s = np.std(x)
    if s < 1e-8: return 0.0
    return float(np.mean(((x - mu) / s) ** 4) - 3)

def extract_session_features(dist0, rssi0, dist1, rssi1):
    """
    Apply sliding window to a full session and extract features.
    Returns: X [n_windows, 40], using both board perspectives.
    """
    n = min(len(dist0), len(dist1))
    X = []
    for start in range(0, n - WINDOW + 1, STEP):
        end = start + WINDOW
        f0 = extract_features_window(dist0[start:end], rssi0[start:end])  # 20 feats Board A
        f1 = extract_features_window(dist1[start:end], rssi1[start:end])  # 20 feats Board B
        X.append(np.concatenate([f0, f1]))                                 # 40 feats total
    return np.array(X, dtype=np.float32) if X else np.empty((0, 40))

FEATURE_NAMES = []
for board in ["A", "B"]:
    for feat in ["mean_dist","std_dist","var_dist","range_dist","roc_dist",
                 "outlier_rate","iqr_dist","max_dist","skew_dist","kurt_dist",
                 "mean_rssi","std_rssi","var_rssi","range_rssi","roc_rssi",
                 "min_rssi","iqr_rssi","p10_rssi","skew_rssi","kurt_rssi"]:
        FEATURE_NAMES.append(f"{feat}_board{board}")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────

print("Loading sessions from data/ folder...")
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.h5")))
if not files:
    print(f"  ERROR: No .h5 files found in '{DATA_DIR}'")
    print("  Make sure you have collected data using capture.py first.")
    sys.exit(1)

sessions = []   # list of { label, session_name, X [n_windows, 40] }
label_counts = defaultdict(int)

for fpath in files:
    fname = os.path.basename(fpath)
    try:
        with h5py.File(fpath, "r") as f:
            label   = int(f.attrs.get("label", -1))
            session = str(f.attrs.get("session", fname))
            dist0   = f["board_0/distance_cm"][:]
            rssi0   = f["board_0/rssi_dbm"][:]
            dist1   = f["board_1/distance_cm"][:]
            rssi1   = f["board_1/rssi_dbm"][:]

        X = extract_session_features(dist0, rssi0, dist1, rssi1)
        if len(X) == 0:
            print(f"  SKIP {fname} — too short for window size {WINDOW}")
            continue

        sessions.append({"label": label, "session": session, "X": X, "file": fname})
        label_counts[label] += 1
        print(f"  Loaded {fname:40s} | Label {label} | {len(X):3d} windows")

    except Exception as e:
        print(f"  ERROR loading {fname}: {e}")

print()
print(f"  Total sessions loaded: {len(sessions)}")
for lbl in sorted(label_counts):
    print(f"  Label {lbl} ({LABEL_NAMES.get(lbl,'?')}): {label_counts[lbl]} sessions")
print()

if len(sessions) == 0:
    print("ERROR: No valid sessions loaded. Exiting.")
    sys.exit(1)

# ── SESSION-STRATIFIED SPLIT ──────────────────────────────────────────────────

print("Splitting by session (70 / 15 / 15)...")

by_label = defaultdict(list)
for s in sessions:
    by_label[s["label"]].append(s)

train_sessions, val_sessions, test_sessions = [], [], []

for lbl, sess_list in by_label.items():
    np.random.seed(RANDOM_STATE)
    idx = np.random.permutation(len(sess_list))
    n_train = max(1, int(0.70 * len(sess_list)))
    n_val   = max(1, int(0.15 * len(sess_list)))

    train_sessions += [sess_list[i] for i in idx[:n_train]]
    val_sessions   += [sess_list[i] for i in idx[n_train:n_train+n_val]]
    test_sessions  += [sess_list[i] for i in idx[n_train+n_val:]]

def build_Xy(sess_list):
    X_all, y_all = [], []
    for s in sess_list:
        X_all.append(s["X"])
        y_all.extend([s["label"]] * len(s["X"]))
    if not X_all:
        return np.empty((0,40)), np.array([])
    return np.vstack(X_all), np.array(y_all)

X_train, y_train = build_Xy(train_sessions)
X_val,   y_val   = build_Xy(val_sessions)
X_test,  y_test  = build_Xy(test_sessions)

print(f"  Train: {len(train_sessions)} sessions → {len(X_train)} windows")
print(f"  Val  : {len(val_sessions)} sessions → {len(X_val)} windows")
print(f"  Test : {len(test_sessions)} sessions → {len(X_test)} windows")
print()

# ── FEATURE SCALING ───────────────────────────────────────────────────────────

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
print("  Scaler saved to models/scaler.pkl")

# ── TRAIN RANDOM FOREST ───────────────────────────────────────────────────────

print()
print("-" * 60)
print("  Training Random Forest...")
print("-" * 60)

t0 = time.time()
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X_train_s, y_train)
rf_train_time = time.time() - t0
print(f"  Training time: {rf_train_time:.1f}s")

# Validation
y_val_pred_rf = rf.predict(X_val_s)
rf_val_f1  = f1_score(y_val, y_val_pred_rf, average="macro", zero_division=0)
rf_val_acc = accuracy_score(y_val, y_val_pred_rf)
print(f"  Validation — Accuracy: {rf_val_acc*100:.1f}%  Macro F1: {rf_val_f1*100:.1f}%")

# Test
y_test_pred_rf = rf.predict(X_test_s)
rf_test_f1  = f1_score(y_test, y_test_pred_rf, average="macro", zero_division=0)
rf_test_acc = accuracy_score(y_test, y_test_pred_rf)
rf_cm       = confusion_matrix(y_test, y_test_pred_rf).tolist()
print(f"  Test       — Accuracy: {rf_test_acc*100:.1f}%  Macro F1: {rf_test_f1*100:.1f}%")

# Feature importance
importances = rf.feature_importances_
top_idx = np.argsort(importances)[::-1][:10]
print("\n  Top 10 most important features:")
for rank, i in enumerate(top_idx, 1):
    print(f"    {rank:2d}. {FEATURE_NAMES[i]:<30s} {importances[i]:.4f}")

joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.pkl"))
print(f"\n  Model saved to models/random_forest.pkl")

# ── TRAIN SVM ─────────────────────────────────────────────────────────────────

print()
print("-" * 60)
print("  Training SVM (RBF kernel)...")
print("-" * 60)

t0 = time.time()

# Quick hyperparameter search on validation set
param_dist = {
    "C":     [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
}
svm_base = SVC(kernel="rbf", class_weight="balanced",
               decision_function_shape="ovr", random_state=RANDOM_STATE)
svm_search = RandomizedSearchCV(
    svm_base, param_dist, n_iter=12, cv=3,
    scoring="f1_macro", n_jobs=-1,
    random_state=RANDOM_STATE, verbose=0
)
svm_search.fit(X_train_s, y_train)
svm = svm_search.best_estimator_
svm_train_time = time.time() - t0

print(f"  Best params: C={svm.C}, gamma={svm.gamma}")
print(f"  Training time: {svm_train_time:.1f}s")

# Validation
y_val_pred_svm = svm.predict(X_val_s)
svm_val_f1  = f1_score(y_val, y_val_pred_svm, average="macro", zero_division=0)
svm_val_acc = accuracy_score(y_val, y_val_pred_svm)
print(f"  Validation — Accuracy: {svm_val_acc*100:.1f}%  Macro F1: {svm_val_f1*100:.1f}%")

# Test
y_test_pred_svm = svm.predict(X_test_s)
svm_test_f1  = f1_score(y_test, y_test_pred_svm, average="macro", zero_division=0)
svm_test_acc = accuracy_score(y_test, y_test_pred_svm)
svm_cm       = confusion_matrix(y_test, y_test_pred_svm).tolist()
print(f"  Test       — Accuracy: {svm_test_acc*100:.1f}%  Macro F1: {svm_test_f1*100:.1f}%")

joblib.dump(svm, os.path.join(MODEL_DIR, "svm.pkl"))
print(f"\n  Model saved to models/svm.pkl")

# ── FINAL REPORT ──────────────────────────────────────────────────────────────

labels_present = sorted(label_counts.keys())

print()
print("=" * 60)
print("  FINAL RESULTS SUMMARY")
print("=" * 60)
print()
print(f"  {'Model':<20} {'Accuracy':>10} {'Macro F1':>10}")
print(f"  {'-'*42}")
print(f"  {'Random Forest':<20} {rf_test_acc*100:>9.1f}% {rf_test_f1*100:>9.1f}%")
print(f"  {'SVM (RBF)':<20} {svm_test_acc*100:>9.1f}% {svm_test_f1*100:>9.1f}%")
print()

# Which model is better
best_model = "Random Forest" if rf_test_f1 >= svm_test_f1 else "SVM (RBF)"
best_f1    = max(rf_test_f1, svm_test_f1)
print(f"  Best DIAG baseline model: {best_model} ({best_f1*100:.1f}% macro F1)")
print()

# Per-class accuracy
print("  Per-class results (Random Forest):")
rf_report = classification_report(y_test, y_test_pred_rf,
                                  target_names=[LABEL_NAMES.get(l,str(l)) for l in labels_present],
                                  labels=labels_present, zero_division=0, output_dict=True)
for lbl in labels_present:
    name = LABEL_NAMES.get(lbl, str(lbl))
    stats = rf_report.get(name, {})
    print(f"    Label {lbl} ({name:<10}): "
          f"P={stats.get('precision',0)*100:.1f}%  "
          f"R={stats.get('recall',0)*100:.1f}%  "
          f"F1={stats.get('f1-score',0)*100:.1f}%")

print()
print("  Confusion Matrix — Random Forest (rows=actual, cols=predicted):")
print(f"  {'':12}", end="")
for lbl in labels_present:
    print(f"  L{lbl}", end="")
print()
for i, lbl in enumerate(labels_present):
    print(f"  L{lbl} ({LABEL_NAMES.get(lbl,'?'):6})", end="")
    for j in range(len(labels_present)):
        print(f"  {rf_cm[i][j]:3d}", end="")
    print()
print()

# ── SAVE RESULTS JSON ─────────────────────────────────────────────────────────

# Per-class F1 for chart
per_class_rf = {}
for lbl in labels_present:
    name = LABEL_NAMES.get(lbl, str(lbl))
    stats = rf_report.get(name, {})
    per_class_rf[str(lbl)] = round(stats.get("f1-score", 0) * 100, 1)

svm_report = classification_report(y_test, y_test_pred_svm,
                                   target_names=[LABEL_NAMES.get(l,str(l)) for l in labels_present],
                                   labels=labels_present, zero_division=0, output_dict=True)
per_class_svm = {}
for lbl in labels_present:
    name = LABEL_NAMES.get(lbl, str(lbl))
    stats = svm_report.get(name, {})
    per_class_svm[str(lbl)] = round(stats.get("f1-score", 0) * 100, 1)

results = {
    "status": "complete",
    "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "dataset": {
        "sessions_total": len(sessions),
        "windows_train":  int(len(X_train)),
        "windows_val":    int(len(X_val)),
        "windows_test":   int(len(X_test)),
        "labels":         {str(k): v for k,v in label_counts.items()},
        "window_size":    WINDOW,
        "step_size":      STEP,
        "n_features":     40,
    },
    "results": [
        {
            "model":    "Random Forest",
            "approach": "DIAG baseline",
            "accuracy": round(rf_test_acc, 4),
            "f1":       round(rf_test_f1, 4),
            "params":   {"n_estimators": 300, "class_weight": "balanced"},
        },
        {
            "model":    "SVM (RBF)",
            "approach": "DIAG baseline",
            "accuracy": round(svm_test_acc, 4),
            "f1":       round(svm_test_f1, 4),
            "params":   {"C": svm.C, "gamma": str(svm.gamma)},
        },
    ],
    "comparison": [
        {"model":"Random Forest","approach":"DIAG","accuracy":round(rf_test_acc,4),"f1":round(rf_test_f1,4)},
        {"model":"SVM (RBF)",    "approach":"DIAG","accuracy":round(svm_test_acc,4),"f1":round(svm_test_f1,4)},
        {"model":"CNN",          "approach":"CIR", "accuracy":None,"f1":None},
    ],
    "cm_diag": rf_cm,
    "per_class": {
        "Random Forest": per_class_rf,
        "SVM (RBF)":     per_class_svm,
    },
    "feature_importance": {
        FEATURE_NAMES[i]: round(float(importances[i]), 6)
        for i in top_idx[:20]
    },
    "labels_present": labels_present,
    "label_names": {str(k): v for k, v in LABEL_NAMES.items()},
}

RESULT_FILE = args.results
with open(RESULT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"  Results saved to {RESULT_FILE}")
print(f"  Open http://localhost:5000/ml to view in dashboard")
print()
print("=" * 60)
print("  Training complete!")
print(f"  Best model : {best_model}")
print(f"  Best F1    : {best_f1*100:.1f}%")
print("=" * 60)
print()
print("  Next steps:")
print("  1. Check /ml page on the web dashboard")
print("  2. Review confusion matrix — which labels are confused?")
print("  3. Proceed to CIR firmware modification (capture_cir.py)")
print()