"""
app.py — FYP Local Web Dashboard
Occupancy Estimation Using UWB Radar and Machine Learning
Muhammad Hafizul Bin Ahmad Husni | 161115 | USM

Run: python app.py
Open: http://localhost:5000
"""

from flask import Flask, render_template, jsonify, Response, request
import serial, threading, time, json, glob, os, queue
import numpy as np, h5py

app = Flask(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
BOARD_A_PORT = "COM6"
BOARD_B_PORT = "COM8"
BAUD         = 115200
DATA_FOLDER  = "data"
LABEL_NAMES  = {0:"Empty", 1:"1 Person", 2:"2 People", 3:"3 People"}

# ── Live state ────────────────────────────────────────────────────────────────
live = {
    "running": False,
    "board0":  {"dist": 0, "rssi": 0, "frames": 0},
    "board1":  {"dist": 0, "rssi": 0, "frames": 0},
    "occupancy": "—",
    "history": {"t": [], "dist0": [], "rssi0": [], "dist1": [], "rssi1": []},
}
live_lock    = threading.Lock()
sse_clients  = []
sse_lock     = threading.Lock()

import re

def parse_session_info(line):
    dist = re.search(r"distance\[cm\]=(\d+)", line)
    rssi = re.search(r"RSSI\[dBm\]=([-\d.]+)", line)
    if dist and rssi:
        return int(dist.group(1)), float(rssi.group(1))
    return None, None

def broadcast(data):
    dead = []
    with sse_lock:
        for q in sse_clients:
            try:
                q.put_nowait(data)
            except:
                dead.append(q)
        for q in dead:
            sse_clients.remove(q)

def read_board(board_idx, port):
    try:
        ser = serial.Serial(port, BAUD, timeout=1)
        time.sleep(1)
        ser.write(b"\r\n")
        time.sleep(0.3)
        ser.write(b"DIAG 1\r\n")
        time.sleep(0.3)
        cmd = b"INITF\r\n" if board_idx == 0 else b"RESPF\r\n"
        ser.write(cmd)
        time.sleep(0.3)
        ser.reset_input_buffer()

        waiting = False
        while live["running"]:
            line = ser.readline().decode(errors="ignore").strip()
            if "SESSION_INFO_NTF" in line:
                waiting = True
            elif waiting and "distance[cm]" in line:
                dist, rssi = parse_session_info(line)
                if dist is not None:
                    with live_lock:
                        live[f"board{board_idx}"]["dist"] = dist
                        live[f"board{board_idx}"]["rssi"] = rssi
                        live[f"board{board_idx}"]["frames"] += 1
                        t_now = time.time()
                        h = live["history"]
                        if len(h["t"]) == 0 or t_now - h["t"][-1] > 0.3:
                            h["t"].append(round(t_now, 2))
                            h["dist0"].append(live["board0"]["dist"])
                            h["rssi0"].append(live["board0"]["rssi"])
                            h["dist1"].append(live["board1"]["dist"])
                            h["rssi1"].append(live["board1"]["rssi"])
                            if len(h["t"]) > 120:
                                for k in h: h[k] = h[k][-120:]
                            # simple occupancy heuristic from RSSI
                            r = (live["board0"]["rssi"] + live["board1"]["rssi"]) / 2
                            d_var = abs(live["board0"]["dist"] - 185)
                            if r > -63:   occ = "Empty (Label 0)"
                            elif r > -64: occ = "1 Person (Label 1)"
                            elif r > -65: occ = "2 People (Label 2)"
                            else:         occ = "3 People (Label 3)"
                            live["occupancy"] = occ
                            broadcast(json.dumps({
                                "dist0": dist if board_idx == 0 else live["board0"]["dist"],
                                "rssi0": rssi if board_idx == 0 else live["board0"]["rssi"],
                                "dist1": live["board1"]["dist"],
                                "rssi1": live["board1"]["rssi"],
                                "frames0": live["board0"]["frames"],
                                "frames1": live["board1"]["frames"],
                                "occupancy": live["occupancy"],
                            }))
                waiting = False
        ser.close()
    except Exception as e:
        print(f"Board {board_idx} error: {e}")
        with live_lock:
            live["running"] = False

# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    # Dynamically count sessions per label from data folder
    files = sorted(glob.glob(os.path.join(DATA_FOLDER, "*.h5")))
    label_counts = {0:0, 1:0, 2:0, 3:0}
    label_target = {0:10, 1:10, 2:10, 3:10}
    for f in files:
        try:
            with h5py.File(f, "r") as hf:
                lbl = int(hf.attrs.get("label", -1))
                if lbl in label_counts:
                    label_counts[lbl] += 1
        except:
            pass
    dataset_status = []
    descs = {0:"Empty room", 1:"1 person", 2:"2 people", 3:"3 people"}
    for lbl in range(4):
        count  = label_counts[lbl]
        target = label_target[lbl]
        done   = count >= target
        dataset_status.append({
            "label":  lbl,
            "desc":   descs[lbl],
            "count":  count,
            "target": target,
            "done":   done,
        })
    return render_template("index.html", dataset_status=dataset_status)

@app.route("/live")
def live_page():
    return render_template("live.html")

@app.route("/data")
def data_page():
    return render_template("data.html")

@app.route("/ml")
def ml_page():
    return render_template("ml.html")

# ── LIVE API ──────────────────────────────────────────────────────────────────

@app.route("/api/live/start", methods=["POST"])
def start_live():
    if live["running"]:
        return jsonify({"ok": False, "msg": "Already running"})
    live["running"] = True
    live["board0"] = {"dist": 0, "rssi": 0, "frames": 0}
    live["board1"] = {"dist": 0, "rssi": 0, "frames": 0}
    live["history"] = {"t": [], "dist0": [], "rssi0": [], "dist1": [], "rssi1": []}
    threading.Thread(target=read_board, args=(1, BOARD_B_PORT), daemon=True).start()
    time.sleep(1.5)
    threading.Thread(target=read_board, args=(0, BOARD_A_PORT), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/live/stop", methods=["POST"])
def stop_live():
    live["running"] = False
    return jsonify({"ok": True})

@app.route("/api/live/status")
def live_status():
    with live_lock:
        return jsonify({
            "running":   live["running"],
            "board0":    live["board0"],
            "board1":    live["board1"],
            "occupancy": live["occupancy"],
            "history":   live["history"],
        })

@app.route("/api/live/stream")
def live_stream():
    q = queue.Queue(maxsize=50)
    with sse_lock:
        sse_clients.append(q)
    def generate():
        while True:
            try:
                data = q.get(timeout=30)
                yield f"data: {data}\n\n"
            except queue.Empty:
                yield "data: {\"ping\":1}\n\n"
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ── DATA API ──────────────────────────────────────────────────────────────────

@app.route("/api/data/sessions")
def list_sessions():
    files = sorted(glob.glob(os.path.join(DATA_FOLDER, "*.h5")))
    sessions = []
    for f in files:
        try:
            with h5py.File(f, "r") as hf:
                label = int(hf.attrs.get("label", -1))
                n0 = len(hf["board_0/distance_cm"][:])
                n1 = len(hf["board_1/distance_cm"][:])
                d0 = hf["board_0/distance_cm"][:]
                r0 = hf["board_0/rssi_dbm"][:]
                sessions.append({
                    "file":       os.path.basename(f),
                    "label":      label,
                    "label_name": LABEL_NAMES.get(label, f"Label {label}"),
                    "frames0":    n0,
                    "frames1":    n1,
                    "mean_dist":  round(float(d0.mean()), 1),
                    "mean_rssi":  round(float(r0.mean()), 1),
                })
        except Exception as e:
            sessions.append({"file": os.path.basename(f), "error": str(e)})
    return jsonify(sessions)

@app.route("/api/data/session/<filename>")
def get_session(filename):
    path = os.path.join(DATA_FOLDER, filename)
    if not os.path.exists(path):
        return jsonify({"error": "Not found"}), 404
    with h5py.File(path, "r") as hf:
        label = int(hf.attrs.get("label", -1))
        result = {"label": label, "label_name": LABEL_NAMES.get(label, f"Label {label}"), "boards": {}}
        for idx in [0, 1]:
            d = hf[f"board_{idx}/distance_cm"][:]
            r = hf[f"board_{idx}/rssi_dbm"][:]
            t = hf[f"board_{idx}/timestamps"][:]
            t_rel = (t - t[0]).tolist()
            result["boards"][str(idx)] = {
                "t":         t_rel,
                "dist":      d.tolist(),
                "rssi":      r.tolist(),
                "mean_dist": round(float(d.mean()), 1),
                "std_dist":  round(float(d.std()), 1),
                "mean_rssi": round(float(r.mean()), 1),
                "std_rssi":  round(float(r.std()), 1),
                "max_dist":  round(float(d.max()), 1),
                "frames":    len(d),
            }
    return jsonify(result)

@app.route("/api/data/summary")
def data_summary():
    files = sorted(glob.glob(os.path.join(DATA_FOLDER, "*.h5")))
    by_label = {}
    for f in files:
        try:
            with h5py.File(f, "r") as hf:
                label = int(hf.attrs.get("label", -1))
                d = hf["board_0/distance_cm"][:]
                r = hf["board_0/rssi_dbm"][:]
                if label not in by_label:
                    by_label[label] = {"dist": [], "rssi": [], "count": 0}
                by_label[label]["dist"].extend(d.tolist())
                by_label[label]["rssi"].extend(r.tolist())
                by_label[label]["count"] += 1
        except:
            pass
    result = {}
    for lbl, vals in by_label.items():
        d = np.array(vals["dist"])
        r = np.array(vals["rssi"])
        result[lbl] = {
            "label_name":  LABEL_NAMES.get(lbl, f"Label {lbl}"),
            "sessions":    vals["count"],
            "dist_median": round(float(np.median(d)), 1),
            "dist_q1":     round(float(np.percentile(d, 25)), 1),
            "dist_q3":     round(float(np.percentile(d, 75)), 1),
            "dist_min":    round(float(d.min()), 1),
            "dist_max":    round(float(d.max()), 1),
            "rssi_median": round(float(np.median(r)), 1),
            "rssi_q1":     round(float(np.percentile(r, 25)), 1),
            "rssi_q3":     round(float(np.percentile(r, 75)), 1),
            "rssi_min":    round(float(r.min()), 1),
            "rssi_max":    round(float(r.max()), 1),
            "dist_raw":    d.tolist(),
            "rssi_raw":    r.tolist(),
        }
    return jsonify(result)

# ── ML API ────────────────────────────────────────────────────────────────────

@app.route("/api/ml/results")
def ml_results():
    path = "ml_results.json"
    if os.path.exists(path):
        with open(path) as f:
            return jsonify(json.load(f))
    # Placeholder until real results exist
    return jsonify({
        "status": "pending",
        "message": "ML training not yet completed. Run train_baseline.py to generate results.",
        "results": []
    })

if __name__ == "__main__":
    os.makedirs(DATA_FOLDER, exist_ok=True)
    print("\n" + "="*50)
    print("  FYP UWB Dashboard — Starting...")
    print("  Open http://localhost:5000 in your browser")
    print("="*50 + "\n")
    app.run(debug=True, threaded=True, port=5000)