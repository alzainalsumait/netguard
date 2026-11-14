"""NetGuard detection engine: flows -> live model -> SQLite alerts.

Usage (from the src folder, venv active; Administrator prompt for --live):
    python detector.py --csv live_flows.csv            # classify a flow CSV
    python detector.py --pcap test.pcap                # classify a capture file
    python detector.py --live --seconds 60             # real-time detection
    python detector.py --summary                       # show what is in the database
Options: --db PATH  --model PATH  --threshold 0.70  --iface NAME  --window 10
"""
import argparse
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scapy.all import AsyncSniffer, conf

from flow_tracker import FEATURES, META, FlowTracker, flows_from_pcap

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "netguard_live.joblib"
DB_PATH = ROOT / "netguard.db"
ALERT_THRESHOLD = 0.70     # an attack label must have at least this confidence to raise an alert

SCHEMA = """
CREATE TABLE IF NOT EXISTS flows (
    id INTEGER PRIMARY KEY AUTOINCREMENT, detected_at TEXT, flow_start TEXT,
    src_ip TEXT, dst_ip TEXT, src_port INTEGER, dst_port INTEGER, protocol INTEGER,
    label TEXT, confidence REAL, total_packets INTEGER, total_bytes REAL, is_alert INTEGER);
CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT, detected_at TEXT, flow_start TEXT,
    src_ip TEXT, dst_ip TEXT, dst_port INTEGER, protocol INTEGER,
    attack_type TEXT, confidence REAL);
"""


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def connect(db_path=DB_PATH):
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.executescript(SCHEMA)
    return conn


def classify(df, bundle, threshold=ALERT_THRESHOLD):
    """Return one row per flow with label, confidence and is_alert."""
    cols = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol", "start_time",
            "label", "confidence", "total_packets", "total_bytes", "is_alert"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    X = (df[bundle["feature_columns"]].apply(pd.to_numeric, errors="coerce")
         .replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32"))
    proba = bundle["model"].predict_proba(X)
    classes = np.asarray(bundle["label_encoder"].classes_)
    idx = proba.argmax(axis=1)
    out = df[["src_ip", "dst_ip", "src_port"]].copy()
    out["dst_port"] = df["Destination Port"]
    out["protocol"], out["start_time"] = df["protocol"], df["start_time"]
    out["label"], out["confidence"] = classes[idx], proba.max(axis=1)
    out["total_packets"] = df["Total Fwd Packets"] + df["Total Backward Packets"]
    out["total_bytes"] = df["Total Length of Fwd Packets"] + df["Total Length of Bwd Packets"]
    out["is_alert"] = ((out["label"] != bundle.get("benign_label", "Normal"))
                       & (out["confidence"] >= threshold)).astype(int)
    return out[cols]


def save(conn, res):
    now = datetime.now().isoformat(timespec="seconds")
    start = res["start_time"].map(lambda t: datetime.fromtimestamp(float(t)).isoformat(timespec="seconds"))
    flows = [(now, s, r.src_ip, r.dst_ip, int(r.src_port), int(r.dst_port), int(r.protocol),
              r.label, float(r.confidence), int(r.total_packets), float(r.total_bytes),
              int(r.is_alert)) for s, r in zip(start, res.itertuples())]
    conn.executemany("INSERT INTO flows (detected_at, flow_start, src_ip, dst_ip, src_port, dst_port,"
                     " protocol, label, confidence, total_packets, total_bytes, is_alert)"
                     " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", flows)
    alerts = [(now, s, r.src_ip, r.dst_ip, int(r.dst_port), int(r.protocol), r.label, float(r.confidence))
              for s, r in zip(start, res.itertuples()) if r.is_alert]
    conn.executemany("INSERT INTO incidents (detected_at, flow_start, src_ip, dst_ip, dst_port,"
                     " protocol, attack_type, confidence) VALUES (?,?,?,?,?,?,?,?)", alerts)
    conn.commit()
    return len(flows), len(alerts)


def process(df, bundle, conn, threshold):
    res = classify(df, bundle, threshold)
    if res.empty:
        return
    n, a = save(conn, res)
    print(f"[{datetime.now():%H:%M:%S}] {n} flows classified, {a} alerts  "
          f"{res['label'].value_counts().to_dict()}")
    for r in res[res["is_alert"] == 1].head(5).itertuples():
        print(f"    ALERT {r.label:<12} {r.src_ip} -> {r.dst_ip}:{r.dst_port}  conf={r.confidence:.2f}")


def run_live(bundle, conn, threshold, seconds, window, iface):
    tracker, buffer, lock = FlowTracker(), [], threading.Lock()

    def on_packet(pkt):
        with lock:
            buffer.extend(tracker.process_packet(pkt))

    sniffer = AsyncSniffer(iface=iface, store=False, prn=on_packet)
    print(f"Live detection on {iface or conf.iface} (Ctrl+C to stop)")
    sniffer.start()
    t_end = time.time() + seconds if seconds else None
    try:
        while t_end is None or time.time() < t_end:
            time.sleep(window)
            with lock:
                buffer.extend(tracker.expire(time.time()))
                done, buffer[:] = list(buffer), []
            process(pd.DataFrame(done, columns=META + FEATURES), bundle, conn, threshold)
    except KeyboardInterrupt:
        pass
    finally:
        sniffer.stop()
        with lock:
            done = buffer + tracker.flush()
        process(pd.DataFrame(done, columns=META + FEATURES), bundle, conn, threshold)


def summary(conn):
    f = pd.read_sql("SELECT label, COUNT(*) AS flows, SUM(is_alert) AS alerts FROM flows"
                    " GROUP BY label ORDER BY flows DESC", conn)
    print(f.to_string(index=False) if len(f) else "No flows stored yet.")
    inc = pd.read_sql("SELECT detected_at, attack_type, src_ip, dst_ip, dst_port, confidence"
                      " FROM incidents ORDER BY id DESC LIMIT 10", conn)
    if len(inc):
        print("\nLatest incidents:\n" + inc.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv"); g.add_argument("--pcap")
    g.add_argument("--live", action="store_true"); g.add_argument("--summary", action="store_true")
    ap.add_argument("--db", default=str(DB_PATH)); ap.add_argument("--model", default=str(MODEL_PATH))
    ap.add_argument("--threshold", type=float, default=ALERT_THRESHOLD)
    ap.add_argument("--seconds", type=int, default=0, help="0 = run until Ctrl+C")
    ap.add_argument("--window", type=int, default=10); ap.add_argument("--iface")
    a = ap.parse_args()

    conn = connect(a.db)
    if a.summary:
        return summary(conn)
    bundle = load_model(a.model)
    if a.csv:
        process(pd.read_csv(a.csv), bundle, conn, a.threshold)
    elif a.pcap:
        process(flows_from_pcap(a.pcap), bundle, conn, a.threshold)
    else:
        run_live(bundle, conn, a.threshold, a.seconds, a.window, a.iface)
    print(f"Database: {a.db}")


if __name__ == "__main__":
    main()