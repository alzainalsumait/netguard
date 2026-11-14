"""Diagnose why live flows are classified differently from CICIDS2017.

Compares flows from YOUR tracker (e.g. an Nmap scan) with CICIDS2017 PortScan and
Normal flows, feature by feature, ordered by how much the model relies on each feature.

Steps (Administrator prompt):
  1) Window A:  python flow_tracker.py --live --seconds 60 --out scan_flows.csv
  2) Window B:  nmap -sS -p 1-1000 192.168.0.112
  3) python compare_features.py --flows scan_flows.csv --ip 192.168.0.112
"""
import argparse

import numpy as np
import pandas as pd

from detector import MODEL_PATH, classify, load_model
from train_live import ROOT, load_cicids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows", required=True)
    ap.add_argument("--ip", required=True, help="IP of the scanned device")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--model", default=str(MODEL_PATH))
    ap.add_argument("--cap", type=int, default=5000)
    a = ap.parse_args()

    bundle = load_model(a.model)
    feats = bundle["feature_columns"]
    flows = pd.read_csv(a.flows)
    scan = flows[(flows["src_ip"] == a.ip) | (flows["dst_ip"] == a.ip)].copy()
    print(f"{len(scan)} flows involve {a.ip} (of {len(flows)} captured)")
    if scan.empty:
        raise SystemExit("No flows for that IP - check the IP and the capture file.")

    res = classify(scan, bundle)
    print("\nWhat the model says about these flows:")
    print(res["label"].value_counts().to_string())
    classes = list(bundle["label_encoder"].classes_)
    X = scan[feats].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    proba = pd.DataFrame(bundle["model"].predict_proba(X), columns=classes)
    print("\nAverage probability per class:")
    print(proba.mean().round(3).sort_values(ascending=False).to_string())

    print("\nLoading CICIDS2017 reference flows ...")
    ref = load_cicids(a.data_dir, feats, a.cap, False)
    port = ref[ref["category"] == "PortScan"][feats]
    norm = ref[ref["category"] == "Normal"][feats]
    imp = pd.Series(bundle["model"].feature_importances_, index=feats).sort_values(ascending=False)

    rows = []
    for f in imp.index:
        lo, hi = port[f].quantile(0.05), port[f].quantile(0.95)
        v = scan[f].median()
        rows.append({"feature": f, "importance": round(imp[f], 3), "your_scan": round(v, 2),
                     "CICIDS_PortScan": round(port[f].median(), 2),
                     "CICIDS_Normal": round(norm[f].median(), 2),
                     "in_PortScan_range": "yes" if lo <= v <= hi else "NO"})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\nFeatures the model relies on most (NO = your value is outside the usual PortScan range):")
    print(out.head(20).to_string(index=False))
    print(f"\n{(out['in_PortScan_range'] == 'NO').sum()} of {len(out)} features are outside the PortScan range.")
    out.to_csv(ROOT / "reports" / "feature_comparison.csv", index=False)


if __name__ == "__main__":
    main()