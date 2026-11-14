"""Turn your own captured flow CSVs into a labelled training file (data/local_flows.csv).

Capture with:   python flow_tracker.py --live --seconds 300 --out local_normal_1.csv
Then run:
    python build_local_dataset.py --normal local_normal_1.csv local_normal_2.csv ^
        --scan local_scan_1.csv --target-ip 192.168.0.112
"""
import argparse

import pandas as pd

from flow_tracker import FEATURES, META
from train_live import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", nargs="*", default=[], help="CSVs captured during normal use")
    ap.add_argument("--scan", nargs="*", default=[], help="CSVs captured while scanning --target-ip")
    ap.add_argument("--target-ip", required=True, help="IP of the device you scanned (your own)")
    ap.add_argument("--out", default=str(ROOT / "data" / "local_flows.csv"))
    a = ap.parse_args()

    parts = []
    for f in a.normal:
        d = pd.read_csv(f)
        d = d[(d["src_ip"] != a.target_ip) & (d["dst_ip"] != a.target_ip)]   # keep scan traffic out
        d["category"] = "Normal"
        print(f"{f}: {len(d)} Normal flows")
        parts.append(d)
    for f in a.scan:
        d = pd.read_csv(f)
        d = d[(d["src_ip"] == a.target_ip) | (d["dst_ip"] == a.target_ip)]   # only the scan flows
        d["category"] = "PortScan"
        print(f"{f}: {len(d)} PortScan flows")
        parts.append(d)
    if not parts:
        raise SystemExit("Give at least one --normal or --scan file.")
    out = pd.concat(parts, ignore_index=True)[META + FEATURES + ["category"]]
    out.to_csv(a.out, index=False)
    print(f"\n{len(out)} labelled flows -> {a.out}")
    print(out["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
