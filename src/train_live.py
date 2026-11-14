"""Train the NetGuard LIVE model on CICIDS2017 using only features the Scapy
flow tracker can compute (see flow_tracker.FEATURES).

Usage (from the src folder, venv active):
    python train_live.py --quick          # small & fast sanity run
    python train_live.py                  # full run
    python train_live.py --keep-port      # also use Destination Port as a feature
    python train_live.py --data-dir D:\\path\\to\\csv_folder
"""
import argparse
import re
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from flow_tracker import FEATURES

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR, REPORTS_DIR = ROOT / "models", ROOT / "reports"
SEED = 42


def clean_label(raw):
    s = re.sub(r"[^\x00-\x7F]+", "-", str(raw)).strip()      # fixes the odd "Web Attack ? XSS" char
    low = s.lower()
    if low == "benign":
        return "Normal"
    if low == "ddos":
        return "DDoS"
    if low.startswith("dos") or low == "heartbleed":
        return "DoS"
    if low == "portscan":
        return "PortScan"
    if low.endswith("patator"):
        return "BruteForce"
    if low.startswith("web attack"):
        return "WebAttack"
    if low == "bot":
        return "Bot"
    if low == "infiltration":
        return "Infiltration"
    raise ValueError(f"Unknown CICIDS2017 label: {raw!r}")


def cap_classes(df, cap):
    """Keep at most `cap` rows of each class (works on any pandas version)."""
    keep = [g.sample(min(len(g), cap), random_state=SEED) for _, g in df.groupby("category")]
    return pd.concat(keep)


def load_cicids(data_dir, features, cap_per_class, quick):
    files = sorted(Path(data_dir).rglob("*ISCX.csv"))
    if not files:
        raise SystemExit(f"No *ISCX.csv files found under {data_dir}")
    wanted = set(features) | {"Label"}
    parts = []
    for f in files:
        t0 = time.time()
        df = pd.read_csv(f, encoding="latin-1", low_memory=False,
                         usecols=lambda c: c.strip() in wanted)
        df.columns = df.columns.str.strip()
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise SystemExit(f"{f.name} is missing columns: {missing}")
        df["category"] = df.pop("Label").map(clean_label)
        df[features] = (df[features].apply(pd.to_numeric, errors="coerce")
                        .replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32"))
        df = df.drop_duplicates()
        df = cap_classes(df, cap_per_class)
        parts.append(df)
        print(f"  {f.name:55s} kept {len(df):>7,} rows ({time.time() - t0:.0f}s)")
    data = pd.concat(parts, ignore_index=True)
    return cap_classes(data, cap_per_class).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--keep-port", action="store_true")
    a = ap.parse_args()
    cap = 3000 if a.quick else 60000
    features = [f for f in FEATURES if a.keep_port or f != "Destination Port"]
    MODELS_DIR.mkdir(exist_ok=True); REPORTS_DIR.mkdir(exist_ok=True)

    print(f"Loading CICIDS2017 (cap {cap:,} rows per class per file) ...")
    data = load_cicids(a.data_dir, features, cap, a.quick)
    print("\nClass counts used for training/testing:")
    print(data["category"].value_counts().to_string(), "\n")

    le = LabelEncoder().fit(data["category"])
    X, y = data[features], le.transform(data["category"])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y,
                                              random_state=SEED)
    counts = np.bincount(y_tr)
    w_class = np.sqrt(counts.max() / counts)                 # gentle class balancing
    w = w_class[y_tr]

    model = XGBClassifier(n_estimators=100 if a.quick else 300, max_depth=8,
                          learning_rate=0.1, tree_method="hist", n_jobs=-1,
                          random_state=SEED, eval_metric="mlogloss")
    t0 = time.time()
    model.fit(X_tr, y_tr, sample_weight=w)
    print(f"Trained in {time.time() - t0:.0f}s on {len(X_tr):,} flows, {len(features)} features")

    proba = model.predict_proba(X_te)
    pred = proba.argmax(axis=1)
    names = list(le.classes_)
    print("\nMulti-class test results")
    print(f"  accuracy  {accuracy_score(y_te, pred):.4f}")
    print(f"  macro-F1  {f1_score(y_te, pred, average='macro'):.4f}")
    report = classification_report(y_te, pred, target_names=names, zero_division=0)
    print(report)

    normal = names.index("Normal")                           # binary view of the same model
    y_bin, p_bin = (y_te != normal).astype(int), (pred != normal).astype(int)
    print("Binary view (normal vs any attack)")
    print(f"  accuracy {accuracy_score(y_bin, p_bin):.4f}  precision {precision_score(y_bin, p_bin):.4f}"
          f"  recall {recall_score(y_bin, p_bin):.4f}  F1 {f1_score(y_bin, p_bin):.4f}")

    (REPORTS_DIR / "live_report.txt").write_text(report)
    cm = confusion_matrix(y_te, pred)
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Live model (CICIDS2017) - test split")
    plt.tight_layout(); plt.savefig(REPORTS_DIR / "confusion_live.png", dpi=150)
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    imp.to_csv(REPORTS_DIR / "feature_importance_live.csv", header=["importance"])
    print("\nTop 10 features:\n", imp.head(10).round(4).to_string())

    joblib.dump({"model": model, "label_encoder": le, "feature_columns": features,
                 "model_name": "XGBoost-live", "benign_label": "Normal"},
                MODELS_DIR / "netguard_live.joblib")
    print("\nSaved model -> models/netguard_live.joblib")


if __name__ == "__main__":
    main()