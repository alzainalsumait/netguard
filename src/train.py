"""Train and evaluate NetGuard classifiers on NSL-KDD.

Usage (from the src folder, venv active):
    python train.py                       # 5-class attack category, 5-fold CV
    python train.py --target is_attack    # binary: normal vs attack
    python train.py --quick               # 20% of train data, 3 folds (fast test)
"""
import argparse
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from preprocess import build_preprocessor, load_nsl_kdd, split_xy

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR, REPORTS_DIR = ROOT / "models", ROOT / "reports"
SEED = 42
SMOTE_FLOOR = 5000  # raise every class below this many rows up to this (train folds only)


def make_pipeline(model, X, y):
    counts = pd.Series(y).value_counts()
    strategy = {c: SMOTE_FLOOR for c, n in counts.items() if n < SMOTE_FLOOR}
    steps = [("pre", build_preprocessor(X))]
    if strategy:  # SMOTE lives inside the pipeline: applied to training folds only
        steps.append(("smote", SMOTE(sampling_strategy=strategy, k_neighbors=3,
                                     random_state=SEED)))
    steps.append(("clf", model))
    return ImbPipeline(steps)


def get_models():
    return {
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                               random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                 tree_method="hist", n_jobs=-1, random_state=SEED),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                             max_iter=100, random_state=SEED),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="attack_category",
                    choices=["attack_category", "is_attack"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.folds = 3
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    train, test = load_nsl_kdd()
    X_tr, y_tr = split_xy(train, args.target)
    X_te, y_te = split_xy(test, args.target)
    if args.quick:
        X_tr, _, y_tr, _ = train_test_split(X_tr, y_tr, train_size=0.2,
                                            stratify=y_tr, random_state=SEED)
    le = LabelEncoder().fit(y_tr)
    y_tr_enc, y_te_enc = le.transform(y_tr), le.transform(y_te)
    binary = len(le.classes_) == 2
    class_names = ["Normal", "Attack"] if binary else [str(c) for c in le.classes_]
    print(f"Target={args.target}  classes={list(le.classes_)}  train={X_tr.shape}")

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    scoring = {"accuracy": "accuracy", "precision": "precision_macro",
               "recall": "recall_macro", "f1": "f1_macro",
               "auc": "roc_auc" if binary else "roc_auc_ovr"}

    rows, fitted = [], {}
    for name, model in get_models().items():
        t0 = time.time()
        pipe = make_pipeline(model, X_tr, y_tr_enc)
        cvres = cross_validate(pipe, X_tr, y_tr_enc, cv=cv, scoring=scoring)
        cv_mean = {k: cvres[f"test_{k}"].mean() for k in scoring}

        pipe.fit(X_tr, y_tr_enc)                       # final fit on all training data
        proba = pipe.predict_proba(X_te)
        pred = proba.argmax(axis=1)
        auc = roc_auc_score(y_te_enc, proba[:, 1]) if binary else \
            roc_auc_score(y_te_enc, proba, multi_class="ovr", average="macro")
        rows.append({
            "model": name,
            **{f"cv_{k}": round(v, 4) for k, v in cv_mean.items()},
            "test_accuracy": round(accuracy_score(y_te_enc, pred), 4),
            "test_precision": round(precision_score(y_te_enc, pred, average="macro",
                                                    zero_division=0), 4),
            "test_recall": round(recall_score(y_te_enc, pred, average="macro",
                                              zero_division=0), 4),
            "test_f1": round(f1_score(y_te_enc, pred, average="macro",
                                      zero_division=0), 4),
            "test_auc": round(auc, 4),
            "seconds": round(time.time() - t0),
        })
        fitted[name] = (pipe, pred)
        print(f"{name:13s} cv_f1={cv_mean['f1']:.4f}  test_f1={rows[-1]['test_f1']:.4f}"
              f"  ({rows[-1]['seconds']}s)")

    results = pd.DataFrame(rows)
    results.to_csv(REPORTS_DIR / f"results_{args.target}.csv", index=False)
    print("\n", results.to_string(index=False))

    best = results.loc[results["cv_f1"].idxmax(), "model"]   # chosen on CV, not on test
    pipe, pred = fitted[best]
    print(f"\nBest model by cross-validated macro-F1: {best}\n")
    print(classification_report(y_te_enc, pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_te_enc, pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"{best} - test set")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"confusion_{args.target}.png", dpi=150)

    joblib.dump({"pipeline": pipe, "label_encoder": le, "target": args.target,
                 "feature_columns": list(X_tr.columns), "model_name": best},
                MODELS_DIR / f"netguard_{args.target}.joblib")
    print(f"Saved model -> models/netguard_{args.target}.joblib")


if __name__ == "__main__":
    main()