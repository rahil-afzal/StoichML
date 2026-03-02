#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_hm_class.py
─────────────────
StoichML training pipeline — hm_class only.

Task:
  hm_class   conductor / insulator / half-metal classification
             5-seed capped undersampling (classes 0+1 capped at 3000 each)
             Seed-outer / fold-inner CV — one LightGBM model per seed

Usage:
    python train_hm_class.py
"""

import json
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

import lightgbm as lgb


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

RANDOM_STATE = 42
N_SPLITS     = 5

DATA_PATH     = "data/data_feat.pkl"
FEATURES_JSON = "data/selected_features.json"
OUT_DIR       = "models"

UNDERSAMPLE_SEEDS = [0, 7, 21, 42, 99]

# Classes 0 and 1 drawn to min(MAJORITY_CAP, available) per fold.
# 661 class-2 + 3000 class-0 + 3000 class-1 = ~3,661 rows per model
# (ratio ≈ 4.5:4.5:1 vs original 63:12:1)
MAJORITY_CAP = 3000

TASK_CFG = {
    "target":         "hm_class",
    "n_classes":      3,
    "minority_class": 2,   # half-metal — classes 0+1 capped at MAJORITY_CAP
}

NON_FEATURE_COLS = [
    "compound", "spacegroup_relax", "Egap", "Egap_type",
    "Egap_type_numeric", "enthalpy_formation_atom",
    "composition", "elements", "hm_class",
]


# ══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def load_features() -> list[str]:
    with open(FEATURES_JSON) as f:
        feats = json.load(f)
    task_feats = feats.get("hm_class", [])
    if not task_feats:
        raise ValueError(
            f"No features found for 'hm_class' in {FEATURES_JSON}. "
            "Run feature_selection.py first."
        )
    return task_feats


def summarize(metric_list: list[dict]) -> dict:
    keys = metric_list[0].keys()
    return {
        k: {
            "mean": float(np.mean([m[k] for m in metric_list])),
            "std":  float(np.std( [m[k] for m in metric_list])),
        }
        for k in keys
    }


def save_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       n_classes: int) -> dict:
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )
    metrics = {
        "macro_f1":     float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
        "weighted_f1":  float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }
    for i in range(n_classes):
        metrics[f"f1_class{i}"]        = float(f[i])
        metrics[f"precision_class{i}"] = float(p[i])
        metrics[f"recall_class{i}"]    = float(r[i])
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# UNDERSAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def undersample_capped(
    X: pd.DataFrame,
    y: pd.Series,
    minority_class: int,
    majority_cap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Capped undersampling for hm_class.
    Keeps all minority class samples. Downsamples every other class to
    min(majority_cap, available) — cap is independent of minority count.
    Must be identical to feature_selection.py.
    """
    rng          = np.random.default_rng(seed)
    idx_minority = y[y == minority_class].index

    balanced_idx = list(idx_minority)
    for cls in sorted(y.unique()):
        if cls == minority_class:
            continue
        idx_cls  = y[y == cls].index
        n_sample = min(majority_cap, len(idx_cls))
        balanced_idx.extend(rng.choice(idx_cls, size=n_sample, replace=False))

    balanced_idx = np.array(balanced_idx)
    rng.shuffle(balanced_idx)
    return X.loc[balanced_idx], y.loc[balanced_idx]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_lgbm_multiclass(
    n_classes: int,
    n_estimators: int = 5000,
) -> lgb.LGBMClassifier:
    """
    No class_weight — capped undersampling already reduces ratio to ~4.5:4.5:1.
    min_data_in_leaf=10 because training sets are smaller after undersampling.
    """
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        learning_rate=0.05,
        num_leaves=64,
        n_estimators=n_estimators,
        min_data_in_leaf=10,
        subsample=0.8,
        colsample_bytree=0.8,
        importance_type="gain",
        verbose=-1,
        random_state=RANDOM_STATE,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TRAINER
# ══════════════════════════════════════════════════════════════════════════════

def train_hm_class() -> None:
    cfg = TASK_CFG

    df    = pd.read_pickle(DATA_PATH)
    feats = load_features()

    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns],
                errors="ignore")[feats]
    y = df[cfg["target"]]

    n_classes      = cfg["n_classes"]
    minority_class = cfg["minority_class"]

    print(f"\n  {len(y)} samples")
    for cls, label in [(0, "Conductor"), (1, "Insulator"), (2, "Half-metal")]:
        print(f"  Class {cls} ({label}): {(y == cls).sum():6d}")
    print(f"\n  Strategy: cap majority classes at {MAJORITY_CAP}")
    effective = (y == minority_class).sum() + MAJORITY_CAP * (n_classes - 1)
    print(f"  ~{effective} rows per model  "
          f"(ratio ≈ {MAJORITY_CAP/(y==minority_class).sum():.1f}:"
          f"{MAJORITY_CAP/(y==minority_class).sum():.1f}:1)")
    print(f"  {N_SPLITS}-fold CV × {len(UNDERSAMPLE_SEEDS)} seeds "
          f"= {N_SPLITS * len(UNDERSAMPLE_SEEDS)} total models")
    print(f"  Features: {len(feats)}")

    cv    = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(cv.split(X, y))

    oof_probas_by_seed: dict[int, np.ndarray] = {}
    seed_cv_metrics:    dict[int, list[dict]] = {}
    best_iters:         dict[int, list[int]]  = {}
    final_models:       dict[int, lgb.LGBMClassifier] = {}
    y_oof = np.full(len(y), -1, dtype=int)

    for seed in UNDERSAMPLE_SEEDS:
        print(f"\n{'═' * 55}")
        print(f"  Seed {seed}")
        print(f"{'═' * 55}")

        oof_proba    = np.zeros((len(y), n_classes))
        fold_metrics = []
        iters        = []

        for fold, (tr_idx, va_idx) in enumerate(folds, 1):
            Xva = X.iloc[va_idx]
            yva = y.iloc[va_idx]

            if seed == UNDERSAMPLE_SEEDS[0]:
                y_oof[va_idx] = yva.values

            Xtr_bal, ytr_bal = undersample_capped(
                X.iloc[tr_idx], y.iloc[tr_idx], minority_class, MAJORITY_CAP, seed
            )

            counts_str = "  ".join(
                f"cls{c}:{(ytr_bal==c).sum()}" for c in range(n_classes)
            )
            print(f"  fold {fold}/{N_SPLITS} | {counts_str}", end="")

            model = build_lgbm_multiclass(n_classes)
            model.fit(
                Xtr_bal, ytr_bal,
                eval_set=[(Xva, yva)],
                eval_metric="multi_logloss",
                callbacks=[
                    lgb.early_stopping(200, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )

            iters.append(model.best_iteration_)
            proba = model.predict_proba(Xva)
            oof_proba[va_idx] = proba

            fold_pred = np.argmax(proba, axis=1)
            fm = multiclass_metrics(yva.values, fold_pred, n_classes)
            fold_metrics.append(fm)
            print(f"  |  iter={model.best_iteration_:4d}  "
                  f"macro_F1={fm['macro_f1']:.4f}  "
                  f"F1_cls2={fm['f1_class2']:.4f}")

        oof_probas_by_seed[seed] = oof_proba
        seed_cv_metrics[seed]    = fold_metrics
        best_iters[seed]         = iters

        seed_pred = np.argmax(oof_proba, axis=1)
        seed_oof  = multiclass_metrics(y_oof, seed_pred, n_classes)
        cm = confusion_matrix(y_oof, seed_pred, labels=list(range(n_classes)))
        print(f"\n  Seed {seed} OOF:  "
              f"macro_F1={seed_oof['macro_f1']:.4f}  "
              f"F1_cls2={seed_oof['f1_class2']:.4f}  "
              f"recall_cls2={seed_oof['recall_class2']:.4f}")
        print(f"  Confusion matrix (rows=true, cols=pred):\n{cm}")

    ens_proba = np.mean([oof_probas_by_seed[s] for s in UNDERSAMPLE_SEEDS], axis=0)
    ens_pred  = np.argmax(ens_proba, axis=1)
    ens_oof_m = multiclass_metrics(y_oof, ens_pred, n_classes)
    cm_ens    = confusion_matrix(y_oof, ens_pred, labels=list(range(n_classes)))

    print(f"\n{'═' * 55}")
    print(f"  Ensemble OOF (mean softmax of {len(UNDERSAMPLE_SEEDS)} seeds)")
    print(f"{'═' * 55}")
    for k, v in ens_oof_m.items():
        print(f"    {k}: {v:.4f}")
    print(f"\n  Ensemble confusion matrix (rows=true, cols=pred):\n{cm_ens}")

    print(f"\n  Final fit — full dataset, one model per seed")
    for seed in UNDERSAMPLE_SEEDS:
        mean_iter = max(1, int(np.mean(best_iters[seed])))
        print(f"  seed {seed:3d} | mean_iter={mean_iter}")

        X_bal, y_bal = undersample_capped(
            X, y, minority_class, MAJORITY_CAP, seed
        )
        model = build_lgbm_multiclass(n_classes, n_estimators=mean_iter)
        model.fit(X_bal, y_bal)
        final_models[seed] = model

    task_dir = os.path.join(OUT_DIR, "hm_class")
    os.makedirs(task_dir, exist_ok=True)

    for seed, model in final_models.items():
        joblib.dump(model, os.path.join(task_dir, f"hm_class_lgbm_seed{seed}.pkl"))

    results = {
        "ensemble_oof":      ens_oof_m,
        "per_seed_cv":  {str(s): summarize(seed_cv_metrics[s]) for s in UNDERSAMPLE_SEEDS},
        "per_seed_oof": {
            str(s): multiclass_metrics(
                y_oof, np.argmax(oof_probas_by_seed[s], axis=1), n_classes,
            )
            for s in UNDERSAMPLE_SEEDS
        },
        "majority_cap":      MAJORITY_CAP,
        "undersample_seeds": UNDERSAMPLE_SEEDS,
        "n_features":        int(len(feats)),
    }
    save_json(results, os.path.join(task_dir, "hm_class_metrics.json"))

    print(f"\n  Saved → {task_dir}/")
    print(f"  hm_class_lgbm_seed{{seed}}.pkl ×{len(UNDERSAMPLE_SEEDS)}  "
          f"+  hm_class_metrics.json")


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def predict_hm_class(
    X_new: pd.DataFrame,
    model_dir: str = OUT_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (pred, proba) where pred is argmax class label and proba is
    mean softmax matrix (n_samples, 3). proba[:, 2] = half-metal probability.
    """
    task_dir = os.path.join(model_dir, "hm_class")
    models   = [
        joblib.load(os.path.join(task_dir, f"hm_class_lgbm_seed{s}.pkl"))
        for s in UNDERSAMPLE_SEEDS
    ]
    proba = np.mean([m.predict_proba(X_new) for m in models], axis=0)
    return np.argmax(proba, axis=1), proba


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{'═' * 55}")
    print(f"  Task: hm_class  |  type: multiclass_ensemble")
    print(f"{'═' * 55}")

    train_hm_class()

    print(f"\n{'═' * 55}")
    print(f"  Done — trained: hm_class")
    print(f"{'═' * 55}")