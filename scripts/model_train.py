#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py
────────
StoichML unified training pipeline — all four tasks in one script.

Tasks:
  enthalpy   formation enthalpy regression (eV/atom)
             LightGBM + XGBoost, 5-fold CV, full-dataset final fit

  egap       band gap regression (eV) — insulators only (hurdle model)
             LightGBM + XGBoost, 5-fold CV, final fit on Egap > 0 subset
             At inference: run only when egap_type predicts class 1
             No log1p transform — deeper models used for this task.

  egap_type  metal vs insulator classification
             5-seed ensemble undersampling (strict 1:1 balance)
             Seed-outer / fold-inner CV — one LightGBM model per seed

  hm_class   conductor / insulator / half-metal classification
             5-seed capped undersampling (classes 0+1 capped at 3000 each)
             Seed-outer / fold-inner CV — one LightGBM model per seed

Usage:
    python train.py --task enthalpy
    python train.py --task egap
    python train.py --task egap_type
    python train.py --task hm_class
"""

import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    balanced_accuracy_score,
    roc_curve,
    precision_recall_fscore_support,
    confusion_matrix,
)

import lightgbm as lgb
import xgboost as xgb


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

RANDOM_STATE = 42
N_SPLITS     = 5

DATA_PATH     = "data/data_feat.pkl"
FEATURES_JSON = "data/selected_features.json"
OUT_DIR       = "models"

# Fixed undersampling seeds — must match feature_selection.py
UNDERSAMPLE_SEEDS = [0, 7, 21, 42, 99]

# hm_class capped undersampling — must match feature_selection.py
# Classes 0 and 1 drawn to min(MAJORITY_CAP, available) per fold.
# 661 class-2 + 3000 class-0 + 3000 class-1 = ~3,661 rows per model
# (ratio ≈ 4.5:4.5:1 vs original 63:12:1)
MAJORITY_CAP = 3000

TASKS = {
    "enthalpy": {
        "type":      "regression",
        "target":    "enthalpy_formation_atom",
        "transform": None,
    },
    "egap": {
        "type":      "regression",
        "target":    "Egap",
        "transform": None,          # No log1p — distribution is clean & continuous
        "filter":    ("Egap", ">", 0),
        "deep":      True,          # Flag to use deeper egap-specific model builders
    },
    "egap_type": {
        "type":           "binary_ensemble",
        "target":         "Egap_type_numeric",
        "minority_class": 1,        # insulator — class 0 downsampled to match
    },
    "hm_class": {
        "type":           "multiclass_ensemble",
        "target":         "hm_class",
        "n_classes":      3,
        "minority_class": 2,        # half-metal — classes 0+1 capped at MAJORITY_CAP
    },
}

NON_FEATURE_COLS = [
    "compound", "spacegroup_relax", "Egap", "Egap_type",
    "Egap_type_numeric", "enthalpy_formation_atom",
    "composition", "elements", "hm_class",
]


# ══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def load_features(task: str) -> list[str]:
    """Load task-specific selected features from feature selection JSON."""
    with open(FEATURES_JSON) as f:
        feats = json.load(f)
    task_feats = feats.get(task, [])
    if not task_feats:
        raise ValueError(
            f"No features found for task '{task}' in {FEATURES_JSON}. "
            "Run feature_selection.py first."
        )
    return task_feats


def apply_filter(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Apply optional row-level filter from task config.
    Filter format: (column, operator, value).
    Applied to raw df before target extraction so that Egap > 0 filters
    on original values, not on log-transformed ones.
    Returns reset-index df so iloc-based fold splits work correctly.
    """
    if "filter" not in cfg:
        return df

    col, op, val = cfg["filter"]
    ops = {
        ">":  lambda s: s >  val,
        ">=": lambda s: s >= val,
        "<":  lambda s: s <  val,
        "<=": lambda s: s <= val,
        "==": lambda s: s == val,
        "!=": lambda s: s != val,
    }
    if op not in ops:
        raise ValueError(f"Unsupported filter operator '{op}'. "
                         f"Choose from {list(ops)}")

    mask = ops[op](df[col])
    print(f"  Filter: {col} {op} {val}  →  {mask.sum()} / {len(df)} rows retained "
          f"({(~mask).sum()} dropped)")
    return df.loc[mask].reset_index(drop=True)


def transform_target(y: np.ndarray, mode: str | None) -> np.ndarray:
    if mode == "log1p":
        return np.log1p(np.clip(y, a_min=0, a_max=None))
    return y


def inverse_transform(y: np.ndarray, mode: str | None) -> np.ndarray:
    if mode == "log1p":
        return np.expm1(y)
    return y


def summarize(metric_list: list[dict]) -> dict:
    """Mean ± std across CV folds for every metric key."""
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

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, thr: float) -> dict:
    pred = (prob >= thr).astype(int)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, pred, labels=[0, 1], zero_division=0
    )
    return {
        "roc_auc":          float(roc_auc_score(y_true, prob)),
        "pr_auc":           float(average_precision_score(y_true, prob)),
        "macro_f1":         float(f1_score(y_true, pred, average="macro",    zero_division=0)),
        "balanced_acc":     float(balanced_accuracy_score(y_true, pred)),
        "f1_class0":        float(f[0]),
        "f1_class1":        float(f[1]),
        "precision_class1": float(p[1]),
        "recall_class1":    float(r[1]),
    }


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       n_classes: int) -> dict:
    """Full per-class metrics. f1_class2 is primary diagnostic for hm_class."""
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


def oof_best_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Youden-J optimal threshold from full OOF predictions — unbiased."""
    fpr, tpr, thr = roc_curve(y_true, prob)
    return float(thr[np.argmax(tpr - fpr)])


# ══════════════════════════════════════════════════════════════════════════════
# UNDERSAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def undersample_strict(
    X: pd.DataFrame,
    y: pd.Series,
    minority_class: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Strict 1:1 undersampling for egap_type.
    Downsamples all non-minority classes to the fold-local minority count.
    Using fold-local count (not full-dataset count) prevents ValueError
    when the majority fold size < full-dataset minority size.
    Must be identical to feature_selection.py.
    """
    rng          = np.random.default_rng(seed)
    idx_minority = y[y == minority_class].index
    n_minority   = len(idx_minority)   # fold-local, not full-dataset

    balanced_idx = list(idx_minority)
    for cls in sorted(y.unique()):
        if cls == minority_class:
            continue
        idx_cls  = y[y == cls].index
        n_sample = min(n_minority, len(idx_cls))
        balanced_idx.extend(rng.choice(idx_cls, size=n_sample, replace=False))

    balanced_idx = np.array(balanced_idx)
    rng.shuffle(balanced_idx)
    return X.loc[balanced_idx], y.loc[balanced_idx]


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
    For hm_class: 661 class-2 + min(3000, avail) class-0 + min(3000, avail) class-1.
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
# MODEL BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_lgbm_regression(n_estimators: int = 5000) -> lgb.LGBMRegressor:
    """Standard LGBM regressor — used for enthalpy."""
    return lgb.LGBMRegressor(
        learning_rate=0.05,
        num_leaves=64,
        n_estimators=n_estimators,
        min_data_in_leaf=30,
        subsample=0.8,
        colsample_bytree=0.8,
        importance_type="gain",
        random_state=RANDOM_STATE,
    )


def build_lgbm_regression_egap(n_estimators: int = 5000) -> lgb.LGBMRegressor:
    """
    Deeper LGBM regressor tuned for band gap prediction.

    Changes vs standard:
      - num_leaves 64 → 127: deeper trees capture nonlinear structure–gap relations
      - min_data_in_leaf 30 → 10: finer splits on 7931 insulator samples
      - learning_rate 0.05 → 0.03: slower learning compensates for deeper trees,
        reduces overfitting at leaf level
      - reg_alpha=0.05, reg_lambda=0.1: L1+L2 regularisation to prevent
        overfitting on noisy composition→gap mapping
      - No log1p transform: distribution is unimodal & continuous after Egap>0
        filter, so transform only compresses the 5–9 eV tail and hurts MAE there
    """
    return lgb.LGBMRegressor(
        learning_rate=0.03,
        num_leaves=127,
        n_estimators=n_estimators,
        min_data_in_leaf=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=0.1,
        importance_type="gain",
        random_state=RANDOM_STATE,
    )


def build_xgb_regression(
    n_estimators: int = 5000,
    early_stopping: bool = True,
) -> xgb.XGBRegressor:
    """
    Standard XGBoost regressor — used for enthalpy.
    XGBoost >= 2.0: early_stopping_rounds lives in the constructor.
    Pass early_stopping=False for the final fit — no eval_set is available.
    """
    return xgb.XGBRegressor(
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.9,
        n_estimators=n_estimators,
        tree_method="hist",
        random_state=RANDOM_STATE,
        eval_metric="rmse",
        early_stopping_rounds=200 if early_stopping else None,
    )


def build_xgb_regression_egap(
    n_estimators: int = 5000,
    early_stopping: bool = True,
) -> xgb.XGBRegressor:
    """
    Deeper XGBoost regressor tuned for band gap prediction.

    Changes vs standard:
      - max_depth 6 → 8: captures deeper feature interactions for gap prediction
      - min_child_weight 1 → 3: prevents overfitting on small leaf nodes
        (pairs with deeper trees — depth without min_child_weight causes overfit)
      - reg_alpha=0.05, reg_lambda=1.5: stronger L2 regularisation than default
        (default lambda=1) to stabilise deeper trees
      - gamma=0.1: minimum loss reduction to make a split — prunes splits that
        add little predictive value, acts as a soft max_depth limiter
      - colsample_bytree 0.9 → 0.8: matches LGBM, adds feature diversity
      - subsample stays 0.8: already well-tuned
    """
    return xgb.XGBRegressor(
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=1.5,
        gamma=0.1,
        n_estimators=n_estimators,
        tree_method="hist",
        random_state=RANDOM_STATE,
        eval_metric="rmse",
        early_stopping_rounds=200 if early_stopping else None,
    )


def build_lgbm_binary(n_estimators: int = 5000) -> lgb.LGBMClassifier:
    """
    Binary classifier for egap_type ensemble.
    No class_weight — training data is already balanced by undersampling.
    num_class=1 explicitly set — LightGBM raises a fatal error if num_class > 1
    is passed for a binary objective.
    """
    return lgb.LGBMClassifier(
        objective="binary",
        num_class=1,
        learning_rate=0.05,
        num_leaves=64,
        n_estimators=n_estimators,
        min_data_in_leaf=20,
        subsample=0.8,
        colsample_bytree=0.8,
        importance_type="gain",
        random_state=RANDOM_STATE,
    )


def build_lgbm_multiclass(
    n_classes: int,
    n_estimators: int = 5000,
) -> lgb.LGBMClassifier:
    """
    Multiclass classifier for hm_class ensemble.
    No class_weight — capped undersampling already reduces ratio to ~4.5:4.5:1.
    Adding class weighting on top would double-correct and over-amplify class 2.
    min_data_in_leaf=10 (smaller than binary) because training sets are smaller.
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
        random_state=RANDOM_STATE,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TASK TRAINERS
# ══════════════════════════════════════════════════════════════════════════════

def train_regression(task_name: str, cfg: dict) -> None:
    """
    LightGBM + XGBoost regression with 5-fold CV.

    For the egap task, deeper task-specific model builders are used
    (build_lgbm_regression_egap / build_xgb_regression_egap) when
    cfg["deep"] is True. All other tasks use the standard builders.

    OOF predictions are stored in transformed space and inverted once at
    metric time — storing already-inverted predictions then inverting again
    would apply expm1 twice for log1p targets.

    CV folds materialised once and shared between LGBM and XGBoost so both
    models are evaluated on identical validation sets.
    """
    df = pd.read_pickle(DATA_PATH)
    print(f"\n  {len(df)} total samples")

    # Filter before target extraction — Egap > 0 filter must operate on
    # original values, not log-transformed ones
    df    = apply_filter(df, cfg)
    feats = load_features(task_name)

    X = df[feats].values
    y = transform_target(df[cfg["target"]].values, cfg.get("transform"))

    use_deep = cfg.get("deep", False)

    print(f"  Features: {len(feats)}")
    print(f"  y range (transformed): [{y.min():.4f}, {y.max():.4f}]  "
          f"mean: {y.mean():.4f}")
    if use_deep:
        print(f"  Model variant: DEEP (egap-specific hyperparameters, no log1p)")
    else:
        print(f"  Model variant: STANDARD")

    cv    = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(cv.split(X))   # materialised once — both models see same splits

    # Use DataFrame for LightGBM so feature names are preserved in the model
    X_df = df[feats]

    metrics_lgb    = []
    metrics_xgb    = []
    oof_lgb        = np.zeros(len(y))   # transformed space — inverted at metric time
    oof_xgb        = np.zeros(len(y))
    best_iters_lgb = []
    best_iters_xgb = []

    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        print(f"\n  ── Fold {fold}/{N_SPLITS} {'─' * 38}")

        Xtr_df, Xva_df = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        ytr,    yva    = y[tr_idx],          y[va_idx]

        # Select model builders based on task depth flag
        if use_deep:
            lgbm = build_lgbm_regression_egap()
            xgbm = build_xgb_regression_egap(early_stopping=True)
        else:
            lgbm = build_lgbm_regression()
            xgbm = build_xgb_regression(early_stopping=True)

        lgbm.fit(
            Xtr_df, ytr,
            eval_set=[(Xva_df, yva)],
            eval_metric="l2",
            callbacks=[
                lgb.early_stopping(200, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        xgbm.fit(
            Xtr_df, ytr,
            eval_set=[(Xva_df, yva)],
            verbose=False,
        )

        best_iters_lgb.append(lgbm.best_iteration_)
        best_iters_xgb.append(xgbm.best_iteration)

        # Store raw transformed-space predictions in OOF
        raw_lgb = lgbm.predict(Xva_df)
        raw_xgb = xgbm.predict(Xva_df)
        oof_lgb[va_idx] = raw_lgb
        oof_xgb[va_idx] = raw_xgb

        # Invert once for per-fold metrics (original units)
        p_lgb = inverse_transform(raw_lgb, cfg.get("transform"))
        p_xgb = inverse_transform(raw_xgb, cfg.get("transform"))
        y_val = inverse_transform(yva,     cfg.get("transform"))

        m_lgb = regression_metrics(y_val, p_lgb)
        m_xgb = regression_metrics(y_val, p_xgb)
        metrics_lgb.append(m_lgb)
        metrics_xgb.append(m_xgb)

        print(f"  LGBM  iter={lgbm.best_iteration_:4d}  "
              f"MAE={m_lgb['mae']:.4f}  RMSE={m_lgb['rmse']:.4f}  R²={m_lgb['r2']:.4f}")
        print(f"  XGB   iter={xgbm.best_iteration:4d}  "
              f"MAE={m_xgb['mae']:.4f}  RMSE={m_xgb['rmse']:.4f}  R²={m_xgb['r2']:.4f}")

    # OOF metrics — invert once from transformed space
    y_orig    = inverse_transform(y,       cfg.get("transform"))
    oof_m_lgb = regression_metrics(y_orig, inverse_transform(oof_lgb, cfg.get("transform")))
    oof_m_xgb = regression_metrics(y_orig, inverse_transform(oof_xgb, cfg.get("transform")))
    # Average in transformed space then invert — more stable for log1p
    oof_m_ens = regression_metrics(
        y_orig,
        inverse_transform((oof_lgb + oof_xgb) / 2, cfg.get("transform")),
    )

    print(f"\n  OOF (original units):")
    print(f"  LGBM      MAE={oof_m_lgb['mae']:.4f}  RMSE={oof_m_lgb['rmse']:.4f}  "
          f"R²={oof_m_lgb['r2']:.4f}")
    print(f"  XGB       MAE={oof_m_xgb['mae']:.4f}  RMSE={oof_m_xgb['rmse']:.4f}  "
          f"R²={oof_m_xgb['r2']:.4f}")
    print(f"  Ensemble  MAE={oof_m_ens['mae']:.4f}  RMSE={oof_m_ens['rmse']:.4f}  "
          f"R²={oof_m_ens['r2']:.4f}")

    # Final fit using mean best iteration across folds
    mean_iter_lgb = max(1, int(np.mean(best_iters_lgb)))
    mean_iter_xgb = max(1, int(np.mean(best_iters_xgb)))
    print(f"\n  Final fit — LGBM iter={mean_iter_lgb}  XGB iter={mean_iter_xgb}")

    if use_deep:
        final_lgb = build_lgbm_regression_egap(n_estimators=mean_iter_lgb)
        final_xgb = build_xgb_regression_egap(n_estimators=mean_iter_xgb, early_stopping=False)
    else:
        final_lgb = build_lgbm_regression(n_estimators=mean_iter_lgb)
        final_xgb = build_xgb_regression(n_estimators=mean_iter_xgb, early_stopping=False)

    final_lgb.fit(X_df, y)
    final_xgb.fit(X_df, y)

    # Save
    task_dir = os.path.join(OUT_DIR, task_name)
    os.makedirs(task_dir, exist_ok=True)
    joblib.dump(final_lgb, os.path.join(task_dir, f"{task_name}_lgbm.pkl"))
    joblib.dump(final_xgb, os.path.join(task_dir, f"{task_name}_xgb.pkl"))

    results = {
        "cv_folds":   {"lgbm": summarize(metrics_lgb), "xgb": summarize(metrics_xgb)},
        "oof":        {"lgbm": oof_m_lgb, "xgb": oof_m_xgb, "ensemble": oof_m_ens},
        "n_samples":  int(len(y)),
        "n_features": int(len(feats)),
        "transform":  cfg.get("transform") or "none",
        "filter":     str(cfg.get("filter", "none")),
        "model_variant": "deep" if use_deep else "standard",
    }
    save_json(results, os.path.join(task_dir, f"{task_name}_metrics.json"))

    print(f"\n  Saved → {task_dir}/")
    _print_regression_summary(results, N_SPLITS)


def train_egap_type(task_name: str, cfg: dict) -> None:
    """
    5-seed ensemble undersampling for egap_type (binary: metal vs insulator).

    Structure: seed-outer / fold-inner
      For each seed → run full N_SPLITS-fold CV → accumulate per-seed OOF
    After all seeds → compute ensemble OOF (mean probability across seeds)

    Validation folds are NEVER undersampled — always the full distribution.
    Features and undersampling logic match feature_selection.py exactly.
    """
    df    = pd.read_pickle(DATA_PATH)
    feats = load_features(task_name)

    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns],
                errors="ignore")[feats]
    y = df[cfg["target"]]

    minority_class = cfg["minority_class"]

    print(f"\n  {len(y)} samples")
    print(f"  Class 0 (Conductor): {(y == 0).sum()}")
    print(f"  Class 1 (Insulator): {(y == 1).sum()}")
    print(f"  Ratio: {(y == 0).sum() / (y == 1).sum():.1f}:1")
    print(f"  Features: {len(feats)}")
    print(f"\n  Strategy: {N_SPLITS}-fold CV × {len(UNDERSAMPLE_SEEDS)} seeds "
          f"= {N_SPLITS * len(UNDERSAMPLE_SEEDS)} total models")

    cv    = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(cv.split(X, y))   # materialised once — same splits for all seeds

    oof_probs_by_seed: dict[int, np.ndarray] = {}
    seed_cv_metrics:   dict[int, list[dict]] = {}
    best_iters:        dict[int, list[int]]  = {}
    final_models:      dict[int, lgb.LGBMClassifier] = {}
    y_oof = np.zeros(len(y))   # ground truth — filled on first seed

    # ── Outer: seeds ──────────────────────────────────────────────────────
    for seed in UNDERSAMPLE_SEEDS:
        print(f"\n{'═' * 55}")
        print(f"  Seed {seed}")
        print(f"{'═' * 55}")

        oof_prob     = np.zeros(len(y))
        fold_metrics = []
        iters        = []

        # ── Inner: folds ──────────────────────────────────────────────────
        for fold, (tr_idx, va_idx) in enumerate(folds, 1):
            Xva = X.iloc[va_idx]
            yva = y.iloc[va_idx]

            if seed == UNDERSAMPLE_SEEDS[0]:
                y_oof[va_idx] = yva.values

            Xtr_bal, ytr_bal = undersample_strict(
                X.iloc[tr_idx], y.iloc[tr_idx], minority_class, seed
            )

            print(f"  fold {fold}/{N_SPLITS} | "
                  f"train: {len(ytr_bal)} "
                  f"({(ytr_bal==0).sum()} / {(ytr_bal==1).sum()})", end="")

            model = build_lgbm_binary()
            model.fit(
                Xtr_bal, ytr_bal,
                eval_set=[(Xva, yva)],
                eval_metric="auc",
                callbacks=[
                    lgb.early_stopping(200, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )

            iters.append(model.best_iteration_)
            prob = model.predict_proba(Xva)[:, 1]
            oof_prob[va_idx] = prob

            fm = binary_metrics(yva.values, prob, thr=0.5)
            fold_metrics.append(fm)
            print(f"  |  iter={model.best_iteration_:4d}  "
                  f"AUC={fm['roc_auc']:.4f}  F1_cls1={fm['f1_class1']:.4f}")

        oof_probs_by_seed[seed] = oof_prob
        seed_cv_metrics[seed]   = fold_metrics
        best_iters[seed]        = iters

        seed_thr = oof_best_threshold(y_oof, oof_prob)
        seed_oof = binary_metrics(y_oof, oof_prob, thr=seed_thr)
        print(f"\n  Seed {seed} OOF (thr={seed_thr:.3f}):  "
              f"AUC={seed_oof['roc_auc']:.4f}  "
              f"macro_F1={seed_oof['macro_f1']:.4f}  "
              f"F1_cls1={seed_oof['f1_class1']:.4f}")

    # Ensemble OOF
    ensemble_prob = np.mean([oof_probs_by_seed[s] for s in UNDERSAMPLE_SEEDS], axis=0)
    thr_ensemble  = oof_best_threshold(y_oof, ensemble_prob)
    ens_oof_m     = binary_metrics(y_oof, ensemble_prob, thr=thr_ensemble)

    print(f"\n{'═' * 55}")
    print(f"  Ensemble OOF (mean of {len(UNDERSAMPLE_SEEDS)} seeds, "
          f"thr={thr_ensemble:.3f})")
    print(f"{'═' * 55}")
    for k, v in ens_oof_m.items():
        print(f"    {k}: {v:.4f}")

    # Final fit — one model per seed, full dataset
    print(f"\n  Final fit — full dataset, one model per seed")
    for seed in UNDERSAMPLE_SEEDS:
        mean_iter = max(1, int(np.mean(best_iters[seed])))
        print(f"  seed {seed:3d} | mean_iter={mean_iter}")

        X_bal, y_bal = undersample_strict(X, y, minority_class, seed)
        model = build_lgbm_binary(n_estimators=mean_iter)
        model.fit(X_bal, y_bal)
        final_models[seed] = model

    # Save
    task_dir = os.path.join(OUT_DIR, task_name)
    os.makedirs(task_dir, exist_ok=True)

    for seed, model in final_models.items():
        joblib.dump(model, os.path.join(task_dir, f"{task_name}_lgbm_seed{seed}.pkl"))

    results = {
        "ensemble_oof":       ens_oof_m,
        "ensemble_threshold": thr_ensemble,
        "per_seed_cv":   {str(s): summarize(seed_cv_metrics[s]) for s in UNDERSAMPLE_SEEDS},
        "per_seed_oof":  {
            str(s): binary_metrics(
                y_oof, oof_probs_by_seed[s],
                thr=oof_best_threshold(y_oof, oof_probs_by_seed[s]),
            )
            for s in UNDERSAMPLE_SEEDS
        },
        "undersample_seeds": UNDERSAMPLE_SEEDS,
        "n_features":        int(len(feats)),
    }
    save_json(results, os.path.join(task_dir, f"{task_name}_metrics.json"))
    save_json({"ensemble_threshold": thr_ensemble},
              os.path.join(task_dir, f"{task_name}_threshold.json"))

    print(f"\n  Saved → {task_dir}/")
    print(f"  {task_name}_lgbm_seed{{seed}}.pkl ×{len(UNDERSAMPLE_SEEDS)}  "
          f"+  metrics.json  +  threshold.json")


def train_hm_class(task_name: str, cfg: dict) -> None:
    """
    5-seed capped-undersampling ensemble for hm_class (3-class).

    Structure: seed-outer / fold-inner  (same as egap_type)
    Undersampling: classes 0 and 1 capped at MAJORITY_CAP (3000) each.
    Minority class 2 (half-metal, 661 samples) always kept whole.
    Strict 1:1:1 would give ~1,983 rows — too few for 151 features.
    Capped gives ~3,661 rows (ratio ≈ 4.5:4.5:1 vs original 63:12:1).

    Validation folds always the full distribution.
    """
    df    = pd.read_pickle(DATA_PATH)
    feats = load_features(task_name)

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

    # ── Outer: seeds ──────────────────────────────────────────────────────
    for seed in UNDERSAMPLE_SEEDS:
        print(f"\n{'═' * 55}")
        print(f"  Seed {seed}")
        print(f"{'═' * 55}")

        oof_proba    = np.zeros((len(y), n_classes))
        fold_metrics = []
        iters        = []

        # ── Inner: folds ──────────────────────────────────────────────────
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

    # Ensemble OOF — mean softmax across seeds
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

    # Final fit — one model per seed, full dataset
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

    # Save
    task_dir = os.path.join(OUT_DIR, task_name)
    os.makedirs(task_dir, exist_ok=True)

    for seed, model in final_models.items():
        joblib.dump(model, os.path.join(task_dir, f"{task_name}_lgbm_seed{seed}.pkl"))

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
    save_json(results, os.path.join(task_dir, f"{task_name}_metrics.json"))

    print(f"\n  Saved → {task_dir}/")
    print(f"  {task_name}_lgbm_seed{{seed}}.pkl ×{len(UNDERSAMPLE_SEEDS)}  "
          f"+  metrics.json")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY PRINTERS
# ══════════════════════════════════════════════════════════════════════════════

def _print_regression_summary(results: dict, n_splits: int) -> None:
    print(f"\n{'═' * 55}")
    print(f"  CV Summary  (mean ± std across {n_splits} folds)")
    print(f"{'═' * 55}")
    for model_name, cv_m in results["cv_folds"].items():
        mae  = cv_m["mae"]
        rmse = cv_m["rmse"]
        r2   = cv_m["r2"]
        print(f"  {model_name.upper():<6}  "
              f"MAE={mae['mean']:.4f}±{mae['std']:.4f}  "
              f"RMSE={rmse['mean']:.4f}±{rmse['std']:.4f}  "
              f"R²={r2['mean']:.4f}±{r2['std']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def predict_regression(
    X_new: pd.DataFrame,
    task_name: str,
    model_dir: str = OUT_DIR,
) -> np.ndarray:
    """
    Ensemble (mean of LGBM + XGBoost) prediction in original target space.
    For egap: call only on samples where egap_type predicts class 1.
    """
    cfg      = TASKS[task_name]
    task_dir = os.path.join(model_dir, task_name)
    lgbm     = joblib.load(os.path.join(task_dir, f"{task_name}_lgbm.pkl"))
    xgbm     = joblib.load(os.path.join(task_dir, f"{task_name}_xgb.pkl"))
    p_lgb    = inverse_transform(lgbm.predict(X_new), cfg.get("transform"))
    p_xgb    = inverse_transform(xgbm.predict(X_new), cfg.get("transform"))
    return (p_lgb + p_xgb) / 2


def predict_egap_type(
    X_new: pd.DataFrame,
    model_dir: str = OUT_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (pred, prob) where pred is 0/1 class and prob is mean class-1
    probability across all 5 seed models.
    Load threshold from egap_type_threshold.json for production use.
    """
    task_dir = os.path.join(model_dir, "egap_type")
    models   = [
        joblib.load(os.path.join(task_dir, f"egap_type_lgbm_seed{s}.pkl"))
        for s in UNDERSAMPLE_SEEDS
    ]
    thr_path = os.path.join(task_dir, "egap_type_threshold.json")
    thr      = json.load(open(thr_path))["ensemble_threshold"] if os.path.exists(thr_path) else 0.5
    prob     = np.mean([m.predict_proba(X_new)[:, 1] for m in models], axis=0)
    return (prob >= thr).astype(int), prob


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
# DISPATCHER
# ══════════════════════════════════════════════════════════════════════════════

def train(task_name: str) -> None:
    cfg = TASKS[task_name]

    print(f"\n{'═' * 55}")
    print(f"  Task: {task_name}  |  type: {cfg['type']}")
    print(f"{'═' * 55}")

    task_type = cfg["type"]

    if task_type == "regression":
        train_regression(task_name, cfg)

    elif task_type == "binary_ensemble":
        train_egap_type(task_name, cfg)

    elif task_type == "multiclass_ensemble":
        train_hm_class(task_name, cfg)

    else:
        raise ValueError(f"Unknown task type '{task_type}' for task '{task_name}'")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="StoichML unified training pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python train.py                      # run all tasks\n"
            "  python train.py --task enthalpy      # single task\n"
            "  python train.py --task egap_type"
        ),
    )
    parser.add_argument(
        "--task",
        choices=list(TASKS.keys()),
        default=None,
        help=(
            "Task to train. If omitted, all tasks are run in order: "
            + ", ".join(TASKS.keys())
        ),
    )
    args = parser.parse_args()

    tasks_to_run = [args.task] if args.task else list(TASKS.keys())

    for task_name in tasks_to_run:
        train(task_name)

    print(f"\n{'═' * 55}")
    print(f"  Done — trained: {', '.join(tasks_to_run)}")
    print(f"{'═' * 55}")