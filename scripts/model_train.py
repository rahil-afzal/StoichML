#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py
────────
StoichML training pipeline — LightGBM + XGBoost, 5-fold CV, full-data final fit.

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

TASKS = {
    "enthalpy": {
        "target":    "enthalpy_formation_atom",
        "type":      "regression",
        "transform": None,
    },
    "egap": {
        "target":    "Egap",
        "type":      "regression",
        "transform": "log1p",
    },
    "egap_type": {
        "target": "Egap_type_numeric",
        "type":   "binary",
    },
    "hm_class": {
        "target":       "hm_class",
        "type":         "multiclass",
        "num_class":    3,
        # class_weight: manually tuned — class 2 (half-metal) is severely rare.
        # These are passed to LightGBM directly and used to compute
        # sample_weight for XGBoost (which has no native class_weight param).
        "class_weight": {0: 1, 1: 1, 2: 8},
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def load_features(task: str) -> list[str]:
    """
    Load selected features for a task from the pruning JSON.
    Core key removed — each task now has its own independent feature list.
    """
    with open(FEATURES_JSON) as f:
        feats = json.load(f)
    task_feats = feats.get(task, [])
    if not task_feats:
        raise ValueError(
            f"No features found for task '{task}' in {FEATURES_JSON}. "
            f"Run feature_pruning_lgbm.py first."
        )
    return task_feats


def transform_target(y: pd.Series, mode: str | None) -> pd.Series:
    if mode == "log1p":
        return np.log1p(y.clip(lower=0))
    return y


def inverse_transform(y: np.ndarray, mode: str | None) -> np.ndarray:
    if mode == "log1p":
        return np.expm1(y)
    return y


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, thr: float) -> dict:
    pred = (prob >= thr).astype(int)
    return {
        "roc_auc":     float(roc_auc_score(y_true, prob)),
        "pr_auc":      float(average_precision_score(y_true, prob)),
        "macro_f1":    float(f1_score(y_true, pred, average="macro")),
        "balanced_acc":float(balanced_accuracy_score(y_true, pred)),
    }


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    """
    Compute macro F1, weighted F1, balanced accuracy, and per-class F1.
    Per-class F1 is essential for hm_class — macro alone hides whether
    class 2 (half-metal) is ever predicted at all.
    """
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


def summarize(metric_list: list[dict]) -> dict:
    """Mean and std across CV folds for each metric."""
    keys = metric_list[0].keys()
    return {
        k: {
            "mean": float(np.mean([m[k] for m in metric_list])),
            "std":  float(np.std( [m[k] for m in metric_list])),
        }
        for k in keys
    }


def compute_sample_weights(y: pd.Series, class_weight: dict) -> np.ndarray:
    """Map class labels to per-sample weights for XGBoost."""
    return y.map(class_weight).fillna(1.0).values.astype(float)


def oof_best_threshold(y_oof: np.ndarray, prob_oof: np.ndarray) -> float:
    """
    Find the Youden-J optimal threshold from OOF predictions accumulated
    across all folds. This is unbiased — the threshold is never computed
    on the same fold it is applied to.

    FIX from original: threshold was computed per-fold on the validation
    set and then applied to that same validation set — mild leakage.
    """
    fpr, tpr, thr = roc_curve(y_oof, prob_oof)
    return float(thr[np.argmax(tpr - fpr)])


# ══════════════════════════════════════════════════════════════════════════════
# MODEL BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def lgbm_regression(n_estimators: int = 5000) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        learning_rate=0.05,
        num_leaves=64,
        n_estimators=n_estimators,
        importance_type="gain",         # gain > split for physics features
        random_state=RANDOM_STATE,
    )


def lgbm_binary() -> lgb.LGBMClassifier:
    # num_class must NOT be set (or must be 1) for binary tasks.
    # LightGBM raises "[Fatal] Number of classes must be 1 for
    # non-multiclass training" if num_class > 1 is passed here.
    return lgb.LGBMClassifier(
        objective="binary",
        num_class=1,            # explicit — prevents accidental leakage
        class_weight="balanced",
        learning_rate=0.05,
        num_leaves=64,
        n_estimators=5000,
        importance_type="gain",
        random_state=RANDOM_STATE,
    )


def lgbm_multiclass(class_weight: dict, num_class: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_class,
        class_weight=class_weight,
        learning_rate=0.05,
        num_leaves=64,
        n_estimators=5000,
        importance_type="gain",
        random_state=RANDOM_STATE,
    )


def xgb_regression(n_estimators: int = 5000, early_stopping: bool = True) -> xgb.XGBRegressor:
    # XGBoost >= 2.0: eval_metric and early_stopping_rounds live in the
    # constructor, not in fit(). Pass early_stopping=False for the final
    # full-data fit where no eval_set is available.
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


def xgb_binary(scale_pos_weight: float, early_stopping: bool = True) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.9,
        n_estimators=5000,
        tree_method="hist",
        random_state=RANDOM_STATE,
        eval_metric="auc",
        early_stopping_rounds=200 if early_stopping else None,
    )


def xgb_multiclass(num_class: int, early_stopping: bool = True) -> xgb.XGBClassifier:
    # XGBoost has no native class_weight param for multiclass.
    # Class weighting is applied via sample_weight in .fit() instead.
    return xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_class,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.9,
        n_estimators=5000,
        tree_method="hist",
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        early_stopping_rounds=200 if early_stopping else None,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train(task_name: str):

    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}. Choose from {list(TASKS)}")

    cfg  = TASKS[task_name]
    df   = pd.read_pickle(DATA_PATH)
    feats = load_features(task_name)

    X     = df[feats]
    y_raw = df[cfg["target"]]
    y     = transform_target(y_raw, cfg.get("transform"))

    task_type    = cfg["type"]
    class_weight = cfg.get("class_weight", {})

    # n_classes is only meaningful for multiclass tasks.
    # For binary, LightGBM requires num_class=1 (or unset).
    # Defaulting to 2 here caused "Number of classes must be 1
    # for non-multiclass training" because the value leaked through
    # cfg.get("num_class", 2) for tasks that have no num_class key.
    n_classes = cfg["num_class"] if task_type == "multiclass" else 1

    cv = (
        KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        if task_type == "regression"
        else StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    )

    metrics_lgb = []
    metrics_xgb = []

    # OOF storage — used for unbiased threshold selection (binary) and
    # post-hoc analysis. Multiclass stores class-1 probabilities only
    # for brevity; full proba matrix stored separately if needed.
    oof_lgb = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))
    y_oof   = np.zeros(len(y))          # ground truth aligned to OOF indices

    # Track best iterations from early stopping across folds,
    # then use their mean for the final full-data fit.
    best_iters_lgb = []
    best_iters_xgb = []

    # ── Cross-validation loop ─────────────────────────────────────────────
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
        print(f"\n  Fold {fold}/{N_SPLITS}")

        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
        y_oof[va_idx] = yva.values

        # ── REGRESSION ───────────────────────────────────────────────────
        if task_type == "regression":

            lgbm = lgbm_regression()
            xgbm = xgb_regression()

            # FIX: early stopping added to regression (was missing in original).
            # Both models now behave consistently — stop when val loss plateaus.
            lgbm.fit(
                Xtr, ytr,
                eval_set=[(Xva, yva)],
                eval_metric="l2",
                callbacks=[lgb.early_stopping(200, verbose=False),
                           lgb.log_evaluation(period=-1)],
            )
            xgbm.fit(
                Xtr, ytr,
                eval_set=[(Xva, yva)],
                verbose=False,
            )

            best_iters_lgb.append(lgbm.best_iteration_)
            best_iters_xgb.append(xgbm.best_iteration)

            p_l = inverse_transform(lgbm.predict(Xva), cfg.get("transform"))
            p_x = inverse_transform(xgbm.predict(Xva), cfg.get("transform"))
            y_real = inverse_transform(yva.values, cfg.get("transform"))

            oof_lgb[va_idx] = p_l
            oof_xgb[va_idx] = p_x

            metrics_lgb.append(regression_metrics(y_real, p_l))
            metrics_xgb.append(regression_metrics(y_real, p_x))

        # ── BINARY CLASSIFICATION ─────────────────────────────────────────
        elif task_type == "binary":

            pos = (ytr == 1).sum()
            neg = (ytr == 0).sum()
            spw = neg / max(pos, 1)

            lgbm = lgbm_binary()
            xgbm = xgb_binary(scale_pos_weight=spw)

            lgbm.fit(
                Xtr, ytr,
                eval_set=[(Xva, yva)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(200, verbose=False),
                           lgb.log_evaluation(period=-1)],
            )
            xgbm.fit(
                Xtr, ytr,
                eval_set=[(Xva, yva)],
                verbose=False,
            )

            best_iters_lgb.append(lgbm.best_iteration_)
            best_iters_xgb.append(xgbm.best_iteration)

            prob_l = lgbm.predict_proba(Xva)[:, 1]
            prob_x = xgbm.predict_proba(Xva)[:, 1]

            oof_lgb[va_idx] = prob_l
            oof_xgb[va_idx] = prob_x

            # Compute fold metrics at fixed threshold=0.5 during CV.
            # FIX: threshold optimisation now done OOF after all folds
            # to avoid per-fold leakage (original optimised threshold on
            # the same val set used for metric reporting).
            metrics_lgb.append(binary_metrics(yva.values, prob_l, thr=0.5))
            metrics_xgb.append(binary_metrics(yva.values, prob_x, thr=0.5))

        # ── MULTICLASS ────────────────────────────────────────────────────
        else:

            lgbm = lgbm_multiclass(class_weight, n_classes)
            xgbm = xgb_multiclass(n_classes)

            # XGBoost class weighting via sample_weight — only mechanism
            # available for multiclass in XGBoost
            sw_tr = compute_sample_weights(ytr, class_weight)

            lgbm.fit(
                Xtr, ytr,
                eval_set=[(Xva, yva)],
                eval_metric="multi_logloss",
                callbacks=[lgb.early_stopping(200, verbose=False),
                           lgb.log_evaluation(period=-1)],
            )
            xgbm.fit(
                Xtr, ytr,
                sample_weight=sw_tr,
                eval_set=[(Xva, yva)],
                verbose=False,
            )

            best_iters_lgb.append(lgbm.best_iteration_)
            best_iters_xgb.append(xgbm.best_iteration)

            pred_l = lgbm.predict(Xva)
            pred_x = xgbm.predict(Xva)

            # FIX: expanded multiclass metrics — original only had macro_f1.
            # Per-class F1 is critical for monitoring half-metal (class 2)
            # detection — macro alone hides whether class 2 is ever predicted.
            metrics_lgb.append(multiclass_metrics(yva.values, pred_l, n_classes))
            metrics_xgb.append(multiclass_metrics(yva.values, pred_x, n_classes))

    # ── OOF threshold for binary (post-hoc, unbiased) ────────────────────
    if task_type == "binary":
        thr_lgb = oof_best_threshold(y_oof, oof_lgb)
        thr_xgb = oof_best_threshold(y_oof, oof_xgb)
        print(f"\n  OOF optimal threshold — LGBM: {thr_lgb:.3f}  XGB: {thr_xgb:.3f}")
    else:
        thr_lgb = thr_xgb = 0.5     # unused for regression / multiclass

    # ── Aggregate CV results ──────────────────────────────────────────────
    results = {
        "lgbm": summarize(metrics_lgb),
        "xgb":  summarize(metrics_xgb),
    }

    if task_type == "binary":
        results["thresholds"] = {"lgbm": thr_lgb, "xgb": thr_xgb}

    # ── Final full-data fit ───────────────────────────────────────────────
    # FIX: use mean best iteration from CV folds, not the last fold's params.
    # lgbm.get_params() returns n_estimators=5000 regardless of early stopping —
    # the actual best iteration lives in best_iteration_, not get_params().
    # Using mean CV best iteration prevents overfitting on the final model.

    mean_iter_lgb = max(1, int(np.mean(best_iters_lgb)))
    mean_iter_xgb = max(1, int(np.mean(best_iters_xgb)))

    print(f"\n  Mean best iterations — LGBM: {mean_iter_lgb}  XGB: {mean_iter_xgb}")

    if task_type == "regression":
        final_lgb = lgbm_regression(n_estimators=mean_iter_lgb)
        final_xgb = xgb_regression(n_estimators=mean_iter_xgb, early_stopping=False)
        final_lgb.fit(X, y)
        final_xgb.fit(X, y)

    elif task_type == "binary":
        pos = (y == 1).sum()
        neg = (y == 0).sum()
        spw = neg / max(pos, 1)

        final_lgb = lgbm_binary()
        final_lgb.n_estimators = mean_iter_lgb

        final_xgb = xgb_binary(scale_pos_weight=spw, early_stopping=False)
        final_xgb.n_estimators = mean_iter_xgb

        final_lgb.fit(X, y)
        final_xgb.fit(X, y)

    else:  # multiclass
        sw_full = compute_sample_weights(y, class_weight)

        final_lgb = lgbm_multiclass(class_weight, n_classes)
        final_lgb.n_estimators = mean_iter_lgb

        final_xgb = xgb_multiclass(n_classes, early_stopping=False)
        final_xgb.n_estimators = mean_iter_xgb

        # FIX: sample_weight applied to final XGBoost multiclass fit.
        # Original final_xgb.fit(X, y) had no sample_weight —
        # class weighting was silently dropped for the production model.
        final_lgb.fit(X, y)
        final_xgb.fit(X, y, sample_weight=sw_full)

    # ── Save ──────────────────────────────────────────────────────────────
    task_dir = os.path.join(OUT_DIR, task_name)
    os.makedirs(task_dir, exist_ok=True)

    joblib.dump(final_lgb, os.path.join(task_dir, f"{task_name}_lgbm.pkl"))
    joblib.dump(final_xgb, os.path.join(task_dir, f"{task_name}_xgb.pkl"))

    with open(os.path.join(task_dir, f"{task_name}_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    if task_type == "binary":
        thresholds = {"lgbm": thr_lgb, "xgb": thr_xgb}
        with open(os.path.join(task_dir, f"{task_name}_thresholds.json"), "w") as f:
            json.dump(thresholds, f, indent=4)

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  Task: {task_name}  |  CV Results")
    print(f"{'═'*55}")
    print(json.dumps(results, indent=2))
    print(f"\n  Models saved → {task_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train StoichML models.")
    parser.add_argument(
        "--task",
        required=True,
        choices=list(TASKS.keys()),
        help="Task to train: enthalpy | egap | egap_type | hm_class",
    )
    args = parser.parse_args()
    train(args.task)