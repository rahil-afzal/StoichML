#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_hm_class.py
─────────────────
StoichML — hm_class: One-shot ternary vs binary half-metal classifier.

Approach 1  One-shot ternary (baseline)
            Conductor / Insulator / Half-metal (3 classes)
            5-seed capped undersampling (majority classes capped at 3000)
            LGBM + XGBoost ensemble, softmax argmax at inference
            Single fixed operating point in precision-recall space

Approach 2  Binary half-metal detector
            Half-metal (1) vs not (0)
            5-seed strict 1:1 undersampling
            LGBM + XGBoost ensemble, tunable threshold
            Three operating points:
              Youden-J, F1-optimal, Precision-matched

Outputs  models/hm_class/
  oneshot_lgbm_seed{s}.pkl  x5
  oneshot_xgb_seed{s}.pkl   x5
  binary_lgbm_seed{s}.pkl   x5
  binary_xgb_seed{s}.pkl    x5
  thresholds.json
  oneshot_metrics.json
  binary_metrics.json
  hm_class_metrics.json
  images/
    oneshot_confusion.png
    binary_roc_pr.png
    binary_threshold_sweep.png
    binary_confusion_youden.png
    feature_importance_oneshot.png
    feature_importance_binary.png
    comparison_pr_curve.png
    comparison_bar.png

Usage:
    python -m scripts.model_hm
"""

from __future__ import annotations

import json
import os

import joblib
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

matplotlib.use("Agg")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

RANDOM_STATE      = 42
N_SPLITS          = 5
MAJORITY_CAP      = 3000
DATA_PATH         = "data/data_feat.pkl"
FEATURES_JSON     = "data/selected_features.json"
OUT_DIR           = "models/hm_class"
IMG_DIR           = os.path.join(OUT_DIR, "images")
UNDERSAMPLE_SEEDS = [0, 7, 21, 42, 99]
CLASS_NAMES       = {0: "Conductor", 1: "Insulator", 2: "Half-metal"}

NON_FEATURE_COLS = [
    "compound", "spacegroup_relax", "Egap", "Egap_type",
    "Egap_type_numeric", "enthalpy_formation_atom",
    "composition", "elements", "hm_class",
]

# ── Matplotlib style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "Times New Roman",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9,
    "figure.dpi":        300,
    "savefig.dpi":       300,
})
SAVEFIG_KW = dict(dpi=300, bbox_inches="tight")


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def load_features() -> list[str]:
    with open(FEATURES_JSON) as f:
        feats = json.load(f)
    task_feats = feats.get("hm_class", [])
    if not task_feats:
        raise ValueError(f"No features for 'hm_class' in {FEATURES_JSON}.")
    return task_feats


def save_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def summarize(metric_list: list[dict]) -> dict:
    keys = metric_list[0].keys()
    return {
        k: {"mean": float(np.mean([m[k] for m in metric_list])),
            "std":  float(np.std( [m[k] for m in metric_list]))}
        for k in keys
    }


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def binary_metrics(
    y_true: np.ndarray,
    prob:   np.ndarray,
    thr:    float,
) -> dict:
    pred = (prob >= thr).astype(int)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, pred, labels=[0, 1], zero_division=0
    )
    return {
        "threshold":        float(thr),
        "roc_auc":          float(roc_auc_score(y_true, prob)),
        "pr_auc":           float(average_precision_score(y_true, prob)),
        "macro_f1":         float(f1_score(y_true, pred,
                                           average="macro", zero_division=0)),
        "balanced_acc":     float(balanced_accuracy_score(y_true, pred)),
        "f1_class0":        float(f[0]),
        "f1_class1":        float(f[1]),
        "precision_class1": float(p[1]),
        "recall_class1":    float(r[1]),
    }


def three_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    return {
        "macro_f1":         float(f1_score(y_true, y_pred,
                                           average="macro", zero_division=0)),
        "weighted_f1":      float(f1_score(y_true, y_pred,
                                           average="weighted", zero_division=0)),
        "balanced_acc":     float(balanced_accuracy_score(y_true, y_pred)),
        "f1_class0":        float(f[0]),
        "precision_class0": float(p[0]),
        "recall_class0":    float(r[0]),
        "f1_class1":        float(f[1]),
        "precision_class1": float(p[1]),
        "recall_class1":    float(r[1]),
        "f1_class2":        float(f[2]),
        "precision_class2": float(p[2]),
        "recall_class2":    float(r[2]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# THRESHOLD SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def youden_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, prob)
    return float(thr[np.argmax(tpr - fpr)])


def f1_optimal_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    thresholds = np.linspace(0.01, 0.99, 500)
    best_f1, best_thr = 0.0, 0.5
    for t in thresholds:
        f1 = f1_score(y_true, (prob >= t).astype(int),
                      pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    return float(best_thr)


def precision_matched_threshold(
    y_true:           np.ndarray,
    prob:             np.ndarray,
    target_precision: float,
) -> float:
    prec, _, thr = precision_recall_curve(y_true, prob)
    valid = prec[:-1] >= target_precision
    return float(thr[valid][0]) if valid.any() else float(thr[-1])


# ══════════════════════════════════════════════════════════════════════════════
# UNDERSAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def undersample_strict(
    X: pd.DataFrame, y: pd.Series,
    minority_class: int, seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    rng     = np.random.default_rng(seed)
    idx_min = list(y[y == minority_class].index)
    n_min   = len(idx_min)
    all_idx = list(idx_min)
    for cls in sorted(y.unique()):
        if cls == minority_class:
            continue
        idx_cls = y[y == cls].index
        all_idx.extend(
            rng.choice(idx_cls, size=min(n_min, len(idx_cls)), replace=False)
        )
    all_idx = np.array(all_idx)
    rng.shuffle(all_idx)
    return X.loc[all_idx], y.loc[all_idx]


def undersample_capped(
    X: pd.DataFrame, y: pd.Series,
    minority_class: int, majority_cap: int, seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    rng     = np.random.default_rng(seed)
    idx_min = list(y[y == minority_class].index)
    all_idx = list(idx_min)
    for cls in sorted(y.unique()):
        if cls == minority_class:
            continue
        idx_cls = y[y == cls].index
        all_idx.extend(
            rng.choice(idx_cls,
                       size=min(majority_cap, len(idx_cls)), replace=False)
        )
    all_idx = np.array(all_idx)
    rng.shuffle(all_idx)
    return X.loc[all_idx], y.loc[all_idx]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_lgbm_binary(n_estimators: int = 5000) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", learning_rate=0.05, num_leaves=64,
        n_estimators=n_estimators, min_data_in_leaf=20,
        subsample=0.8, colsample_bytree=0.8,
        importance_type="gain", random_state=RANDOM_STATE, verbose=-1,
    )


def build_xgb_binary(
    n_estimators: int = 5000, early_stopping: bool = True,
) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        learning_rate=0.05, max_depth=6, min_child_weight=3,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.05, reg_lambda=1.5,
        n_estimators=n_estimators, tree_method="hist",
        eval_metric="auc", random_state=RANDOM_STATE,
        early_stopping_rounds=200 if early_stopping else None,
    )


def build_lgbm_multiclass(
    n_classes: int, n_estimators: int = 5000,
) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass", num_class=n_classes,
        learning_rate=0.05, num_leaves=64,
        n_estimators=n_estimators, min_data_in_leaf=10,
        subsample=0.8, colsample_bytree=0.8,
        importance_type="gain", random_state=RANDOM_STATE, verbose=-1,
    )


def build_xgb_multiclass(
    n_estimators: int = 5000, early_stopping: bool = True,
) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="multi:softprob", num_class=3,
        learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        n_estimators=n_estimators, tree_method="hist",
        eval_metric="mlogloss", random_state=RANDOM_STATE,
        early_stopping_rounds=200 if early_stopping else None,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, **SAVEFIG_KW)
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_confusion(
    y_true: np.ndarray, y_pred: np.ndarray,
    labels: list[str], title: str, fname: str,
) -> None:
    cm      = confusion_matrix(y_true, y_pred,
                               labels=list(range(len(labels))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontweight="bold")
    for ax, data, fmt, t in [
        (axes[0], cm_norm, ".2f", "Row-normalised"),
        (axes[1], cm,      "d",   "Raw counts"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=labels, yticklabels=labels,
                    linewidths=0.5, ax=ax, annot_kws={"size": 10})
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(t)
    plt.tight_layout()
    _save(fig, fname)


def plot_roc_pr(
    y_true: np.ndarray, prob: np.ndarray,
    pos_label_name: str, title: str, fname: str,
) -> None:
    fpr, tpr, _  = roc_curve(y_true, prob)
    prec, rec, _ = precision_recall_curve(y_true, prob)
    auc_roc      = roc_auc_score(y_true, prob)
    auc_pr       = average_precision_score(y_true, prob)
    prevalence   = y_true.mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontweight="bold")

    axes[0].plot(fpr, tpr, lw=2, color="steelblue",
                 label=f"ROC-AUC = {auc_roc:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    axes[1].plot(rec, prec, lw=2, color="steelblue",
                 label=f"PR-AUC = {auc_pr:.4f}")
    axes[1].axhline(prevalence, color="k", lw=1, ls="--",
                    label=f"Random baseline ({prevalence:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel(f"Precision ({pos_label_name})")
    axes[1].set_title("Precision–Recall Curve")
    axes[1].legend()

    plt.tight_layout()
    _save(fig, fname)


def plot_threshold_sweep(
    y_true: np.ndarray, prob: np.ndarray,
    thresholds_dict: dict, pos_label_name: str, fname: str,
) -> None:
    ts = np.linspace(0.01, 0.99, 300)
    f1s, precs, recs = [], [], []
    for t in ts:
        pred = (prob >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            y_true, pred, labels=[0, 1], zero_division=0
        )
        f1s.append(f[1]); precs.append(p[1]); recs.append(r[1])

    style = {
        "youden":     ("black",    "Youden-J"),
        "f1_opt":     ("seagreen", "F1-optimal"),
        "prec_match": ("tomato",   "Precision-matched"),
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ts, f1s,   lw=2, color="steelblue",
            label=f"F1 ({pos_label_name})")
    ax.plot(ts, precs, lw=2, color="tomato",    alpha=0.7,
            label=f"Precision ({pos_label_name})")
    ax.plot(ts, recs,  lw=2, color="seagreen",  alpha=0.7,
            label=f"Recall ({pos_label_name})")
    for key, thr in thresholds_dict.items():
        color, label = style[key]
        ax.axvline(thr, color=color, lw=1.5, ls="--",
                   label=f"{label} = {thr:.3f}")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Binary Classifier — Threshold Sweep ({pos_label_name})",
                 fontweight="bold")
    ax.legend(loc="center left")
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.grid(which="major", ls="--", alpha=0.4)
    plt.tight_layout()
    _save(fig, fname)


def plot_pr_comparison(
    y_true:         np.ndarray,
    prob:           np.ndarray,
    oneshot_m:      dict,
    thresholds:     dict,
    fname:          str,
) -> None:
    """
    Key figure: PR curve for binary classifier vs fixed one-shot point.
    Demonstrates that the binary classifier exposes a tunable frontier
    unavailable to the softmax formulation.
    """
    prec, rec, _ = precision_recall_curve(y_true, prob)
    auc_pr       = average_precision_score(y_true, prob)

    op_style = {
        "youden":     ("black",    "Youden-J"),
        "f1_opt":     ("seagreen", "F1-optimal"),
        "prec_match": ("tomato",   "Precision-matched"),
    }

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, lw=2.5, color="steelblue", zorder=2,
            label=f"Binary classifier (PR-AUC = {auc_pr:.4f})")
    ax.axhline(y_true.mean(), color="grey", lw=1, ls="--",
               label=f"Random (prevalence = {y_true.mean():.3f})")

    # One-shot fixed point
    ax.scatter(
        [oneshot_m["recall_class2"]],
        [oneshot_m["precision_class2"]],
        marker="*", s=350, color="orange", zorder=5,
        label=(f"One-shot  "
               f"P={oneshot_m['precision_class2']:.3f}  "
               f"R={oneshot_m['recall_class2']:.3f}"),
    )

    # Binary operating points
    for key, thr in thresholds.items():
        pred = (prob >= thr).astype(int)
        p_arr, r_arr, f_arr, _ = precision_recall_fscore_support(
            y_true, pred, labels=[0, 1], zero_division=0
        )
        color, label = op_style[key]
        ax.scatter(
            [r_arr[1]], [p_arr[1]],
            marker="o", s=120, color=color, zorder=5,
            label=(f"{label}  "
                   f"P={p_arr[1]:.3f}  R={r_arr[1]:.3f}  "
                   f"F1={f_arr[1]:.3f}"),
        )

    ax.set_xlabel("Recall (Half-metal)")
    ax.set_ylabel("Precision (Half-metal)")
    ax.set_title(
        "One-Shot vs Binary Classifier\nPrecision–Recall Space",
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim([0, 1.02]); ax.set_ylim([0, 1.05])
    ax.grid(ls="--", alpha=0.3)
    plt.tight_layout()
    _save(fig, fname)


def plot_comparison_bar(
    comparison_rows: list[dict], fname: str,
) -> None:
    labels = [r["model"] for r in comparison_rows]
    prec   = [r["precision_hm"] for r in comparison_rows]
    rec    = [r["recall_hm"]    for r in comparison_rows]
    f1     = [r["f1_hm"]        for r in comparison_rows]
    x      = np.arange(len(labels))
    w      = 0.25

    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.8), 5))
    ax.bar(x - w, prec, w, label="Precision", color="tomato",    alpha=0.85)
    ax.bar(x,     rec,  w, label="Recall",    color="steelblue", alpha=0.85)
    ax.bar(x + w, f1,   w, label="F1",        color="seagreen",  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Score"); ax.set_ylim([0, 1.05])
    ax.set_title("Half-metal Detection — Model Comparison\n"
                 "(Precision / Recall / F1 for half-metal class)",
                 fontweight="bold")
    ax.legend(); ax.grid(axis="y", ls="--", alpha=0.4)
    plt.tight_layout()
    _save(fig, fname)


def plot_feature_importance(
    importances:   list[np.ndarray],
    feature_names: list[str],
    title:         str,
    fname:         str,
    top_n:         int = 20,
) -> None:
    mean_imp = np.mean(importances, axis=0)
    std_imp  = np.std(importances,  axis=0)
    idx      = np.argsort(mean_imp)[-top_n:]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.30)))
    y_pos   = np.arange(top_n)
    ax.barh(y_pos, mean_imp[idx], xerr=std_imp[idx],
            color="steelblue", ecolor="grey",
            alpha=0.85, height=0.7, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel("Mean Gain Importance (± std)")
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    _save(fig, fname)


# ══════════════════════════════════════════════════════════════════════════════
# APPROACH 1 — ONE-SHOT TERNARY
# ══════════════════════════════════════════════════════════════════════════════

def train_oneshot(
    X: pd.DataFrame, y: pd.Series, feature_names: list[str],
) -> dict:
    print(f"\n{'═'*60}")
    print(f"  APPROACH 1 — One-Shot Ternary (Baseline)")
    print(f"{'═'*60}")
    for cls, label in CLASS_NAMES.items():
        print(f"  Class {cls} ({label}): {(y==cls).sum()}")

    cv    = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                            random_state=RANDOM_STATE)
    folds = list(cv.split(X, y))

    oof_proba_seeds: dict[int, np.ndarray] = {}
    seed_cv:         dict[int, list[dict]] = {}
    best_iters_lgb:  dict[int, list[int]]  = {}
    best_iters_xgb:  dict[int, list[int]]  = {}
    final_lgb:       dict[int, lgb.LGBMClassifier] = {}
    final_xgb:       dict[int, xgb.XGBClassifier]  = {}
    lgb_imps: list[np.ndarray] = []
    y_oof = np.full(len(y), -1, dtype=int)

    for seed in UNDERSAMPLE_SEEDS:
        print(f"\n  Seed {seed}  {'─'*45}")
        oof_p = np.zeros((len(y), 3))
        folds_m, il, ix = [], [], []

        for fold, (tr_idx, va_idx) in enumerate(folds, 1):
            Xva, yva = X.iloc[va_idx], y.iloc[va_idx]
            if seed == UNDERSAMPLE_SEEDS[0]:
                y_oof[va_idx] = yva.values

            Xb, yb = undersample_capped(
                X.iloc[tr_idx], y.iloc[tr_idx],
                minority_class=2, majority_cap=MAJORITY_CAP, seed=seed,
            )
            lgbm = build_lgbm_multiclass(3)
            xgbm = build_xgb_multiclass(early_stopping=True)
            lgbm.fit(Xb, yb, eval_set=[(Xva, yva)],
                     eval_metric="multi_logloss",
                     callbacks=[lgb.early_stopping(200, verbose=False),
                                lgb.log_evaluation(period=-1)])
            xgbm.fit(Xb, yb, eval_set=[(Xva, yva)], verbose=False)

            il.append(lgbm.best_iteration_)
            ix.append(xgbm.best_iteration)
            lgb_imps.append(lgbm.feature_importances_)

            p_ens = (lgbm.predict_proba(Xva) +
                     xgbm.predict_proba(Xva)) / 2
            oof_p[va_idx] += p_ens

            fm = three_class_metrics(yva.values, np.argmax(p_ens, axis=1))
            folds_m.append(fm)
            print(f"  fold {fold}  lgbm={lgbm.best_iteration_:4d}  "
                  f"xgb={xgbm.best_iteration:4d}  "
                  f"macro_F1={fm['macro_f1']:.4f}  "
                  f"F1_hm={fm['f1_class2']:.4f}  "
                  f"P_hm={fm['precision_class2']:.4f}  "
                  f"R_hm={fm['recall_class2']:.4f}")

        oof_proba_seeds[seed] = oof_p
        seed_cv[seed]        = folds_m
        best_iters_lgb[seed] = il
        best_iters_xgb[seed] = ix

    # Ensemble OOF
    ens_proba = np.mean(list(oof_proba_seeds.values()), axis=0)
    ens_pred  = np.argmax(ens_proba, axis=1)
    ens_m     = three_class_metrics(y_oof, ens_pred)
    cm        = confusion_matrix(y_oof, ens_pred, labels=[0, 1, 2])

    print(f"\n  One-shot OOF: macro_F1={ens_m['macro_f1']:.4f}  "
          f"F1_hm={ens_m['f1_class2']:.4f}  "
          f"P_hm={ens_m['precision_class2']:.4f}  "
          f"R_hm={ens_m['recall_class2']:.4f}")
    print(f"  Confusion:\n{cm}")

    # Final fit
    for seed in UNDERSAMPLE_SEEDS:
        ml = max(1, int(np.mean(best_iters_lgb[seed])))
        mx = max(1, int(np.mean(best_iters_xgb[seed])))
        Xb, yb = undersample_capped(
            X, y, minority_class=2, majority_cap=MAJORITY_CAP, seed=seed,
        )
        m_lgb = build_lgbm_multiclass(3, n_estimators=ml)
        m_xgb = build_xgb_multiclass(n_estimators=mx, early_stopping=False)
        m_lgb.fit(Xb, yb); m_xgb.fit(Xb, yb)
        final_lgb[seed] = m_lgb
        final_xgb[seed] = m_xgb

    plot_confusion(
        y_oof, ens_pred,
        [CLASS_NAMES[i] for i in range(3)],
        "One-Shot Ternary — Confusion Matrix (Ensemble OOF)",
        os.path.join(IMG_DIR, "oneshot_confusion.png"),
    )
    plot_feature_importance(
        lgb_imps, feature_names,
        "One-Shot LGBM — Feature Importance (Top 20)",
        os.path.join(IMG_DIR, "feature_importance_oneshot.png"),
    )

    return dict(
        final_lgb    = final_lgb,
        final_xgb    = final_xgb,
        oof_proba    = ens_proba,
        oof_pred     = ens_pred,
        oof_true     = y_oof,
        oof_metrics  = ens_m,
        seed_cv      = seed_cv,
        confusion    = cm.tolist(),
    )


# ══════════════════════════════════════════════════════════════════════════════
# APPROACH 2 — BINARY HALF-METAL DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

def train_binary(
    X: pd.DataFrame, y_3class: pd.Series,
    feature_names: list[str], oneshot_precision: float,
) -> dict:
    """
    5-seed 1:1 undersampling binary classifier.
    Half-metal=1, everything else=0.
    Three thresholds selected from OOF probabilities.
    """
    y_bin = (y_3class == 2).astype(int)
    y_bin.index = y_3class.index

    print(f"\n{'═'*60}")
    print(f"  APPROACH 2 — Binary Half-metal Detector")
    print(f"{'═'*60}")
    print(f"  Half-metal: {(y_bin==1).sum()}   "
          f"Not: {(y_bin==0).sum()}   "
          f"Ratio {(y_bin==0).sum()/(y_bin==1).sum():.1f}:1")
    print(f"  Precision target (one-shot match): {oneshot_precision:.4f}")

    cv    = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                            random_state=RANDOM_STATE)
    folds = list(cv.split(X, y_bin))

    oof_lgb_seeds: dict[int, np.ndarray] = {}
    oof_xgb_seeds: dict[int, np.ndarray] = {}
    seed_cv:       dict[int, list[dict]] = {}
    best_iters_lgb: dict[int, list[int]] = {}
    best_iters_xgb: dict[int, list[int]] = {}
    final_lgb:      dict[int, lgb.LGBMClassifier] = {}
    final_xgb:      dict[int, xgb.XGBClassifier]  = {}
    lgb_imps: list[np.ndarray] = []
    y_oof = np.zeros(len(y_bin))

    for seed in UNDERSAMPLE_SEEDS:
        print(f"\n  Seed {seed}  {'─'*45}")
        oof_l = np.zeros(len(y_bin))
        oof_x = np.zeros(len(y_bin))
        folds_m, il, ix = [], [], []

        for fold, (tr_idx, va_idx) in enumerate(folds, 1):
            Xva, yva = X.iloc[va_idx], y_bin.iloc[va_idx]
            if seed == UNDERSAMPLE_SEEDS[0]:
                y_oof[va_idx] = yva.values

            Xb, yb = undersample_strict(
                X.iloc[tr_idx], y_bin.iloc[tr_idx],
                minority_class=1, seed=seed,
            )
            lgbm = build_lgbm_binary()
            xgbm = build_xgb_binary(early_stopping=True)
            lgbm.fit(Xb, yb, eval_set=[(Xva, yva)],
                     eval_metric="auc",
                     callbacks=[lgb.early_stopping(200, verbose=False),
                                lgb.log_evaluation(period=-1)])
            xgbm.fit(Xb, yb, eval_set=[(Xva, yva)], verbose=False)

            il.append(lgbm.best_iteration_)
            ix.append(xgbm.best_iteration)
            lgb_imps.append(lgbm.feature_importances_)

            oof_l[va_idx] = lgbm.predict_proba(Xva)[:, 1]
            oof_x[va_idx] = xgbm.predict_proba(Xva)[:, 1]
            p_ens = (oof_l[va_idx] + oof_x[va_idx]) / 2
            fm = binary_metrics(yva.values, p_ens, thr=0.5)
            folds_m.append(fm)
            print(f"  fold {fold}  lgbm={lgbm.best_iteration_:4d}  "
                  f"xgb={xgbm.best_iteration:4d}  "
                  f"AUC={fm['roc_auc']:.4f}  "
                  f"P={fm['precision_class1']:.4f}  "
                  f"R={fm['recall_class1']:.4f}")

        oof_lgb_seeds[seed] = oof_l
        oof_xgb_seeds[seed] = oof_x
        seed_cv[seed]       = folds_m
        best_iters_lgb[seed] = il
        best_iters_xgb[seed] = ix

    # Ensemble OOF (all seeds × both models)
    all_probs = (list(oof_lgb_seeds.values()) +
                 list(oof_xgb_seeds.values()))
    ens_prob  = np.mean(all_probs, axis=0)

    # Three thresholds
    thresholds = {
        "youden":     youden_threshold(y_oof, ens_prob),
        "f1_opt":     f1_optimal_threshold(y_oof, ens_prob),
        "prec_match": precision_matched_threshold(
                          y_oof, ens_prob, oneshot_precision),
    }

    print(f"\n  Binary OOF  ROC-AUC={roc_auc_score(y_oof, ens_prob):.4f}  "
          f"PR-AUC={average_precision_score(y_oof, ens_prob):.4f}")

    metrics_per_thr = {}
    for name, thr in thresholds.items():
        m = binary_metrics(y_oof, ens_prob, thr)
        metrics_per_thr[name] = m
        print(f"  {name:<12} thr={thr:.3f}  "
              f"P={m['precision_class1']:.4f}  "
              f"R={m['recall_class1']:.4f}  "
              f"F1={m['f1_class1']:.4f}")

    # Final fit
    for seed in UNDERSAMPLE_SEEDS:
        ml = max(1, int(np.mean(best_iters_lgb[seed])))
        mx = max(1, int(np.mean(best_iters_xgb[seed])))
        Xb, yb = undersample_strict(X, y_bin, minority_class=1, seed=seed)
        m_lgb = build_lgbm_binary(n_estimators=ml)
        m_xgb = build_xgb_binary(n_estimators=mx, early_stopping=False)
        m_lgb.fit(Xb, yb); m_xgb.fit(Xb, yb)
        final_lgb[seed] = m_lgb
        final_xgb[seed] = m_xgb

    # Figures
    plot_roc_pr(
        y_oof, ens_prob, "Half-metal",
        "Binary Half-metal Detector — ROC and PR Curves (Ensemble OOF)",
        os.path.join(IMG_DIR, "binary_roc_pr.png"),
    )
    plot_threshold_sweep(
        y_oof, ens_prob, thresholds, "Half-metal",
        os.path.join(IMG_DIR, "binary_threshold_sweep.png"),
    )
    plot_confusion(
        y_oof, (ens_prob >= thresholds["youden"]).astype(int),
        ["Not half-metal", "Half-metal"],
        "Binary Classifier — Confusion Matrix (Youden-J, Ensemble OOF)",
        os.path.join(IMG_DIR, "binary_confusion_youden.png"),
    )
    plot_feature_importance(
        lgb_imps, feature_names,
        "Binary LGBM — Feature Importance (Top 20)",
        os.path.join(IMG_DIR, "feature_importance_binary.png"),
    )

    return dict(
        final_lgb         = final_lgb,
        final_xgb         = final_xgb,
        oof_prob          = ens_prob,
        oof_true          = y_oof,
        thresholds        = thresholds,
        seed_cv           = seed_cv,
        metrics_per_thr   = metrics_per_thr,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    df    = pd.read_pickle(DATA_PATH)
    feats = load_features()
    X     = df.drop(columns=[c for c in NON_FEATURE_COLS
                              if c in df.columns], errors="ignore")[feats]
    y     = df["hm_class"]

    print(f"\n{'═'*60}")
    print(f"  hm_class — One-Shot vs Binary Comparison")
    print(f"{'═'*60}")
    print(f"  Total: {len(y)}    Features: {len(feats)}")
    for cls, label in CLASS_NAMES.items():
        print(f"  Class {cls} ({label}): {(y==cls).sum()}")

    # ── Approach 1 ────────────────────────────────────────────────────────
    os_result  = train_oneshot(X, y, feats)
    oneshot_p  = os_result["oof_metrics"]["precision_class2"]

    # ── Approach 2 ────────────────────────────────────────────────────────
    bin_result = train_binary(X, y, feats, oneshot_precision=oneshot_p)

    # ── Comparison ────────────────────────────────────────────────────────
    thr_labels = {
        "youden":     "Youden-J",
        "f1_opt":     "F1-optimal",
        "prec_match": "Precision-matched",
    }
    comparison_rows = [{
        "model":        "One-shot (softmax)",
        "precision_hm": os_result["oof_metrics"]["precision_class2"],
        "recall_hm":    os_result["oof_metrics"]["recall_class2"],
        "f1_hm":        os_result["oof_metrics"]["f1_class2"],
        "macro_f1":     os_result["oof_metrics"]["macro_f1"],
    }]
    for key, label in thr_labels.items():
        m = bin_result["metrics_per_thr"][key]
        comparison_rows.append({
            "model":        f"Binary ({label})",
            "precision_hm": m["precision_class1"],
            "recall_hm":    m["recall_class1"],
            "f1_hm":        m["f1_class1"],
            "macro_f1":     m["macro_f1"],
        })

    print(f"\n{'═'*70}")
    print(f"  {'Model':<32} {'P_hm':>7} {'R_hm':>7} "
          f"{'F1_hm':>7} {'Macro_F1':>9}")
    print(f"  {'─'*64}")
    for row in comparison_rows:
        print(f"  {row['model']:<32} "
              f"{row['precision_hm']:>7.4f} {row['recall_hm']:>7.4f} "
              f"{row['f1_hm']:>7.4f} {row['macro_f1']:>9.4f}")
    print(f"{'═'*70}")

    plot_pr_comparison(
        bin_result["oof_true"], bin_result["oof_prob"],
        os_result["oof_metrics"], bin_result["thresholds"],
        os.path.join(IMG_DIR, "comparison_pr_curve.png"),
    )
    plot_comparison_bar(
        comparison_rows,
        os.path.join(IMG_DIR, "comparison_bar.png"),
    )

    # ── Save models ───────────────────────────────────────────────────────
    for seed in UNDERSAMPLE_SEEDS:
        joblib.dump(os_result["final_lgb"][seed],
                    os.path.join(OUT_DIR, f"oneshot_lgbm_seed{seed}.pkl"))
        joblib.dump(os_result["final_xgb"][seed],
                    os.path.join(OUT_DIR, f"oneshot_xgb_seed{seed}.pkl"))
        joblib.dump(bin_result["final_lgb"][seed],
                    os.path.join(OUT_DIR, f"binary_lgbm_seed{seed}.pkl"))
        joblib.dump(bin_result["final_xgb"][seed],
                    os.path.join(OUT_DIR, f"binary_xgb_seed{seed}.pkl"))

    # ── Save metrics ──────────────────────────────────────────────────────
    save_json(bin_result["thresholds"],
              os.path.join(OUT_DIR, "thresholds.json"))
    save_json({
        "oof_metrics": os_result["oof_metrics"],
        "confusion":   os_result["confusion"],
        "seed_cv":     {str(s): summarize(os_result["seed_cv"][s])
                        for s in UNDERSAMPLE_SEEDS},
    }, os.path.join(OUT_DIR, "oneshot_metrics.json"))
    save_json({
        "metrics_per_threshold": bin_result["metrics_per_thr"],
        "thresholds":            bin_result["thresholds"],
        "seed_cv":               {str(s): summarize(bin_result["seed_cv"][s])
                                  for s in UNDERSAMPLE_SEEDS},
    }, os.path.join(OUT_DIR, "binary_metrics.json"))
    save_json({
        "comparison": comparison_rows,
        "oneshot":    os_result["oof_metrics"],
        "binary":     bin_result["metrics_per_thr"],
        "n_features": len(feats),
    }, os.path.join(OUT_DIR, "hm_class_metrics.json"))

    print(f"\n{'═'*60}")
    print(f"  Done.  All outputs in: {OUT_DIR}/")
    print(f"{'═'*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def predict_hm(
    X_new:         pd.DataFrame,
    model_dir:     str = OUT_DIR,
    threshold_key: str = "f1_opt",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Binary inference.
    threshold_key: 'youden' | 'f1_opt' | 'prec_match'
    Returns (pred_binary, proba_halfmetal).
    """
    thr_all = json.load(open(os.path.join(model_dir, "thresholds.json")))
    thr     = thr_all[threshold_key]

    models = [
        joblib.load(os.path.join(model_dir, f"binary_{m}_seed{s}.pkl"))
        for m in ("lgbm", "xgb") for s in UNDERSAMPLE_SEEDS
    ]
    proba_hm = np.mean(
        [m.predict_proba(X_new)[:, 1] for m in models], axis=0
    )
    return (proba_hm >= thr).astype(int), proba_hm


if __name__ == "__main__":
    main()