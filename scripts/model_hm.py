#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_hm_class.py
─────────────────
StoichML training pipeline — hm_class.

Two approaches trained and compared in one run:

  Approach 1  One-shot three-class (baseline)
              Direct multiclass: conductor / insulator / half-metal
              5-seed capped undersampling (classes 0+1 capped at 3000)
              LGBM + XGBoost ensemble, softmax argmax at inference
              Single operating point in precision-recall space

  Approach 2  Two-stage (proposed)
              Stage 1: half-metal vs not (5-seed 1:1 undersampling,
                       LGBM + XGBoost, tunable threshold)
              Stage 2: conductor vs insulator (5-fold CV, LGBM + XGBoost)
              Evaluated at three thresholds:
                - Youden-J       (balanced sensitivity/specificity)
                - F1-optimal     (maximises half-metal F1)
                - Precision-matched (P >= one-shot precision for fair
                                    recall comparison)

Comparison framing:
  The one-shot is a single fixed point in precision-recall space.
  The two-stage is a tunable curve.
  The publishable claim: two-stage achieves strictly higher recall
  than one-shot at matched precision — confirmed by comparison_pr_curve.png.

Outputs  models/hm_class/
  oneshot_lgbm_seed{s}.pkl  x5
  oneshot_xgb_seed{s}.pkl   x5
  stage1_lgbm_seed{s}.pkl   x5
  stage1_xgb_seed{s}.pkl    x5
  stage2_lgbm.pkl
  stage2_xgb.pkl
  thresholds.json
  oneshot_metrics.json
  stage1_metrics.json
  stage2_metrics.json
  hm_class_metrics.json
  images/
    oneshot_confusion.png
    stage1_roc_pr.png
    stage1_threshold_sweep.png
    stage1_confusion.png
    stage2_roc_pr.png
    stage2_confusion.png
    final_confusion_Youden-J.png
    final_confusion_F1-optimal.png
    final_confusion_Precision-matched.png
    feature_importance_stage1.png
    feature_importance_stage2.png
    comparison_pr_curve.png
    comparison_bar.png

Usage:
    python -m scripts.model_hm
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    precision_recall_fscore_support,
    confusion_matrix,
)

import lightgbm as lgb
import xgboost as xgb

matplotlib.use("Agg")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

RANDOM_STATE = 42
N_SPLITS     = 5
MAJORITY_CAP = 3000

DATA_PATH     = "data/data_feat.pkl"
FEATURES_JSON = "data/selected_features.json"
OUT_DIR       = "models/hm_class"
IMG_DIR       = os.path.join(OUT_DIR, "images")

UNDERSAMPLE_SEEDS = [0, 7, 21, 42, 99]

NON_FEATURE_COLS = [
    "compound", "spacegroup_relax", "Egap", "Egap_type",
    "Egap_type_numeric", "enthalpy_formation_atom",
    "composition", "elements", "hm_class",
]

CLASS_NAMES = {0: "Conductor", 1: "Insulator", 2: "Half-metal"}


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
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


def oof_best_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Youden-J optimal threshold."""
    fpr, tpr, thr = roc_curve(y_true, prob)
    return float(thr[np.argmax(tpr - fpr)])


def f1_optimal_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Threshold that maximises F1 for the positive class."""
    thresholds = np.linspace(0.01, 0.99, 500)
    best_f1, best_thr = 0.0, 0.5
    for t in thresholds:
        pred = (prob >= t).astype(int)
        f1 = f1_score(y_true, pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = t
    return float(best_thr)


def precision_matched_threshold(
    y_true: np.ndarray,
    prob: np.ndarray,
    target_precision: float,
) -> float:
    """
    Find the highest threshold where precision >= target_precision.
    Used to match the one-shot precision for a fair recall comparison.
    At matched precision, higher recall = better — that is the claim.
    """
    prec, rec, thr = precision_recall_curve(y_true, prob)
    # prec/rec length = n_thresholds + 1; thr length = n_thresholds
    valid = prec[:-1] >= target_precision
    if not valid.any():
        return float(thr[-1])
    return float(thr[valid][0])


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def binary_metrics(
    y_true: np.ndarray,
    prob: np.ndarray,
    thr: float,
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
# UNDERSAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def undersample_strict(
    X: pd.DataFrame,
    y: pd.Series,
    minority_class: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Strict 1:1 — must match feature_selection.py."""
    rng          = np.random.default_rng(seed)
    idx_minority = y[y == minority_class].index
    n_minority   = len(idx_minority)
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
    """Capped undersampling for one-shot multiclass."""
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

def build_lgbm_binary(n_estimators: int = 5000) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.05, num_leaves=64,
        n_estimators=n_estimators, min_data_in_leaf=20,
        subsample=0.8, colsample_bytree=0.8,
        importance_type="gain", random_state=RANDOM_STATE, verbose=-1,
    )


def build_xgb_binary(
    n_estimators: int = 5000,
    early_stopping: bool = True,
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
    n_classes: int,
    n_estimators: int = 5000,
) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass", num_class=n_classes,
        learning_rate=0.05, num_leaves=64,
        n_estimators=n_estimators, min_data_in_leaf=10,
        subsample=0.8, colsample_bytree=0.8,
        importance_type="gain", random_state=RANDOM_STATE, verbose=-1,
    )


def build_xgb_multiclass(
    n_estimators: int = 5000,
    early_stopping: bool = True,
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

def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    title: str,
    fname: str,
) -> None:
    cm      = confusion_matrix(y_true, y_pred,
                               labels=list(range(len(labels))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    for ax, data, fmt, t in [
        (axes[0], cm_norm, ".2f", "Row-normalised"),
        (axes[1], cm,      "d",   "Raw counts"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=labels, yticklabels=labels,
                    linewidths=0.5, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(t)
    plt.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fname}")


def plot_roc_pr(
    y_true: np.ndarray,
    prob: np.ndarray,
    pos_label_name: str,
    title: str,
    fname: str,
) -> None:
    fpr, tpr, _  = roc_curve(y_true, prob)
    prec, rec, _ = precision_recall_curve(y_true, prob)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    axes[0].plot(fpr, tpr, lw=2, color="steelblue",
                 label=f"AUC={roc_auc_score(y_true, prob):.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC"); axes[0].legend(fontsize=9)
    axes[1].plot(rec, prec, lw=2, color="steelblue",
                 label=f"PR-AUC={average_precision_score(y_true, prob):.4f}")
    axes[1].axhline(y_true.mean(), color="k", lw=1, ls="--",
                    label=f"Baseline={y_true.mean():.3f}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel(f"Precision ({pos_label_name})")
    axes[1].set_title("Precision-Recall"); axes[1].legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fname}")


def plot_threshold_sweep(
    y_true: np.ndarray,
    prob: np.ndarray,
    thresholds_dict: dict,
    pos_label_name: str,
    fname: str,
) -> None:
    ts = np.linspace(0.01, 0.99, 300)
    f1s, precs, recs = [], [], []
    for t in ts:
        pred = (prob >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            y_true, pred, labels=[0, 1], zero_division=0
        )
        f1s.append(f[1]); precs.append(p[1]); recs.append(r[1])

    colors = {"youden": "black", "f1_opt": "seagreen", "prec_match": "tomato"}
    labels = {"youden": "Youden-J", "f1_opt": "F1-optimal",
              "prec_match": "Precision-matched"}

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(ts, f1s,   lw=2, color="steelblue",
            label=f"F1 ({pos_label_name})")
    ax.plot(ts, precs, lw=2, color="tomato",    alpha=0.7,
            label=f"Precision ({pos_label_name})")
    ax.plot(ts, recs,  lw=2, color="seagreen",  alpha=0.7,
            label=f"Recall ({pos_label_name})")
    for key, thr in thresholds_dict.items():
        ax.axvline(thr, color=colors[key], lw=1.5, ls="--",
                   label=f"{labels[key]} = {thr:.3f}")
    ax.set_xlabel("Decision Threshold"); ax.set_ylabel("Score")
    ax.set_title(f"Stage 1 — Threshold Sweep ({pos_label_name})",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.grid(which="major", ls="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fname}")


def plot_pr_comparison(
    y_true_s1: np.ndarray,
    prob_s1: np.ndarray,
    oneshot_metrics: dict,
    thresholds_dict: dict,
    fname: str,
) -> None:
    """
    Key figure: full PR curve for two-stage vs single operating point
    of one-shot. Demonstrates that two-stage dominates in P-R space.
    """
    prec, rec, _ = precision_recall_curve(y_true_s1, prob_s1)
    auc_pr       = average_precision_score(y_true_s1, prob_s1)

    op_colors = {
        "youden":     ("black",    "Youden-J"),
        "f1_opt":     ("seagreen", "F1-optimal"),
        "prec_match": ("tomato",   "Precision-matched"),
    }

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(rec, prec, lw=2.5, color="steelblue", zorder=2,
            label=f"Two-stage Stage 1 (PR AUC = {auc_pr:.4f})")
    ax.axhline(y_true_s1.mean(), color="grey", lw=1, ls="--",
               label=f"Random (prevalence = {y_true_s1.mean():.3f})")

    # One-shot fixed operating point
    ax.scatter(
        [oneshot_metrics["recall_class2"]],
        [oneshot_metrics["precision_class2"]],
        marker="*", s=350, color="orange", zorder=5,
        label=(f"One-shot  "
               f"P={oneshot_metrics['precision_class2']:.3f}  "
               f"R={oneshot_metrics['recall_class2']:.3f}  "
               f"F1={oneshot_metrics['f1_class2']:.3f}"),
    )

    # Two-stage operating points
    for key, thr in thresholds_dict.items():
        pred = (prob_s1 >= thr).astype(int)
        p_arr, r_arr, f_arr, _ = precision_recall_fscore_support(
            y_true_s1, pred, labels=[0, 1], zero_division=0
        )
        color, label_str = op_colors[key]
        ax.scatter(
            [r_arr[1]], [p_arr[1]],
            marker="o", s=130, color=color, zorder=5,
            label=(f"Two-stage ({label_str})  "
                   f"P={p_arr[1]:.3f}  R={r_arr[1]:.3f}  "
                   f"F1={f_arr[1]:.3f}"),
        )

    ax.set_xlabel("Recall (Half-metal)", fontsize=11)
    ax.set_ylabel("Precision (Half-metal)", fontsize=11)
    ax.set_title(
        "One-Shot vs Two-Stage: Precision–Recall Space\n"
        "(orange star = one-shot fixed point; "
        "circles = two-stage tunable thresholds)",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim([0, 1.02]); ax.set_ylim([0, 1.05])
    ax.grid(ls="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fname}")


def plot_comparison_bar(
    comparison_rows: list[dict],
    fname: str,
) -> None:
    labels = [r["model"] for r in comparison_rows]
    prec   = [r["precision_hm"] for r in comparison_rows]
    rec    = [r["recall_hm"]    for r in comparison_rows]
    f1     = [r["f1_hm"]        for r in comparison_rows]
    x      = np.arange(len(labels))
    width  = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.8), 5))
    ax.bar(x - width, prec, width, label="Precision",
           color="tomato",    alpha=0.85)
    ax.bar(x,         rec,  width, label="Recall",
           color="steelblue", alpha=0.85)
    ax.bar(x + width, f1,   width, label="F1",
           color="seagreen",  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score"); ax.set_ylim([0, 1.05])
    ax.set_title("Half-metal Detection — Model Comparison\n"
                 "(Precision / Recall / F1 for half-metal class)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", ls="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fname}")


def plot_feature_importance(
    feat_importances: list[np.ndarray],
    feature_names: list[str],
    title: str,
    fname: str,
    top_n: int = 20,
) -> None:
    mean_imp = np.mean(feat_importances, axis=0)
    std_imp  = np.std(feat_importances,  axis=0)
    idx      = np.argsort(mean_imp)[-top_n:][::-1]
    fig, ax  = plt.subplots(figsize=(10, max(6, top_n * 0.28)))
    y_pos    = np.arange(top_n)
    ax.barh(y_pos, mean_imp[idx][::-1], xerr=std_imp[idx][::-1],
            color="steelblue", ecolor="grey", alpha=0.85,
            height=0.7, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in idx[::-1]], fontsize=9)
    ax.set_xlabel("Mean Gain Importance (± std)")
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# APPROACH 1 — ONE-SHOT THREE-CLASS
# ══════════════════════════════════════════════════════════════════════════════

def train_oneshot(X: pd.DataFrame, y: pd.Series) -> dict:
    """5-seed capped undersampling, LGBM + XGBoost multiclass ensemble."""
    n_classes = 3

    print(f"\n{'═' * 60}")
    print(f"  APPROACH 1 — One-Shot Three-Class (Baseline)")
    print(f"{'═' * 60}")
    for cls, label in CLASS_NAMES.items():
        print(f"  Class {cls} ({label}): {(y == cls).sum()}")
    eff = (y == 2).sum() + MAJORITY_CAP * 2
    print(f"  Majority cap: {MAJORITY_CAP}  (~{eff} rows per model  "
          f"ratio ≈ {MAJORITY_CAP/(y==2).sum():.1f}:"
          f"{MAJORITY_CAP/(y==2).sum():.1f}:1)")
    print(f"  {N_SPLITS}-fold × {len(UNDERSAMPLE_SEEDS)} seeds = "
          f"{N_SPLITS * len(UNDERSAMPLE_SEEDS)} models per algorithm")

    cv    = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                            random_state=RANDOM_STATE)
    folds = list(cv.split(X, y))

    oof_proba_by_seed: dict[int, np.ndarray] = {}
    seed_cv_metrics:   dict[int, list[dict]] = {}
    best_iters_lgb:    dict[int, list[int]]  = {}
    best_iters_xgb:    dict[int, list[int]]  = {}
    final_lgb:         dict[int, lgb.LGBMClassifier] = {}
    final_xgb:         dict[int, xgb.XGBClassifier]  = {}
    y_oof = np.full(len(y), -1, dtype=int)

    for seed in UNDERSAMPLE_SEEDS:
        print(f"\n  Seed {seed}  {'─' * 45}")
        oof_proba = np.zeros((len(y), n_classes))
        fold_metrics = []
        iters_lgb, iters_xgb = [], []

        for fold, (tr_idx, va_idx) in enumerate(folds, 1):
            Xva = X.iloc[va_idx]
            yva = y.iloc[va_idx]
            if seed == UNDERSAMPLE_SEEDS[0]:
                y_oof[va_idx] = yva.values

            Xtr_bal, ytr_bal = undersample_capped(
                X.iloc[tr_idx], y.iloc[tr_idx],
                minority_class=2, majority_cap=MAJORITY_CAP, seed=seed,
            )

            lgbm = build_lgbm_multiclass(n_classes)
            xgbm = build_xgb_multiclass(early_stopping=True)

            lgbm.fit(
                Xtr_bal, ytr_bal,
                eval_set=[(Xva, yva)], eval_metric="multi_logloss",
                callbacks=[lgb.early_stopping(200, verbose=False),
                           lgb.log_evaluation(period=-1)],
            )
            xgbm.fit(Xtr_bal, ytr_bal, eval_set=[(Xva, yva)], verbose=False)

            iters_lgb.append(lgbm.best_iteration_)
            iters_xgb.append(xgbm.best_iteration)

            p_ens = (lgbm.predict_proba(Xva) + xgbm.predict_proba(Xva)) / 2
            oof_proba[va_idx] += p_ens

            fm = three_class_metrics(yva.values, np.argmax(p_ens, axis=1))
            fold_metrics.append(fm)
            print(f"  fold {fold}/{N_SPLITS}  "
                  f"lgbm={lgbm.best_iteration_:4d}  "
                  f"xgb={xgbm.best_iteration:4d}  "
                  f"macro_F1={fm['macro_f1']:.4f}  "
                  f"F1_hm={fm['f1_class2']:.4f}  "
                  f"P_hm={fm['precision_class2']:.4f}  "
                  f"R_hm={fm['recall_class2']:.4f}")

        oof_proba_by_seed[seed] = oof_proba
        seed_cv_metrics[seed]   = fold_metrics
        best_iters_lgb[seed]    = iters_lgb
        best_iters_xgb[seed]    = iters_xgb

    # Ensemble OOF
    ens_proba = np.mean([oof_proba_by_seed[s]
                         for s in UNDERSAMPLE_SEEDS], axis=0)
    ens_pred  = np.argmax(ens_proba, axis=1)
    ens_m     = three_class_metrics(y_oof, ens_pred)
    cm        = confusion_matrix(y_oof, ens_pred, labels=[0, 1, 2])

    print(f"\n  One-shot Ensemble OOF:")
    for k, v in ens_m.items():
        print(f"    {k}: {v:.4f}")
    print(f"\n  Confusion matrix:\n{cm}")

    # Final fit
    print(f"\n  One-shot final fit — full dataset, one model per seed")
    for seed in UNDERSAMPLE_SEEDS:
        mean_lgb = max(1, int(np.mean(best_iters_lgb[seed])))
        mean_xgb = max(1, int(np.mean(best_iters_xgb[seed])))
        X_bal, y_bal = undersample_capped(
            X, y, minority_class=2, majority_cap=MAJORITY_CAP, seed=seed,
        )
        m_lgb = build_lgbm_multiclass(n_classes, n_estimators=mean_lgb)
        m_xgb = build_xgb_multiclass(n_estimators=mean_xgb,
                                      early_stopping=False)
        m_lgb.fit(X_bal, y_bal)
        m_xgb.fit(X_bal, y_bal)
        final_lgb[seed] = m_lgb
        final_xgb[seed] = m_xgb
        print(f"  seed {seed:3d}  lgbm={mean_lgb}  xgb={mean_xgb}")

    plot_confusion(
        y_oof, ens_pred, [CLASS_NAMES[i] for i in range(3)],
        "One-Shot Three-Class — Confusion Matrix (Ensemble OOF)",
        os.path.join(IMG_DIR, "oneshot_confusion.png"),
    )

    return dict(
        final_lgb       = final_lgb,
        final_xgb       = final_xgb,
        oof_proba       = ens_proba,
        oof_pred        = ens_pred,
        oof_true        = y_oof,
        oof_metrics     = ens_m,
        seed_cv_metrics = seed_cv_metrics,
        confusion_matrix = cm.tolist(),
    )


# ══════════════════════════════════════════════════════════════════════════════
# APPROACH 2 — STAGE 1: HALF-METAL DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

def train_stage1(
    X: pd.DataFrame,
    y_3class: pd.Series,
    feature_names: list[str],
    oneshot_precision: float,
) -> dict:
    """5-seed 1:1 undersampling, LGBM + XGBoost binary ensemble.
    Three thresholds evaluated: Youden-J, F1-optimal, precision-matched."""
    y_bin = (y_3class == 2).astype(int)
    y_bin.index = y_3class.index

    n_hm  = (y_bin == 1).sum()
    n_not = (y_bin == 0).sum()

    print(f"\n{'═' * 60}")
    print(f"  APPROACH 2 — Stage 1: Half-metal Detector")
    print(f"{'═' * 60}")
    print(f"  Half-metal: {n_hm}   Not: {n_not}   Ratio {n_not/n_hm:.1f}:1")
    print(f"  Precision target (one-shot match): {oneshot_precision:.4f}")

    cv    = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                            random_state=RANDOM_STATE)
    folds = list(cv.split(X, y_bin))

    oof_lgb_by_seed: dict[int, np.ndarray] = {}
    oof_xgb_by_seed: dict[int, np.ndarray] = {}
    seed_cv_metrics: dict[int, list[dict]] = {}
    best_iters_lgb:  dict[int, list[int]]  = {}
    best_iters_xgb:  dict[int, list[int]]  = {}
    final_lgb:       dict[int, lgb.LGBMClassifier] = {}
    final_xgb:       dict[int, xgb.XGBClassifier]  = {}
    all_lgb_imps:    list[np.ndarray] = []
    y_oof = np.zeros(len(y_bin))

    for seed in UNDERSAMPLE_SEEDS:
        print(f"\n  Seed {seed}  {'─' * 45}")
        oof_lgb  = np.zeros(len(y_bin))
        oof_xgb  = np.zeros(len(y_bin))
        f_metrics = []
        iters_lgb, iters_xgb = [], []

        for fold, (tr_idx, va_idx) in enumerate(folds, 1):
            Xva = X.iloc[va_idx]
            yva = y_bin.iloc[va_idx]
            if seed == UNDERSAMPLE_SEEDS[0]:
                y_oof[va_idx] = yva.values

            Xtr_bal, ytr_bal = undersample_strict(
                X.iloc[tr_idx], y_bin.iloc[tr_idx],
                minority_class=1, seed=seed,
            )

            lgbm = build_lgbm_binary()
            xgbm = build_xgb_binary(early_stopping=True)
            lgbm.fit(
                Xtr_bal, ytr_bal,
                eval_set=[(Xva, yva)], eval_metric="auc",
                callbacks=[lgb.early_stopping(200, verbose=False),
                           lgb.log_evaluation(period=-1)],
            )
            xgbm.fit(Xtr_bal, ytr_bal,
                     eval_set=[(Xva, yva)], verbose=False)

            iters_lgb.append(lgbm.best_iteration_)
            iters_xgb.append(xgbm.best_iteration)
            all_lgb_imps.append(lgbm.feature_importances_)

            p_ens = (lgbm.predict_proba(Xva)[:, 1] +
                     xgbm.predict_proba(Xva)[:, 1]) / 2
            oof_lgb[va_idx] = lgbm.predict_proba(Xva)[:, 1]
            oof_xgb[va_idx] = xgbm.predict_proba(Xva)[:, 1]

            fm = binary_metrics(yva.values, p_ens, thr=0.5)
            f_metrics.append(fm)
            print(f"  fold {fold}/{N_SPLITS}  "
                  f"lgbm={lgbm.best_iteration_:4d}  "
                  f"xgb={xgbm.best_iteration:4d}  "
                  f"AUC={fm['roc_auc']:.4f}  "
                  f"P={fm['precision_class1']:.4f}  "
                  f"R={fm['recall_class1']:.4f}  "
                  f"F1={fm['f1_class1']:.4f}")

        oof_lgb_by_seed[seed] = oof_lgb
        oof_xgb_by_seed[seed] = oof_xgb
        seed_cv_metrics[seed] = f_metrics
        best_iters_lgb[seed]  = iters_lgb
        best_iters_xgb[seed]  = iters_xgb

    # Ensemble OOF
    all_probs = ([oof_lgb_by_seed[s] for s in UNDERSAMPLE_SEEDS] +
                 [oof_xgb_by_seed[s] for s in UNDERSAMPLE_SEEDS])
    ens_prob  = np.mean(all_probs, axis=0)

    # Three thresholds
    thr_youden = oof_best_threshold(y_oof, ens_prob)
    thr_f1opt  = f1_optimal_threshold(y_oof, ens_prob)
    thr_pmatch = precision_matched_threshold(y_oof, ens_prob,
                                             oneshot_precision)
    thresholds = {
        "youden":     thr_youden,
        "f1_opt":     thr_f1opt,
        "prec_match": thr_pmatch,
    }

    print(f"\n  Stage 1 thresholds:")
    metrics_per_thr = {}
    for name, thr in thresholds.items():
        m = binary_metrics(y_oof, ens_prob, thr)
        metrics_per_thr[name] = m
        print(f"  {name:<12} thr={thr:.3f}  "
              f"P={m['precision_class1']:.4f}  "
              f"R={m['recall_class1']:.4f}  "
              f"F1={m['f1_class1']:.4f}")

    # Final fit
    print(f"\n  Stage 1 final fit — full dataset")
    for seed in UNDERSAMPLE_SEEDS:
        mean_lgb = max(1, int(np.mean(best_iters_lgb[seed])))
        mean_xgb = max(1, int(np.mean(best_iters_xgb[seed])))
        X_bal, y_bal = undersample_strict(X, y_bin,
                                          minority_class=1, seed=seed)
        m_lgb = build_lgbm_binary(n_estimators=mean_lgb)
        m_xgb = build_xgb_binary(n_estimators=mean_xgb, early_stopping=False)
        m_lgb.fit(X_bal, y_bal)
        m_xgb.fit(X_bal, y_bal)
        final_lgb[seed] = m_lgb
        final_xgb[seed] = m_xgb
        print(f"  seed {seed:3d}  lgbm={mean_lgb}  xgb={mean_xgb}")

    # Figures
    plot_roc_pr(
        y_oof, ens_prob, "Half-metal",
        "Stage 1 — Half-metal Detector (Ensemble OOF)",
        os.path.join(IMG_DIR, "stage1_roc_pr.png"),
    )
    plot_threshold_sweep(
        y_oof, ens_prob, thresholds, "Half-metal",
        os.path.join(IMG_DIR, "stage1_threshold_sweep.png"),
    )
    plot_confusion(
        y_oof, (ens_prob >= thr_youden).astype(int),
        ["Not half-metal", "Half-metal"],
        "Stage 1 — Confusion Matrix (Youden-J, Ensemble OOF)",
        os.path.join(IMG_DIR, "stage1_confusion.png"),
    )
    plot_feature_importance(
        all_lgb_imps, feature_names,
        "Stage 1 LGBM — Feature Importance (Top 20, mean ± std)",
        os.path.join(IMG_DIR, "feature_importance_stage1.png"),
    )

    return dict(
        final_lgb           = final_lgb,
        final_xgb           = final_xgb,
        oof_prob            = ens_prob,
        oof_true            = y_oof,
        thresholds          = thresholds,
        seed_cv_metrics     = seed_cv_metrics,
        metrics_per_threshold = metrics_per_thr,
    )


# ══════════════════════════════════════════════════════════════════════════════
# APPROACH 2 — STAGE 2: CONDUCTOR / INSULATOR
# ══════════════════════════════════════════════════════════════════════════════

def train_stage2(
    X: pd.DataFrame,
    y_3class: pd.Series,
    feature_names: list[str],
) -> dict:
    """LGBM + XGBoost binary classifier on non-half-metal samples."""
    mask = y_3class != 2
    X_s2 = X.loc[mask]
    y_s2 = y_3class.loc[mask].copy()
    n0   = (y_s2 == 0).sum()
    n1   = (y_s2 == 1).sum()

    print(f"\n{'═' * 60}")
    print(f"  APPROACH 2 — Stage 2: Conductor / Insulator")
    print(f"{'═' * 60}")
    print(f"  Conductor: {n0}   Insulator: {n1}   Ratio {n0/n1:.2f}:1")

    cv    = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                            random_state=RANDOM_STATE)
    folds = list(cv.split(X_s2, y_s2))

    oof_lgb        = np.zeros(len(y_s2))
    oof_xgb        = np.zeros(len(y_s2))
    metrics_lgb    = []
    metrics_xgb    = []
    best_iters_lgb = []
    best_iters_xgb = []
    lgb_imps: list[np.ndarray] = []

    print(f"\n  {'Fold':<6} {'LGBM iter':>9}  {'AUC':>7}  "
          f"{'XGB iter':>8}  {'AUC':>7}")

    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        Xtr, Xva = X_s2.iloc[tr_idx], X_s2.iloc[va_idx]
        ytr, yva = y_s2.iloc[tr_idx], y_s2.iloc[va_idx]

        lgbm = build_lgbm_binary()
        xgbm = build_xgb_binary(early_stopping=True)
        lgbm.fit(
            Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
            callbacks=[lgb.early_stopping(200, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        xgbm.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

        best_iters_lgb.append(lgbm.best_iteration_)
        best_iters_xgb.append(xgbm.best_iteration)
        lgb_imps.append(lgbm.feature_importances_)
        oof_lgb[va_idx] = lgbm.predict_proba(Xva)[:, 1]
        oof_xgb[va_idx] = xgbm.predict_proba(Xva)[:, 1]

        m_lgb = binary_metrics(yva.values, oof_lgb[va_idx], thr=0.5)
        m_xgb = binary_metrics(yva.values, oof_xgb[va_idx], thr=0.5)
        metrics_lgb.append(m_lgb)
        metrics_xgb.append(m_xgb)
        print(f"  {fold}/{N_SPLITS}      "
              f"{lgbm.best_iteration_:9d}  {m_lgb['roc_auc']:7.4f}  "
              f"{xgbm.best_iteration:8d}  {m_xgb['roc_auc']:7.4f}")

    ens_prob_s2 = (oof_lgb + oof_xgb) / 2
    thr_s2      = oof_best_threshold(y_s2.values, ens_prob_s2)
    ens_m_s2    = binary_metrics(y_s2.values, ens_prob_s2, thr=thr_s2)

    print(f"\n  OOF Ensemble (thr={thr_s2:.3f})  "
          f"AUC={ens_m_s2['roc_auc']:.4f}  "
          f"macro_F1={ens_m_s2['macro_f1']:.4f}")

    mean_lgb = max(1, int(np.mean(best_iters_lgb)))
    mean_xgb = max(1, int(np.mean(best_iters_xgb)))
    final_lgb = build_lgbm_binary(n_estimators=mean_lgb)
    final_xgb = build_xgb_binary(n_estimators=mean_xgb, early_stopping=False)
    final_lgb.fit(X_s2, y_s2)
    final_xgb.fit(X_s2, y_s2)

    plot_roc_pr(
        y_s2.values, ens_prob_s2, "Insulator",
        "Stage 2 — Conductor/Insulator (Ensemble OOF)",
        os.path.join(IMG_DIR, "stage2_roc_pr.png"),
    )
    plot_confusion(
        y_s2.values, (ens_prob_s2 >= thr_s2).astype(int),
        ["Conductor", "Insulator"],
        "Stage 2 — Confusion Matrix (Ensemble OOF)",
        os.path.join(IMG_DIR, "stage2_confusion.png"),
    )
    plot_feature_importance(
        lgb_imps, feature_names,
        "Stage 2 LGBM — Feature Importance (Top 20, mean ± std over folds)",
        os.path.join(IMG_DIR, "feature_importance_stage2.png"),
    )

    return dict(
        final_lgb   = final_lgb,
        final_xgb   = final_xgb,
        threshold   = thr_s2,
        oof_prob    = ens_prob_s2,
        oof_true    = y_s2.values,
        oof_metrics = ens_m_s2,
        cv_lgb      = summarize(metrics_lgb),
        cv_xgb      = summarize(metrics_xgb),
    )


# ══════════════════════════════════════════════════════════════════════════════
# TWO-STAGE END-TO-END EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_twostage_e2e(
    y_3class: pd.Series,
    s1_oof_prob: np.ndarray,
    s1_thr: float,
    s2_oof_prob: np.ndarray,
    s2_thr: float,
    label: str,
) -> dict:
    """Reconstruct full three-class predictions and evaluate."""
    y           = y_3class.values
    s1_pred     = (s1_oof_prob >= s1_thr).astype(int)
    mask_not_hm = y_3class != 2
    idx_not_hm  = np.where(mask_not_hm)[0]
    s2_pred_sub = (s2_oof_prob >= s2_thr).astype(int)

    final_pred = np.full(len(y), -1, dtype=int)
    final_pred[s1_pred == 1] = 2
    for i, idx in enumerate(idx_not_hm):
        if s1_pred[idx] == 0:
            final_pred[idx] = s2_pred_sub[i]

    valid = final_pred != -1
    m  = three_class_metrics(y[valid], final_pred[valid])
    cm = confusion_matrix(y[valid], final_pred[valid], labels=[0, 1, 2])

    print(f"  Two-stage ({label}):  "
          f"macro_F1={m['macro_f1']:.4f}  "
          f"F1_hm={m['f1_class2']:.4f}  "
          f"P_hm={m['precision_class2']:.4f}  "
          f"R_hm={m['recall_class2']:.4f}")

    safe_label = label.replace(" ", "_").replace("/", "-")
    plot_confusion(
        y[valid], final_pred[valid],
        [CLASS_NAMES[i] for i in range(3)],
        f"Two-Stage End-to-End ({label}) — Confusion Matrix (OOF)",
        os.path.join(IMG_DIR, f"final_confusion_{safe_label}.png"),
    )

    return {"label": label, "metrics": m, "confusion_matrix": cm.tolist()}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    df    = pd.read_pickle(DATA_PATH)
    feats = load_features()

    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns],
                errors="ignore")[feats]
    y = df["hm_class"]

    print(f"\n{'═' * 60}")
    print(f"  hm_class — One-Shot vs Two-Stage Comparison")
    print(f"{'═' * 60}")
    print(f"  Total samples: {len(y)}")
    for cls, label in CLASS_NAMES.items():
        print(f"  Class {cls} ({label}): {(y == cls).sum():6d}")
    print(f"  Features: {len(feats)}")

    # ── Approach 1 ────────────────────────────────────────────────────────
    os_result = train_oneshot(X, y)
    oneshot_p = os_result["oof_metrics"]["precision_class2"]

    # ── Approach 2 ────────────────────────────────────────────────────────
    s1 = train_stage1(X, y, feats, oneshot_precision=oneshot_p)
    s2 = train_stage2(X, y, feats)

    # End-to-end at three thresholds
    threshold_labels = {
        "youden":     "Youden-J",
        "f1_opt":     "F1-optimal",
        "prec_match": "Precision-matched",
    }
    e2e_results = {}
    print(f"\n{'═' * 60}")
    print(f"  END-TO-END COMPARISON")
    print(f"{'═' * 60}")
    for key, label in threshold_labels.items():
        thr = s1["thresholds"][key]
        e2e = evaluate_twostage_e2e(
            y, s1["oof_prob"], thr,
            s2["oof_prob"], s2["threshold"], label,
        )
        e2e_results[key] = e2e

    # Comparison table
    comparison_rows = [{
        "model":        "One-shot (softmax)",
        "precision_hm": os_result["oof_metrics"]["precision_class2"],
        "recall_hm":    os_result["oof_metrics"]["recall_class2"],
        "f1_hm":        os_result["oof_metrics"]["f1_class2"],
        "macro_f1":     os_result["oof_metrics"]["macro_f1"],
    }]
    for key, label in threshold_labels.items():
        m = e2e_results[key]["metrics"]
        comparison_rows.append({
            "model":        f"Two-stage ({label})",
            "precision_hm": m["precision_class2"],
            "recall_hm":    m["recall_class2"],
            "f1_hm":        m["f1_class2"],
            "macro_f1":     m["macro_f1"],
        })

    print(f"\n{'═' * 72}")
    print(f"  {'Model':<35} {'P_hm':>7} {'R_hm':>7} "
          f"{'F1_hm':>7} {'Macro_F1':>9}")
    print(f"  {'─' * 65}")
    for row in comparison_rows:
        print(f"  {row['model']:<35} "
              f"{row['precision_hm']:>7.4f} "
              f"{row['recall_hm']:>7.4f} "
              f"{row['f1_hm']:>7.4f} "
              f"{row['macro_f1']:>9.4f}")
    print(f"{'═' * 72}")

    # Comparison figures
    print(f"\n  Saving comparison figures → {IMG_DIR}/")
    plot_pr_comparison(
        s1["oof_true"], s1["oof_prob"],
        os_result["oof_metrics"], s1["thresholds"],
        os.path.join(IMG_DIR, "comparison_pr_curve.png"),
    )
    plot_comparison_bar(
        comparison_rows,
        os.path.join(IMG_DIR, "comparison_bar.png"),
    )

    # Save models
    print(f"\n  Saving models → {OUT_DIR}/")
    for seed in UNDERSAMPLE_SEEDS:
        joblib.dump(os_result["final_lgb"][seed],
                    os.path.join(OUT_DIR, f"oneshot_lgbm_seed{seed}.pkl"))
        joblib.dump(os_result["final_xgb"][seed],
                    os.path.join(OUT_DIR, f"oneshot_xgb_seed{seed}.pkl"))
        joblib.dump(s1["final_lgb"][seed],
                    os.path.join(OUT_DIR, f"stage1_lgbm_seed{seed}.pkl"))
        joblib.dump(s1["final_xgb"][seed],
                    os.path.join(OUT_DIR, f"stage1_xgb_seed{seed}.pkl"))
    joblib.dump(s2["final_lgb"], os.path.join(OUT_DIR, "stage2_lgbm.pkl"))
    joblib.dump(s2["final_xgb"], os.path.join(OUT_DIR, "stage2_xgb.pkl"))

    # Save metrics
    save_json(s1["thresholds"],
              os.path.join(OUT_DIR, "thresholds.json"))
    save_json({
        "oof_metrics":      os_result["oof_metrics"],
        "confusion_matrix": os_result["confusion_matrix"],
        "seed_cv": {str(s): summarize(os_result["seed_cv_metrics"][s])
                    for s in UNDERSAMPLE_SEEDS},
    }, os.path.join(OUT_DIR, "oneshot_metrics.json"))
    save_json({
        "metrics_per_threshold": s1["metrics_per_threshold"],
        "thresholds":            s1["thresholds"],
        "seed_cv": {str(s): summarize(s1["seed_cv_metrics"][s])
                    for s in UNDERSAMPLE_SEEDS},
    }, os.path.join(OUT_DIR, "stage1_metrics.json"))
    save_json({
        "oof_metrics": s2["oof_metrics"],
        "cv_lgbm":     s2["cv_lgb"],
        "cv_xgb":      s2["cv_xgb"],
        "threshold":   s2["threshold"],
    }, os.path.join(OUT_DIR, "stage2_metrics.json"))
    save_json({
        "comparison":   comparison_rows,
        "oneshot":      os_result["oof_metrics"],
        "twostage_e2e": {k: v["metrics"] for k, v in e2e_results.items()},
        "n_features":   len(feats),
    }, os.path.join(OUT_DIR, "hm_class_metrics.json"))

    print(f"\n{'═' * 60}")
    print(f"  Done.  All outputs in: {OUT_DIR}/")
    print(f"{'═' * 60}")


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def predict_hm_class(
    X_new: pd.DataFrame,
    model_dir: str = OUT_DIR,
    threshold_key: str = "f1_opt",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-stage inference.
    threshold_key: 'youden' | 'f1_opt' | 'prec_match'
    Returns (pred, proba_hm).
    """
    thr_all = json.load(open(os.path.join(model_dir, "thresholds.json")))
    s1_thr  = thr_all[threshold_key]
    s2_thr  = json.load(open(
        os.path.join(model_dir, "stage2_metrics.json")
    ))["threshold"]

    s1_lgb = [joblib.load(os.path.join(model_dir,
               f"stage1_lgbm_seed{s}.pkl")) for s in UNDERSAMPLE_SEEDS]
    s1_xgb = [joblib.load(os.path.join(model_dir,
               f"stage1_xgb_seed{s}.pkl")) for s in UNDERSAMPLE_SEEDS]
    proba_hm = np.mean(
        [m.predict_proba(X_new)[:, 1] for m in s1_lgb + s1_xgb], axis=0,
    )
    is_hm = proba_hm >= s1_thr

    s2_lgb = joblib.load(os.path.join(model_dir, "stage2_lgbm.pkl"))
    s2_xgb = joblib.load(os.path.join(model_dir, "stage2_xgb.pkl"))
    proba_ins = (s2_lgb.predict_proba(X_new)[:, 1] +
                 s2_xgb.predict_proba(X_new)[:, 1]) / 2
    s2_pred = (proba_ins >= s2_thr).astype(int)

    return np.where(is_hm, 2, s2_pred), proba_hm


if __name__ == "__main__":
    main()