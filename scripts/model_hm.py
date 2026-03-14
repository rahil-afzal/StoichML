#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_hm_class.py
─────────────────
StoichML training pipeline — hm_class, two-stage approach.

Stage 1  half-metal detector
         Binary: half-metal (class 2) vs not-half-metal (class 0+1)
         5-seed strict 1:1 undersampling — minority = class 2 (661 samples)
         LGBM + XGBoost ensemble, Youden-J threshold tuned on OOF

Stage 2  conductor / insulator classifier
         Binary: conductor (class 0) vs insulator (class 1)
         Trained only on non-half-metal samples
         5-fold CV, LGBM + XGBoost ensemble

Inference
         p2 = stage1_ensemble.predict_proba[:, 1]
         if p2 >= stage1_threshold  →  class 2  (half-metal)
         else                       →  stage2_ensemble.predict  (0 or 1)

Why two-stage?
         The half-metal boundary is tuned independently via a threshold
         rather than softmax argmax over three classes. This directly
         fixes the low-precision problem (P=0.26) from the one-shot
         approach where the model over-predicted class 2.

Outputs (models/hm_class/)
         stage1_lgbm_seed{s}.pkl  ×5
         stage1_xgb_seed{s}.pkl   ×5
         stage1_threshold.json
         stage2_lgbm.pkl
         stage2_xgb.pkl
         stage1_metrics.json
         stage2_metrics.json
         hm_class_metrics.json    end-to-end three-class evaluation
         images/
           stage1_roc_pr.png
           stage1_threshold_sweep.png
           stage1_confusion.png
           stage2_roc_pr.png
           stage2_confusion.png
           final_confusion.png
           feature_importance_stage1.png
           feature_importance_stage2.png

Usage:
    python train_hm_class.py
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
    fpr, tpr, thr = roc_curve(y_true, prob)
    return float(thr[np.argmax(tpr - fpr)])


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

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


def three_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    return {
        "macro_f1":         float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
        "weighted_f1":      float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_acc":     float(balanced_accuracy_score(y_true, y_pred)),
        "f1_class0":        float(f[0]), "precision_class0": float(p[0]), "recall_class0": float(r[0]),
        "f1_class1":        float(f[1]), "precision_class1": float(p[1]), "recall_class1": float(r[1]),
        "f1_class2":        float(f[2]), "precision_class2": float(p[2]), "recall_class2": float(r[2]),
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
    """
    Strict 1:1 undersampling — must be identical to feature_selection.py.
    """
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


# ══════════════════════════════════════════════════════════════════════════════
# MODEL BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_lgbm_binary(n_estimators: int = 5000) -> lgb.LGBMClassifier:
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
        verbose=-1,
    )


def build_xgb_binary(
    n_estimators: int = 5000,
    early_stopping: bool = True,
) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=1.5,
        n_estimators=n_estimators,
        tree_method="hist",
        use_label_encoder=False,
        eval_metric="auc",
        random_state=RANDOM_STATE,
        early_stopping_rounds=200 if early_stopping else None,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc_pr(y_true: np.ndarray,
                prob: np.ndarray,
                pos_label_name: str,
                title: str,
                fname: str) -> None:
    """2-panel ROC + Precision-Recall curves."""
    fpr, tpr, _     = roc_curve(y_true, prob)
    prec, rec, _    = precision_recall_curve(y_true, prob)
    auc_roc         = roc_auc_score(y_true, prob)
    auc_pr          = average_precision_score(y_true, prob)
    baseline_pr     = y_true.mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(fpr, tpr, lw=2, color="steelblue", label=f"ROC AUC = {auc_roc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(rec, prec, lw=2, color="steelblue", label=f"PR AUC = {auc_pr:.4f}")
    ax.axhline(baseline_pr, color="k", lw=1, ls="--",
               label=f"Baseline (prevalence = {baseline_pr:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel(f"Precision ({pos_label_name})")
    ax.set_title("Precision-Recall Curve")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fname}")


def plot_threshold_sweep(y_true: np.ndarray,
                         prob: np.ndarray,
                         chosen_thr: float,
                         pos_label_name: str,
                         fname: str) -> None:
    """F1 / Precision / Recall of the positive class across thresholds."""
    thresholds = np.linspace(0.01, 0.99, 200)
    f1s, precs, recs = [], [], []
    for t in thresholds:
        pred = (prob >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            y_true, pred, labels=[0, 1], zero_division=0
        )
        f1s.append(f[1]); precs.append(p[1]); recs.append(r[1])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, f1s,   lw=2, label=f"F1 ({pos_label_name})",        color="steelblue")
    ax.plot(thresholds, precs, lw=2, label=f"Precision ({pos_label_name})",  color="tomato")
    ax.plot(thresholds, recs,  lw=2, label=f"Recall ({pos_label_name})",     color="seagreen")
    ax.axvline(chosen_thr, color="k", lw=1.5, ls="--",
               label=f"Chosen threshold = {chosen_thr:.3f}")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Stage 1 — Threshold Sweep  ({pos_label_name})")
    ax.legend(fontsize=9)
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.grid(which="major", ls="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fname}")


def plot_confusion(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   labels: list[str],
                   title: str,
                   fname: str) -> None:
    """Normalised + raw count confusion matrix."""
    cm      = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, data, fmt, t in [
        (axes[0], cm_norm, ".2f", "Row-normalised (recall per class)"),
        (axes[1], cm,      "d",   "Raw counts"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=labels, yticklabels=labels,
                    linewidths=0.5, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(t)

    plt.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fname}")


def plot_feature_importance(feat_importances: list[np.ndarray],
                            feature_names: list[str],
                            title: str,
                            fname: str,
                            top_n: int = 30) -> None:
    """Mean ± std LGBM gain importance across seeds/folds."""
    mean_imp = np.mean(feat_importances, axis=0)
    std_imp  = np.std(feat_importances,  axis=0)
    idx      = np.argsort(mean_imp)[-top_n:][::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.28)))
    y_pos   = np.arange(top_n)
    ax.barh(y_pos,
            mean_imp[idx][::-1],
            xerr=std_imp[idx][::-1],
            color="steelblue", ecolor="grey",
            alpha=0.85, height=0.7, capsize=3)
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
# STAGE 1 — HALF-METAL DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

def train_stage1(
    X: pd.DataFrame,
    y_3class: pd.Series,
    feature_names: list[str],
) -> dict:
    """
    5-seed ensemble binary classifier: half-metal (1) vs not-half-metal (0).
    Both LGBM and XGBoost trained per seed, OOF ensemble is mean of all 10 models.
    """
    y_bin = (y_3class == 2).astype(int)
    y_bin.index = y_3class.index

    n_hm  = (y_bin == 1).sum()
    n_not = (y_bin == 0).sum()

    print(f"\n{'═' * 60}")
    print(f"  STAGE 1 — Half-metal Detector")
    print(f"{'═' * 60}")
    print(f"  Half-metal: {n_hm}   Not: {n_not}   Ratio {n_not/n_hm:.1f}:1")
    print(f"  After 1:1 undersampling: {n_hm} vs {n_hm}")
    print(f"  {N_SPLITS}-fold × {len(UNDERSAMPLE_SEEDS)} seeds "
          f"= {N_SPLITS * len(UNDERSAMPLE_SEEDS)} models per algorithm")

    cv    = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(cv.split(X, y_bin))

    oof_lgb_by_seed: dict[int, np.ndarray] = {}
    oof_xgb_by_seed: dict[int, np.ndarray] = {}
    seed_cv_metrics: dict[int, list[dict]] = {}
    best_iters_lgb:  dict[int, list[int]]  = {}
    best_iters_xgb:  dict[int, list[int]]  = {}
    final_lgb:       dict[int, lgb.LGBMClassifier]  = {}
    final_xgb:       dict[int, xgb.XGBClassifier]   = {}
    all_lgb_importances: list[np.ndarray]  = []
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
                X.iloc[tr_idx], y_bin.iloc[tr_idx], minority_class=1, seed=seed
            )

            lgbm = build_lgbm_binary()
            xgbm = build_xgb_binary(early_stopping=True)

            lgbm.fit(
                Xtr_bal, ytr_bal,
                eval_set=[(Xva, yva)],
                eval_metric="auc",
                callbacks=[
                    lgb.early_stopping(200, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )
            xgbm.fit(Xtr_bal, ytr_bal, eval_set=[(Xva, yva)], verbose=False)

            iters_lgb.append(lgbm.best_iteration_)
            iters_xgb.append(xgbm.best_iteration)
            all_lgb_importances.append(lgbm.feature_importances_)

            p_lgb = lgbm.predict_proba(Xva)[:, 1]
            p_xgb = xgbm.predict_proba(Xva)[:, 1]
            p_ens = (p_lgb + p_xgb) / 2

            oof_lgb[va_idx] = p_lgb
            oof_xgb[va_idx] = p_xgb

            fm = binary_metrics(yva.values, p_ens, thr=0.5)
            f_metrics.append(fm)

            print(f"  fold {fold}/{N_SPLITS}  "
                  f"lgbm_iter={lgbm.best_iteration_:4d}  "
                  f"xgb_iter={xgbm.best_iteration:4d}  "
                  f"AUC={fm['roc_auc']:.4f}  "
                  f"F1_hm={fm['f1_class1']:.4f}  "
                  f"P={fm['precision_class1']:.4f}  "
                  f"R={fm['recall_class1']:.4f}")

        oof_lgb_by_seed[seed] = oof_lgb
        oof_xgb_by_seed[seed] = oof_xgb
        seed_cv_metrics[seed] = f_metrics
        best_iters_lgb[seed]  = iters_lgb
        best_iters_xgb[seed]  = iters_xgb

    # Ensemble OOF — mean across all seeds × both models
    all_oof_probs = (
        [oof_lgb_by_seed[s] for s in UNDERSAMPLE_SEEDS] +
        [oof_xgb_by_seed[s] for s in UNDERSAMPLE_SEEDS]
    )
    ens_prob = np.mean(all_oof_probs, axis=0)
    thr      = oof_best_threshold(y_oof, ens_prob)
    ens_m    = binary_metrics(y_oof, ens_prob, thr=thr)

    print(f"\n  Stage 1 Ensemble OOF  (Youden-J thr={thr:.3f})")
    print(f"  AUC={ens_m['roc_auc']:.4f}  PR-AUC={ens_m['pr_auc']:.4f}  "
          f"macro_F1={ens_m['macro_f1']:.4f}")
    print(f"  Half-metal:  "
          f"P={ens_m['precision_class1']:.4f}  "
          f"R={ens_m['recall_class1']:.4f}  "
          f"F1={ens_m['f1_class1']:.4f}")

    # Final fit
    print(f"\n  Stage 1 final fit — full dataset")
    for seed in UNDERSAMPLE_SEEDS:
        mean_lgb = max(1, int(np.mean(best_iters_lgb[seed])))
        mean_xgb = max(1, int(np.mean(best_iters_xgb[seed])))
        print(f"  seed {seed:3d}  lgbm={mean_lgb}  xgb={mean_xgb}")

        X_bal, y_bal = undersample_strict(X, y_bin, minority_class=1, seed=seed)
        m_lgb = build_lgbm_binary(n_estimators=mean_lgb)
        m_xgb = build_xgb_binary(n_estimators=mean_xgb, early_stopping=False)
        m_lgb.fit(X_bal, y_bal)
        m_xgb.fit(X_bal, y_bal)
        final_lgb[seed] = m_lgb
        final_xgb[seed] = m_xgb

    # Figures
    print(f"\n  Stage 1 figures → {IMG_DIR}/")
    plot_roc_pr(
        y_oof, ens_prob, "Half-metal",
        "Stage 1 — Half-metal Detector  (Ensemble OOF)",
        os.path.join(IMG_DIR, "stage1_roc_pr.png"),
    )
    plot_threshold_sweep(
        y_oof, ens_prob, thr, "Half-metal",
        os.path.join(IMG_DIR, "stage1_threshold_sweep.png"),
    )
    ens_pred = (ens_prob >= thr).astype(int)
    plot_confusion(
        y_oof, ens_pred,
        ["Not half-metal", "Half-metal"],
        "Stage 1 — Confusion Matrix  (Ensemble OOF)",
        os.path.join(IMG_DIR, "stage1_confusion.png"),
    )
    plot_feature_importance(
        all_lgb_importances, feature_names,
        "Stage 1 LGBM — Feature Importance (Top 30, mean ± std over seeds × folds)",
        os.path.join(IMG_DIR, "feature_importance_stage1.png"),
    )

    return dict(
        final_lgb        = final_lgb,
        final_xgb        = final_xgb,
        threshold        = thr,
        oof_prob         = ens_prob,
        oof_true         = y_oof,
        oof_metrics      = ens_m,
        seed_cv_metrics  = seed_cv_metrics,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — CONDUCTOR / INSULATOR CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

def train_stage2(
    X: pd.DataFrame,
    y_3class: pd.Series,
    feature_names: list[str],
) -> dict:
    """
    LGBM + XGBoost binary classifier on non-half-metal samples only.
    Classes 0 (conductor) and 1 (insulator).
    5-fold CV — no undersampling needed, ratio is much more balanced here.
    """
    mask   = y_3class != 2
    X_s2   = X.loc[mask]
    y_s2   = y_3class.loc[mask].copy()

    n0 = (y_s2 == 0).sum()
    n1 = (y_s2 == 1).sum()

    print(f"\n{'═' * 60}")
    print(f"  STAGE 2 — Conductor / Insulator Classifier")
    print(f"{'═' * 60}")
    print(f"  Conductor: {n0}   Insulator: {n1}   Ratio {n0/n1:.2f}:1")
    print(f"  Features: {len(feature_names)}")

    cv    = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(cv.split(X_s2, y_s2))

    oof_lgb        = np.zeros(len(y_s2))
    oof_xgb        = np.zeros(len(y_s2))
    metrics_lgb    = []
    metrics_xgb    = []
    best_iters_lgb = []
    best_iters_xgb = []
    lgb_importances: list[np.ndarray] = []

    print(f"\n  {'─' * 55}")
    print(f"  {'Fold':<6} {'LGBM iter':>9}  {'AUC':>7} {'F1_cls1':>8}  "
          f"{'XGB iter':>8}  {'AUC':>7} {'F1_cls1':>8}")
    print(f"  {'─' * 55}")

    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        Xtr, Xva = X_s2.iloc[tr_idx], X_s2.iloc[va_idx]
        ytr, yva = y_s2.iloc[tr_idx], y_s2.iloc[va_idx]

        lgbm = build_lgbm_binary()
        xgbm = build_xgb_binary(early_stopping=True)

        lgbm.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(200, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        xgbm.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

        best_iters_lgb.append(lgbm.best_iteration_)
        best_iters_xgb.append(xgbm.best_iteration)
        lgb_importances.append(lgbm.feature_importances_)

        oof_lgb[va_idx] = lgbm.predict_proba(Xva)[:, 1]
        oof_xgb[va_idx] = xgbm.predict_proba(Xva)[:, 1]

        m_lgb = binary_metrics(yva.values, oof_lgb[va_idx], thr=0.5)
        m_xgb = binary_metrics(yva.values, oof_xgb[va_idx], thr=0.5)
        metrics_lgb.append(m_lgb)
        metrics_xgb.append(m_xgb)

        print(f"  {fold}/{N_SPLITS}      "
              f"{lgbm.best_iteration_:9d}  "
              f"{m_lgb['roc_auc']:7.4f} {m_lgb['f1_class1']:8.4f}  "
              f"{xgbm.best_iteration:8d}  "
              f"{m_xgb['roc_auc']:7.4f} {m_xgb['f1_class1']:8.4f}")

    ens_prob_s2 = (oof_lgb + oof_xgb) / 2
    thr_s2      = oof_best_threshold(y_s2.values, ens_prob_s2)
    ens_m_s2    = binary_metrics(y_s2.values, ens_prob_s2, thr=thr_s2)

    print(f"\n  OOF Ensemble  (thr={thr_s2:.3f})")
    print(f"  AUC={ens_m_s2['roc_auc']:.4f}  PR-AUC={ens_m_s2['pr_auc']:.4f}  "
          f"macro_F1={ens_m_s2['macro_f1']:.4f}")
    print(f"  Insulator:  P={ens_m_s2['precision_class1']:.4f}  "
          f"R={ens_m_s2['recall_class1']:.4f}  "
          f"F1={ens_m_s2['f1_class1']:.4f}")

    # Final fit
    mean_lgb = max(1, int(np.mean(best_iters_lgb)))
    mean_xgb = max(1, int(np.mean(best_iters_xgb)))
    print(f"\n  Stage 2 final fit — lgbm={mean_lgb}  xgb={mean_xgb}")
    final_lgb = build_lgbm_binary(n_estimators=mean_lgb)
    final_xgb = build_xgb_binary(n_estimators=mean_xgb, early_stopping=False)
    final_lgb.fit(X_s2, y_s2)
    final_xgb.fit(X_s2, y_s2)

    # Figures
    print(f"\n  Stage 2 figures → {IMG_DIR}/")
    plot_roc_pr(
        y_s2.values, ens_prob_s2, "Insulator",
        "Stage 2 — Conductor/Insulator  (Ensemble OOF)",
        os.path.join(IMG_DIR, "stage2_roc_pr.png"),
    )
    ens_pred_s2 = (ens_prob_s2 >= thr_s2).astype(int)
    plot_confusion(
        y_s2.values, ens_pred_s2,
        ["Conductor", "Insulator"],
        "Stage 2 — Confusion Matrix  (Ensemble OOF)",
        os.path.join(IMG_DIR, "stage2_confusion.png"),
    )
    plot_feature_importance(
        lgb_importances, feature_names,
        "Stage 2 LGBM — Feature Importance (Top 30, mean ± std over folds)",
        os.path.join(IMG_DIR, "feature_importance_stage2.png"),
    )

    return dict(
        final_lgb      = final_lgb,
        final_xgb      = final_xgb,
        threshold      = thr_s2,
        oof_prob       = ens_prob_s2,
        oof_true       = y_s2.values,
        oof_metrics    = ens_m_s2,
        cv_lgb         = summarize(metrics_lgb),
        cv_xgb         = summarize(metrics_xgb),
    )


# ══════════════════════════════════════════════════════════════════════════════
# END-TO-END OOF EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_end_to_end(
    X: pd.DataFrame,
    y_3class: pd.Series,
    s1_oof_prob: np.ndarray,
    s1_thr: float,
    s2_oof_prob: np.ndarray,
    s2_thr: float,
) -> dict:
    """
    Reconstruct three-class OOF predictions from both stages and evaluate.
    Stage 2 OOF is only defined on non-half-metal samples — we reconstruct
    the full-length prediction array by filling class 2 where stage 1 fires.
    """
    y = y_3class.values
    s1_pred = (s1_oof_prob >= s1_thr).astype(int)   # 1 = half-metal

    # Start with stage 2 predictions on the non-half-metal subset
    mask_not_hm   = y_3class != 2
    idx_not_hm    = np.where(mask_not_hm)[0]

    final_pred = np.full(len(y), -1, dtype=int)

    # Where stage 1 predicts half-metal
    final_pred[s1_pred == 1] = 2

    # Where stage 1 predicts not-half-metal, use stage 2
    # s2_oof_prob is aligned to the not-half-metal subset
    s2_pred_subset = (s2_oof_prob >= s2_thr).astype(int)
    for i, idx in enumerate(idx_not_hm):
        if s1_pred[idx] == 0:   # stage 1 says not half-metal
            final_pred[idx] = s2_pred_subset[i]

    # Samples stage 1 misclassified as half-metal have final_pred=2 but
    # were actually class 0 or 1 — this is the residual error we measure.
    valid = final_pred != -1
    m = three_class_metrics(y[valid], final_pred[valid])
    cm = confusion_matrix(y[valid], final_pred[valid], labels=[0, 1, 2])

    print(f"\n{'═' * 60}")
    print(f"  END-TO-END THREE-CLASS OOF")
    print(f"{'═' * 60}")
    for k, v in m.items():
        print(f"    {k}: {v:.4f}")
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    header = f"  {'':>12}" + "".join(f"  {CLASS_NAMES[i]:>12}" for i in range(3))
    print(header)
    for i in range(3):
        row = f"  {CLASS_NAMES[i]:>12}" + "".join(f"  {cm[i,j]:>12d}" for j in range(3))
        print(row)

    return {"metrics": m, "confusion_matrix": cm.tolist()}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    # Load
    df    = pd.read_pickle(DATA_PATH)
    feats = load_features()

    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns],
                errors="ignore")[feats]
    y = df["hm_class"]

    print(f"\n{'═' * 60}")
    print(f"  hm_class — Two-Stage Pipeline")
    print(f"{'═' * 60}")
    print(f"  Total samples: {len(y)}")
    for cls, label in CLASS_NAMES.items():
        print(f"  Class {cls} ({label}): {(y == cls).sum():6d}")
    print(f"  Features: {len(feats)}")

    # Stage 1
    s1 = train_stage1(X, y, feats)

    # Stage 2
    s2 = train_stage2(X, y, feats)

    # End-to-end evaluation
    e2e = evaluate_end_to_end(
        X, y,
        s1["oof_prob"], s1["threshold"],
        s2["oof_prob"], s2["threshold"],
    )

    # Final confusion figure
    y_arr = y.values
    s1_pred = (s1["oof_prob"] >= s1["threshold"]).astype(int)
    mask_not_hm = y != 2
    idx_not_hm  = np.where(mask_not_hm)[0]
    s2_pred_sub = (s2["oof_prob"] >= s2["threshold"]).astype(int)

    final_pred = np.full(len(y_arr), -1, dtype=int)
    final_pred[s1_pred == 1] = 2
    for i, idx in enumerate(idx_not_hm):
        if s1_pred[idx] == 0:
            final_pred[idx] = s2_pred_sub[i]

    valid = final_pred != -1
    plot_confusion(
        y_arr[valid], final_pred[valid],
        [CLASS_NAMES[i] for i in range(3)],
        "End-to-End Three-Class  (OOF)",
        os.path.join(IMG_DIR, "final_confusion.png"),
    )

    # Save models
    print(f"\n  Saving models → {OUT_DIR}/")
    for seed in UNDERSAMPLE_SEEDS:
        joblib.dump(s1["final_lgb"][seed],
                    os.path.join(OUT_DIR, f"stage1_lgbm_seed{seed}.pkl"))
        joblib.dump(s1["final_xgb"][seed],
                    os.path.join(OUT_DIR, f"stage1_xgb_seed{seed}.pkl"))
    joblib.dump(s2["final_lgb"], os.path.join(OUT_DIR, "stage2_lgbm.pkl"))
    joblib.dump(s2["final_xgb"], os.path.join(OUT_DIR, "stage2_xgb.pkl"))

    save_json(
        {"stage1_threshold": s1["threshold"], "stage2_threshold": s2["threshold"]},
        os.path.join(OUT_DIR, "thresholds.json"),
    )
    save_json(
        {
            "oof_metrics":   s1["oof_metrics"],
            "seed_cv": {str(s): summarize(s1["seed_cv_metrics"][s])
                        for s in UNDERSAMPLE_SEEDS},
            "undersample_seeds": UNDERSAMPLE_SEEDS,
            "threshold": s1["threshold"],
        },
        os.path.join(OUT_DIR, "stage1_metrics.json"),
    )
    save_json(
        {
            "oof_metrics": s2["oof_metrics"],
            "cv_lgbm":     s2["cv_lgb"],
            "cv_xgb":      s2["cv_xgb"],
            "threshold":   s2["threshold"],
        },
        os.path.join(OUT_DIR, "stage2_metrics.json"),
    )
    save_json(
        {
            "end_to_end": e2e["metrics"],
            "confusion_matrix": e2e["confusion_matrix"],
            "stage1_threshold": s1["threshold"],
            "stage2_threshold": s2["threshold"],
            "n_features":       len(feats),
        },
        os.path.join(OUT_DIR, "hm_class_metrics.json"),
    )

    print(f"\n{'═' * 60}")
    print(f"  Done.  All outputs in: {OUT_DIR}/")
    print(f"{'═' * 60}")


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def predict_hm_class(
    X_new: pd.DataFrame,
    model_dir: str = OUT_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (pred, proba_hm) where:
      pred     0=Conductor, 1=Insulator, 2=Half-metal
      proba_hm stage 1 half-metal probability (n_samples,)
    """
    thr = json.load(open(os.path.join(model_dir, "thresholds.json")))

    # Stage 1 — half-metal vs not
    s1_models_lgb = [
        joblib.load(os.path.join(model_dir, f"stage1_lgbm_seed{s}.pkl"))
        for s in UNDERSAMPLE_SEEDS
    ]
    s1_models_xgb = [
        joblib.load(os.path.join(model_dir, f"stage1_xgb_seed{s}.pkl"))
        for s in UNDERSAMPLE_SEEDS
    ]
    proba_hm = np.mean(
        [m.predict_proba(X_new)[:, 1] for m in s1_models_lgb + s1_models_xgb],
        axis=0,
    )
    is_hm = proba_hm >= thr["stage1_threshold"]

    # Stage 2 — conductor vs insulator for non-half-metals
    s2_lgb = joblib.load(os.path.join(model_dir, "stage2_lgbm.pkl"))
    s2_xgb = joblib.load(os.path.join(model_dir, "stage2_xgb.pkl"))
    proba_ins = (
        s2_lgb.predict_proba(X_new)[:, 1] +
        s2_xgb.predict_proba(X_new)[:, 1]
    ) / 2
    s2_pred = (proba_ins >= thr["stage2_threshold"]).astype(int)

    pred = np.where(is_hm, 2, s2_pred)
    return pred, proba_hm


if __name__ == "__main__":
    main()