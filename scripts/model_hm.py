#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_hm_class.py
─────────────────
Half-metal detection: one-shot ternary baseline vs. binary detector.

Approach A (baseline)     : Direct 3-class softmax, capped undersampling.
Approach B (contribution) : Binary HM-vs-rest detector with strict 1:1
                            undersampling and a tunable Youden-J threshold.

End-to-end 3-class predictions are reconstructed by routing non-HM
samples through an auxiliary C/I classifier, enabling a fair apples-to-
apples comparison with the one-shot baseline.

Bug fixes vs. previous version
───────────────────────────────
  1. precision_matched_threshold: thr[valid][0] not thr[valid][-1]
  2. XGBoost early_stopping_rounds moved to constructor (XGB >= 1.6)
  3. best_iteration_ → best_iteration for XGBoost (no underscore)
  4. C/I classifier XGBoost also gets early_stopping_rounds
  5. Comparison table uses E2E reconstructed 3-class metrics throughout

Outputs: models/hm_class/
         models/hm_class/images/
         models/hm_class/hm_class_metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

import lightgbm as lgb
import xgboost as xgb

matplotlib.use("Agg")

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

RANDOM_STATE      = 42
N_SPLITS          = 5
MAJORITY_CAP      = 3000
UNDERSAMPLE_SEEDS = [0, 7, 21, 42, 99]
EARLY_STOP        = 200

DATA_PATH     = Path("data/data_feat.pkl")
FEATURES_JSON = Path("data/selected_features.json")
OUT_DIR       = Path("models/hm_class")
IMG_DIR       = OUT_DIR / "images"

CLASS_NAMES = {0: "Conductor", 1: "Insulator", 2: "Half-metal"}


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = pd.read_pickle(DATA_PATH)
    with open(FEATURES_JSON) as f:
        all_feats = json.load(f)

    feats = all_feats.get("hm_class", [])
    if not feats:
        raise ValueError("Run feature_selection.py first.")

    drop_cols = [
        "compound", "spacegroup_relax", "Egap", "Egap_type",
        "Egap_type_numeric", "enthalpy_formation_atom",
        "composition", "elements", "hm_class",
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])[feats]
    y = df["hm_class"]

    print(f"Samples: {len(y)} | Features: {len(feats)}")
    for cls, name in CLASS_NAMES.items():
        print(f"  Class {cls} ({name}): {(y == cls).sum()}")

    return X, y, feats


# ═══════════════════════════════════════════════════════════════════════
# MODEL BUILDERS
# ═══════════════════════════════════════════════════════════════════════

def lgbm_clf(objective: str, n_classes: int | None = None) -> lgb.LGBMClassifier:
    base = dict(
        learning_rate    = 0.05,
        num_leaves       = 64,
        n_estimators     = 5000,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        importance_type  = "gain",
        random_state     = RANDOM_STATE,
        verbose          = -1,
    )
    if objective == "binary":
        base.update(objective="binary", min_data_in_leaf=20)
    else:
        base.update(objective="multiclass", num_class=n_classes,
                    min_data_in_leaf=10)
    return lgb.LGBMClassifier(**base)


def xgb_clf(objective: str, n_classes: int | None = None) -> xgb.XGBClassifier:
    # early_stopping_rounds lives in the constructor for XGBoost >= 1.6
    base = dict(
        learning_rate         = 0.05,
        max_depth             = 6,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        reg_alpha             = 0.05,
        reg_lambda            = 1.5,
        n_estimators          = 5000,
        tree_method           = "hist",
        random_state          = RANDOM_STATE,
        early_stopping_rounds = EARLY_STOP,
        verbosity             = 0,
    )
    if objective == "binary":
        base.update(objective="binary:logistic", eval_metric="auc",
                    min_child_weight=3)
    else:
        base.update(objective="multi:softprob", num_class=n_classes,
                    eval_metric="mlogloss")
    return xgb.XGBClassifier(**base)


# ═══════════════════════════════════════════════════════════════════════
# UNDERSAMPLING
# ═══════════════════════════════════════════════════════════════════════

def strict_11(X, y, minority_class, seed):
    rng        = np.random.default_rng(seed)
    n_minority = (y == minority_class).sum()
    idx        = []
    for cls in sorted(y.unique()):
        idx_cls  = y[y == cls].index
        n_sample = n_minority if cls != minority_class else len(idx_cls)
        chosen   = (rng.choice(idx_cls, size=n_sample, replace=False)
                    if n_sample < len(idx_cls) else idx_cls)
        idx.extend(chosen)
    rng.shuffle(idx)
    return X.loc[idx], y.loc[idx]


def capped(X, y, minority_class, cap, seed):
    rng = np.random.default_rng(seed)
    idx = list(y[y == minority_class].index)
    for cls in sorted(y.unique()):
        if cls == minority_class:
            continue
        idx_cls = y[y == cls].index
        idx.extend(rng.choice(idx_cls, size=min(cap, len(idx_cls)),
                               replace=False))
    rng.shuffle(idx)
    return X.loc[idx], y.loc[idx]


# ═══════════════════════════════════════════════════════════════════════
# METRICS & THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════

def binary_metrics(y_true, prob, thr) -> dict:
    pred   = (prob >= thr).astype(int)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, pred, labels=[0, 1], zero_division=0)
    return {
        "threshold":     float(thr),
        "roc_auc":       float(roc_auc_score(y_true, prob)),
        "pr_auc":        float(average_precision_score(y_true, prob)),
        "macro_f1":      float(f1_score(y_true, pred, average="macro",
                                        zero_division=0)),
        "balanced_acc":  float(balanced_accuracy_score(y_true, pred)),
        "f1_class0":     float(f[0]),
        "f1_class1":     float(f[1]),
        "precision_class1": float(p[1]),
        "recall_class1":    float(r[1]),
    }


def ternary_metrics(y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    return {
        "macro_f1":      float(f1_score(y_true, y_pred, average="macro",
                                        zero_division=0)),
        "weighted_f1":   float(f1_score(y_true, y_pred, average="weighted",
                                        zero_division=0)),
        "balanced_acc":  float(balanced_accuracy_score(y_true, y_pred)),
        "f1_class0":     float(f[0]),
        "precision_class0": float(p[0]),
        "recall_class0":    float(r[0]),
        "f1_class1":     float(f[1]),
        "precision_class1": float(p[1]),
        "recall_class1":    float(r[1]),
        "f1_class2":     float(f[2]),
        "precision_class2": float(p[2]),
        "recall_class2":    float(r[2]),
    }


def youden_threshold(y_true, prob) -> float:
    fpr, tpr, thr = roc_curve(y_true, prob)
    return float(thr[np.argmax(tpr - fpr)])


def f1_optimal_threshold(y_true, prob) -> float:
    thresholds       = np.linspace(0.01, 0.99, 500)
    best_f1, best_t  = 0.0, 0.5
    for t in thresholds:
        f = f1_score(y_true, (prob >= t).astype(int),
                     pos_label=1, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return float(best_t)


def precision_matched_threshold(y_true, prob, target_precision) -> float:
    """
    Lowest threshold at which precision >= target_precision.
    sklearn's precision_recall_curve returns thresholds in ascending order.
    thr[valid][0] is the lowest valid threshold → maximises recall.
    """
    prec, _, thr = precision_recall_curve(y_true, prob)
    valid = prec[:-1] >= target_precision
    if not valid.any():
        return float(thr[-1])
    return float(thr[valid][0])   # FIX: [0] not [-1]


# ═══════════════════════════════════════════════════════════════════════
# TRAINING: ONE-SHOT TERNARY BASELINE
# ═══════════════════════════════════════════════════════════════════════

def train_oneshot(X, y) -> dict:
    print(f"\n{'='*60}\n  BASELINE: One-shot ternary classifier\n{'='*60}")
    cv    = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                            random_state=RANDOM_STATE)
    folds = list(cv.split(X, y))

    oof_proba = np.zeros((len(y), 3))
    y_oof     = np.full(len(y), -1, dtype=int)
    best_iters = {"lgb": {}, "xgb": {}}

    for seed in UNDERSAMPLE_SEEDS:
        it_lgb, it_xgb = [], []
        for fold, (tr, va) in enumerate(folds, 1):
            if seed == UNDERSAMPLE_SEEDS[0]:
                y_oof[va] = y.iloc[va].values

            Xtr, ytr = capped(X.iloc[tr], y.iloc[tr],
                               minority_class=2, cap=MAJORITY_CAP, seed=seed)
            Xva, yva = X.iloc[va], y.iloc[va]

            m_lgb = lgbm_clf("multiclass", n_classes=3)
            m_xgb = xgb_clf("multiclass",  n_classes=3)

            m_lgb.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                      callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                                 lgb.log_evaluation(-1)])
            m_xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

            it_lgb.append(m_lgb.best_iteration_)
            it_xgb.append(m_xgb.best_iteration)   # no underscore for XGB

            p = (m_lgb.predict_proba(Xva) + m_xgb.predict_proba(Xva)) / 2
            oof_proba[va] += p

            fm = ternary_metrics(yva.values, np.argmax(p, axis=1))
            print(f"  seed {seed} fold {fold} | macro_F1={fm['macro_f1']:.4f} "
                  f"HM_F1={fm['f1_class2']:.4f} "
                  f"HM_P={fm['precision_class2']:.4f} "
                  f"HM_R={fm['recall_class2']:.4f}")

        best_iters["lgb"][seed] = it_lgb
        best_iters["xgb"][seed] = it_xgb

    oof_proba /= len(UNDERSAMPLE_SEEDS)
    oof_pred   = np.argmax(oof_proba, axis=1)
    oof_m      = ternary_metrics(y_oof, oof_pred)

    print(f"\n  OOF | macro_F1={oof_m['macro_f1']:.4f}  "
          f"HM_F1={oof_m['f1_class2']:.4f}  "
          f"HM_P={oof_m['precision_class2']:.4f}  "
          f"HM_R={oof_m['recall_class2']:.4f}")

    # Final models on full data
    final = {"lgb": {}, "xgb": {}}
    for seed in UNDERSAMPLE_SEEDS:
        n_lgb = max(1, int(np.mean(best_iters["lgb"][seed])))
        n_xgb = max(1, int(np.mean(best_iters["xgb"][seed])))
        Xb, yb = capped(X, y, minority_class=2, cap=MAJORITY_CAP, seed=seed)
        final["lgb"][seed] = lgbm_clf("multiclass", 3).set_params(
            n_estimators=n_lgb)
        final["xgb"][seed] = xgb_clf("multiclass",  3).set_params(
            n_estimators=n_xgb, early_stopping_rounds=None)
        final["lgb"][seed].fit(Xb, yb)
        final["xgb"][seed].fit(Xb, yb)

    return dict(oof_proba=oof_proba, oof_pred=oof_pred,
                oof_true=y_oof, oof_metrics=oof_m, final=final)


# ═══════════════════════════════════════════════════════════════════════
# TRAINING: BINARY HM DETECTOR
# ═══════════════════════════════════════════════════════════════════════

def train_hm_detector(X, y_3class, feature_names) -> dict:
    y_bin  = (y_3class == 2).astype(int)
    n_hm   = (y_bin == 1).sum()
    n_not  = (y_bin == 0).sum()
    print(f"\n{'='*60}\n  CONTRIBUTION: Binary HM detector\n{'='*60}")
    print(f"  HM: {n_hm} | Not-HM: {n_not} | Ratio {n_not/n_hm:.1f}:1")

    cv         = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                 random_state=RANDOM_STATE)
    folds      = list(cv.split(X, y_bin))
    oof_lgb    = np.zeros(len(y_bin))
    oof_xgb    = np.zeros(len(y_bin))
    y_oof      = np.zeros(len(y_bin), dtype=int)
    best_iters = {"lgb": {}, "xgb": {}}
    importances: list[np.ndarray] = []

    for seed in UNDERSAMPLE_SEEDS:
        it_lgb, it_xgb = [], []
        for fold, (tr, va) in enumerate(folds, 1):
            if seed == UNDERSAMPLE_SEEDS[0]:
                y_oof[va] = y_bin.iloc[va].values

            Xtr, ytr = strict_11(X.iloc[tr], y_bin.iloc[tr],
                                  minority_class=1, seed=seed)
            Xva, yva = X.iloc[va], y_bin.iloc[va]

            m_lgb = lgbm_clf("binary")
            m_xgb = xgb_clf("binary")

            m_lgb.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                      callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                                 lgb.log_evaluation(-1)])
            m_xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

            it_lgb.append(m_lgb.best_iteration_)
            it_xgb.append(m_xgb.best_iteration)
            importances.append(m_lgb.feature_importances_)

            p = (m_lgb.predict_proba(Xva)[:, 1]
                 + m_xgb.predict_proba(Xva)[:, 1]) / 2
            oof_lgb[va] += m_lgb.predict_proba(Xva)[:, 1]
            oof_xgb[va] += m_xgb.predict_proba(Xva)[:, 1]

            fm = binary_metrics(yva.values, p, thr=0.5)
            print(f"  seed {seed} fold {fold} | "
                  f"AUC={fm['roc_auc']:.4f} "
                  f"P={fm['precision_class1']:.4f} "
                  f"R={fm['recall_class1']:.4f} "
                  f"F1={fm['f1_class1']:.4f}")

        best_iters["lgb"][seed] = it_lgb
        best_iters["xgb"][seed] = it_xgb

    oof_lgb /= len(UNDERSAMPLE_SEEDS)
    oof_xgb /= len(UNDERSAMPLE_SEEDS)
    oof_prob = (oof_lgb + oof_xgb) / 2

    thr_y  = youden_threshold(y_oof, oof_prob)
    thr_f1 = f1_optimal_threshold(y_oof, oof_prob)

    print(f"\n  OOF | AUC={roc_auc_score(y_oof, oof_prob):.4f}  "
          f"PR-AUC={average_precision_score(y_oof, oof_prob):.4f}")
    print(f"  Thresholds | Youden-J={thr_y:.3f} | F1-opt={thr_f1:.3f}")

    # Final models
    final = {"lgb": {}, "xgb": {}}
    for seed in UNDERSAMPLE_SEEDS:
        n_lgb = max(1, int(np.mean(best_iters["lgb"][seed])))
        n_xgb = max(1, int(np.mean(best_iters["xgb"][seed])))
        Xb, yb = strict_11(X, y_bin, minority_class=1, seed=seed)
        final["lgb"][seed] = lgbm_clf("binary").set_params(n_estimators=n_lgb)
        final["xgb"][seed] = xgb_clf("binary").set_params(
            n_estimators=n_xgb, early_stopping_rounds=None)
        final["lgb"][seed].fit(Xb, yb)
        final["xgb"][seed].fit(Xb, yb)

    return dict(
        oof_prob     = oof_prob,
        oof_true     = y_oof,
        thresholds   = {"youden": thr_y, "f1_opt": thr_f1, "prec_match": 0.5},
        final        = final,
        importances  = importances,
        feature_names= feature_names,
    )


# ═══════════════════════════════════════════════════════════════════════
# TRAINING: AUXILIARY C/I CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════

def train_ci(X, y_3class) -> dict:
    mask = y_3class != 2
    Xs   = X.loc[mask]
    ys   = y_3class.loc[mask].copy()
    print(f"\n{'='*60}\n  AUXILIARY: C/I classifier\n{'='*60}")
    print(f"  Non-HM: {len(ys)} | C: {(ys==0).sum()} | I: {(ys==1).sum()}")

    cv         = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                 random_state=RANDOM_STATE)
    oof_lgb    = np.zeros(len(ys))
    oof_xgb    = np.zeros(len(ys))
    y_oof      = np.zeros(len(ys), dtype=int)
    best_iters = {"lgb": [], "xgb": []}

    for fold, (tr, va) in enumerate(cv.split(Xs, ys), 1):
        y_oof[va] = ys.iloc[va].values
        Xtr, ytr  = Xs.iloc[tr], ys.iloc[tr]
        Xva, yva  = Xs.iloc[va], ys.iloc[va]

        m_lgb = lgbm_clf("binary")
        m_xgb = xgb_clf("binary")   # early_stopping_rounds in constructor

        m_lgb.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                             lgb.log_evaluation(-1)])
        m_xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

        best_iters["lgb"].append(m_lgb.best_iteration_)
        best_iters["xgb"].append(m_xgb.best_iteration)  # no underscore

        oof_lgb[va] = m_lgb.predict_proba(Xva)[:, 1]
        oof_xgb[va] = m_xgb.predict_proba(Xva)[:, 1]

    oof_prob = (oof_lgb + oof_xgb) / 2
    thr      = youden_threshold(y_oof, oof_prob)
    oof_pred = (oof_prob >= thr).astype(int)

    print(f"  OOF | AUC={roc_auc_score(y_oof, oof_prob):.4f}  "
          f"Bal-Acc={balanced_accuracy_score(y_oof, oof_pred):.4f}  "
          f"thr={thr:.3f}")

    n_lgb = max(1, int(np.mean(best_iters["lgb"])))
    n_xgb = max(1, int(np.mean(best_iters["xgb"])))
    fl    = lgbm_clf("binary").set_params(n_estimators=n_lgb)
    fx    = xgb_clf("binary").set_params(
        n_estimators=n_xgb, early_stopping_rounds=None)
    fl.fit(Xs, ys)
    fx.fit(Xs, ys)

    return dict(oof_prob=oof_prob, oof_true=y_oof,
                threshold=thr, final_lgb=fl, final_xgb=fx)


# ═══════════════════════════════════════════════════════════════════════
# END-TO-END EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def evaluate_e2e(y_3class, hm_prob, hm_thr, ci, X, label) -> dict:
    """
    Route samples through:
      detector says HM  → class 2
      detector says not → C/I final model → class 0 or 1
    All samples are classified; none are dropped.
    """
    y        = y_3class.values
    hm_pred  = (hm_prob >= hm_thr).astype(int)
    final_p  = np.full(len(y), -1, dtype=int)
    final_p[hm_pred == 1] = 2

    non_hm = hm_pred == 0
    if non_hm.any():
        p_ins = (
            ci["final_lgb"].predict_proba(X.loc[non_hm])[:, 1]
            + ci["final_xgb"].predict_proba(X.loc[non_hm])[:, 1]
        ) / 2
        final_p[non_hm] = (p_ins >= ci["threshold"]).astype(int)

    assert (final_p != -1).all(), "Some samples were not classified!"

    m  = ternary_metrics(y, final_p)
    cm = confusion_matrix(y, final_p, labels=[0, 1, 2])
    print(f"  E2E ({label:18s}) | macro_F1={m['macro_f1']:.4f}  "
          f"HM_F1={m['f1_class2']:.4f}  "
          f"HM_P={m['precision_class2']:.4f}  "
          f"HM_R={m['recall_class2']:.4f}")
    return dict(label=label, metrics=m, confusion=cm.tolist())


# ═══════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════

def _savefig(fig, path, msg=True):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if msg:
        print(f"  saved {path}")


def plot_confusion_matrix(y_true, y_pred, title, fname):
    labels = [CLASS_NAMES[i] for i in range(3)]
    cm      = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, data, fmt, subtitle in [
        (axes[0], cm_norm, ".2f", "Row-normalised"),
        (axes[1], cm,      "d",   "Raw counts"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=labels, yticklabels=labels,
                    ax=ax, linewidths=0.5, annot_kws={"size": 10})
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True",      fontsize=10)
        ax.set_title(subtitle,     fontsize=11)

    plt.tight_layout()
    _savefig(fig, fname)


def plot_roc_pr(y_true, prob, title, fname):
    fpr, tpr, _   = roc_curve(y_true, prob)
    prec, rec, _  = precision_recall_curve(y_true, prob)
    auc_roc       = roc_auc_score(y_true, prob)
    auc_pr        = average_precision_score(y_true, prob)
    prevalence    = y_true.mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax1.plot(fpr, tpr, lw=2, color="steelblue",
             label=f"ROC-AUC = {auc_roc:.4f}")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(fontsize=9)
    ax1.grid(ls="--", alpha=0.3)

    ax2.plot(rec, prec, lw=2, color="steelblue",
             label=f"PR-AUC = {auc_pr:.4f}")
    ax2.axhline(prevalence, color="k", ls="--",
                label=f"Random (prev={prevalence:.3f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision–Recall Curve")
    ax2.legend(fontsize=9)
    ax2.grid(ls="--", alpha=0.3)

    plt.tight_layout()
    _savefig(fig, fname)


def plot_threshold_sweep(y_true, prob, thresholds, fname):
    ts             = np.linspace(0.01, 0.99, 400)
    f1s, precs, recs = [], [], []
    for t in ts:
        pred = (prob >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            y_true, pred, labels=[0, 1], zero_division=0)
        f1s.append(f[1]);  precs.append(p[1]);  recs.append(r[1])

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(ts, f1s,   lw=2, color="steelblue",          label="F1")
    ax.plot(ts, precs, lw=2, color="tomato",   alpha=0.8, label="Precision")
    ax.plot(ts, recs,  lw=2, color="seagreen", alpha=0.8, label="Recall")

    colors = {"youden": "black", "f1_opt": "seagreen", "prec_match": "tomato"}
    labels = {"youden": "Youden-J", "f1_opt": "F1-optimal",
              "prec_match": "Prec-matched"}
    for k, thr in thresholds.items():
        ax.axvline(thr, color=colors[k], ls="--", lw=1.5,
                   label=f"{labels[k]} = {thr:.3f}")

    ax.set_xlabel("Threshold");  ax.set_ylabel("Score")
    ax.set_title("HM Binary Detector: Threshold Sweep")
    ax.legend(fontsize=8);  ax.grid(ls="--", alpha=0.3)
    plt.tight_layout()
    _savefig(fig, fname)


def plot_pr_comparison(hm_prob, hm_true, oneshot_m, thresholds, fname):
    """
    Full binary PR curve with one-shot fixed point and three detector
    operating points — the key figure for the paper.
    """
    prec, rec, _ = precision_recall_curve(hm_true, hm_prob)
    auc_pr       = average_precision_score(hm_true, hm_prob)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(rec, prec, lw=2.5, color="steelblue", zorder=2,
            label=f"Binary detector  PR-AUC = {auc_pr:.4f}")
    ax.axhline(hm_true.mean(), color="grey", ls="--",
               label=f"Random (prev = {hm_true.mean():.3f})")

    # one-shot fixed point
    ax.scatter(
        [oneshot_m["recall_class2"]], [oneshot_m["precision_class2"]],
        marker="*", s=400, color="orange", zorder=6,
        label=(f"One-shot  "
               f"P={oneshot_m['precision_class2']:.3f}  "
               f"R={oneshot_m['recall_class2']:.3f}  "
               f"F1={oneshot_m['f1_class2']:.3f}"),
    )

    pt_colors = {"youden": "black", "f1_opt": "seagreen", "prec_match": "tomato"}
    pt_labels = {"youden": "Youden-J", "f1_opt": "F1-optimal",
                 "prec_match": "Prec-matched"}
    for k, thr in thresholds.items():
        pred   = (hm_prob >= thr).astype(int)
        p_, r_, f_, _ = precision_recall_fscore_support(
            hm_true, pred, labels=[0, 1], zero_division=0)
        ax.scatter(
            [r_[1]], [p_[1]], marker="o", s=150,
            color=pt_colors[k], zorder=5,
            label=(f"{pt_labels[k]}  "
                   f"P={p_[1]:.3f}  R={r_[1]:.3f}  F1={f_[1]:.3f}"),
        )

    ax.set_xlabel("Recall (Half-metal)",    fontsize=11)
    ax.set_ylabel("Precision (Half-metal)", fontsize=11)
    ax.set_title("One-Shot vs. Binary Detector: Precision–Recall Space",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim([0, 1.02]);  ax.set_ylim([0, 1.05])
    ax.grid(ls="--", alpha=0.3)
    plt.tight_layout()
    _savefig(fig, fname)


def plot_comparison_bar(rows, fname):
    labels = [r["model"] for r in rows]
    prec   = [r["precision_hm"] for r in rows]
    rec    = [r["recall_hm"]    for r in rows]
    f1     = [r["f1_hm"]        for r in rows]
    x, w   = np.arange(len(labels)), 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 2), 5))
    ax.bar(x - w, prec, w, label="Precision", color="tomato",    alpha=0.85)
    ax.bar(x,     rec,  w, label="Recall",    color="steelblue", alpha=0.85)
    ax.bar(x + w, f1,   w, label="F1",        color="seagreen",  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score");  ax.set_ylim([0, 1.05])
    ax.set_title("Half-Metal Detection: Model Comparison",
                 fontsize=12, fontweight="bold")
    ax.legend();  ax.grid(axis="y", ls="--", alpha=0.4)
    plt.tight_layout()
    _savefig(fig, fname)


def plot_feature_importance(importances, names, fname, top_n=20):
    mean_imp = np.mean(importances, axis=0)
    std_imp  = np.std(importances,  axis=0)
    idx      = np.argsort(mean_imp)[-top_n:][::-1]
    y_pos    = np.arange(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.30)))
    ax.barh(y_pos, mean_imp[idx][::-1], xerr=std_imp[idx][::-1],
            color="steelblue", ecolor="grey", alpha=0.85,
            height=0.7, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([names[i] for i in idx[::-1]], fontsize=9)
    ax.set_xlabel("Mean Gain Importance (± std)")
    ax.set_title(f"Binary Detector: LGBM Feature Importance (Top {top_n})",
                 fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    _savefig(fig, fname)


def plot_per_class_cv_stability(seed_metrics_list, fname):
    """
    Box plots of per-fold F1 for each class across all seeds × folds,
    showing variance of the one-shot baseline.
    """
    records = []
    for seed_folds in seed_metrics_list:
        for fm in seed_folds:
            for cls in range(3):
                records.append({
                    "class": CLASS_NAMES[cls],
                    "F1":    fm[f"f1_class{cls}"],
                })
    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="class", y="F1",
                palette=["steelblue", "seagreen", "tomato"], ax=ax,
                width=0.45, linewidth=1.2)
    ax.set_title("One-Shot: Per-Class F1 Stability (5 seeds × 5 folds)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Class");  ax.set_ylabel("F1 Score")
    ax.set_ylim([0, 1.05]);  ax.grid(axis="y", ls="--", alpha=0.4)
    plt.tight_layout()
    _savefig(fig, fname)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    X, y, feats = load_data()

    # ── A: One-shot baseline ─────────────────────────────────────────
    oneshot  = train_oneshot(X, y)

    # ── B: Binary detector ───────────────────────────────────────────
    detector = train_hm_detector(X, y, feats)

    # Set precision-matched threshold from one-shot baseline precision
    one_p = oneshot["oof_metrics"]["precision_class2"]
    detector["thresholds"]["prec_match"] = precision_matched_threshold(
        detector["oof_true"], detector["oof_prob"], one_p)
    print(f"\n  Precision-matched threshold = "
          f"{detector['thresholds']['prec_match']:.3f}  "
          f"(target P = {one_p:.3f})")

    # ── C: Auxiliary C/I classifier ───────────────────────────────────
    ci = train_ci(X, y)

    # ── End-to-end 3-class comparison ────────────────────────────────
    print(f"\n{'='*60}\n  END-TO-END 3-CLASS COMPARISON\n{'='*60}")
    e2e = {}
    for key, label in [("youden",     "Youden-J"),
                        ("f1_opt",    "F1-optimal"),
                        ("prec_match","Precision-matched")]:
        e2e[key] = evaluate_e2e(
            y, detector["oof_prob"],
            detector["thresholds"][key], ci, X, label)

    # ── Comparison table ─────────────────────────────────────────────
    comparison = [{
        "model":        "One-shot (softmax)",
        "precision_hm": oneshot["oof_metrics"]["precision_class2"],
        "recall_hm":    oneshot["oof_metrics"]["recall_class2"],
        "f1_hm":        oneshot["oof_metrics"]["f1_class2"],
        "macro_f1":     oneshot["oof_metrics"]["macro_f1"],
    }]
    for key, label in [("youden",     "Binary (Youden-J)"),
                        ("f1_opt",    "Binary (F1-optimal)"),
                        ("prec_match","Binary (Precision-matched)")]:
        m = e2e[key]["metrics"]
        comparison.append({
            "model":        label,
            "precision_hm": m["precision_class2"],
            "recall_hm":    m["recall_class2"],
            "f1_hm":        m["f1_class2"],
            "macro_f1":     m["macro_f1"],
        })

    print(f"\n{'='*72}")
    print(f"  {'Model':<32} {'P_hm':>7} {'R_hm':>7} {'F1_hm':>7} "
          f"{'Macro-F1':>9}")
    print(f"  {'-'*65}")
    for row in comparison:
        print(f"  {row['model']:<32} "
              f"{row['precision_hm']:>7.4f} "
              f"{row['recall_hm']:>7.4f} "
              f"{row['f1_hm']:>7.4f} "
              f"{row['macro_f1']:>9.4f}")
    print(f"{'='*72}")

    # ── Plots ─────────────────────────────────────────────────────────
    print(f"\n  Generating figures → {IMG_DIR}/")

    plot_confusion_matrix(
        oneshot["oof_true"], oneshot["oof_pred"],
        "One-Shot Baseline: Confusion Matrix (OOF)",
        IMG_DIR / "oneshot_confusion.png")

    for key, label in [("youden",     "Youden-J"),
                        ("f1_opt",    "F1-optimal"),
                        ("prec_match","Precision-matched")]:
        thr      = detector["thresholds"][key]
        pred_bin = (detector["oof_prob"] >= thr).astype(int)
        final_p  = np.full(len(y), -1, dtype=int)
        final_p[pred_bin == 1] = 2
        non_hm   = pred_bin == 0
        if non_hm.any():
            p_ins = (
                ci["final_lgb"].predict_proba(X.loc[non_hm])[:, 1]
                + ci["final_xgb"].predict_proba(X.loc[non_hm])[:, 1]
            ) / 2
            final_p[non_hm] = (p_ins >= ci["threshold"]).astype(int)
        plot_confusion_matrix(
            y.values, final_p,
            f"Binary Detector ({label}): Confusion Matrix (E2E OOF)",
            IMG_DIR / f"e2e_confusion_{key}.png")

    plot_roc_pr(
        detector["oof_true"], detector["oof_prob"],
        "Binary HM Detector: ROC & PR (OOF)",
        IMG_DIR / "detector_roc_pr.png")

    plot_threshold_sweep(
        detector["oof_true"], detector["oof_prob"],
        detector["thresholds"],
        IMG_DIR / "detector_threshold_sweep.png")

    plot_pr_comparison(
        detector["oof_prob"], detector["oof_true"],
        oneshot["oof_metrics"], detector["thresholds"],
        IMG_DIR / "comparison_pr_curve.png")

    plot_comparison_bar(comparison, IMG_DIR / "comparison_bar.png")

    plot_feature_importance(
        detector["importances"], detector["feature_names"],
        IMG_DIR / "detector_feature_importance.png")

    # ── Save models ───────────────────────────────────────────────────
    print(f"\n  Saving models → {OUT_DIR}/")
    for seed in UNDERSAMPLE_SEEDS:
        joblib.dump(oneshot["final"]["lgb"][seed],
                    OUT_DIR / f"oneshot_lgb_seed{seed}.pkl")
        joblib.dump(oneshot["final"]["xgb"][seed],
                    OUT_DIR / f"oneshot_xgb_seed{seed}.pkl")
        joblib.dump(detector["final"]["lgb"][seed],
                    OUT_DIR / f"detector_lgb_seed{seed}.pkl")
        joblib.dump(detector["final"]["xgb"][seed],
                    OUT_DIR / f"detector_xgb_seed{seed}.pkl")
    joblib.dump(ci["final_lgb"], OUT_DIR / "ci_lgb.pkl")
    joblib.dump(ci["final_xgb"], OUT_DIR / "ci_xgb.pkl")

    # ── Save metrics JSON ─────────────────────────────────────────────
    results = {
        "oneshot": oneshot["oof_metrics"],
        "binary_detector": {
            "roc_auc":   float(roc_auc_score(
                             detector["oof_true"], detector["oof_prob"])),
            "pr_auc":    float(average_precision_score(
                             detector["oof_true"], detector["oof_prob"])),
            "thresholds": {k: float(v)
                           for k, v in detector["thresholds"].items()},
            "oof_at_youden": binary_metrics(
                detector["oof_true"], detector["oof_prob"],
                detector["thresholds"]["youden"]),
        },
        "e2e": {k: v["metrics"] for k, v in e2e.items()},
        "comparison": comparison,
        "n_features": len(feats),
    }
    out_json = OUT_DIR / "hm_class_metrics.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n  Metrics saved → {out_json}")
    print(f"\n{'='*60}\n  Done. All outputs in {OUT_DIR}/\n{'='*60}")


if __name__ == "__main__":
    main()