#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
topk_analysis.py
────────────────
Evaluate model performance as a function of top-k SHAP features.

For each task, loads the SHAP importance CSV(s), selects the top-k
features for k in K_LIST, and runs cross-validated evaluation.
Results saved to topk/results.csv; one summary figure per task.

Bugs fixed vs. original:
  - Early stopping no longer uses an inner train_test_split on the
    fold training set (which leaks validation signal). The fold
    validation set is used directly as the eval_set.
  - eval_metric="l2" replaced with "rmse" for consistency.
  - Supercon glob logic was broken (if-block was overwritten by
    the unconditional assignment below it). Fixed.
  - Aggregation key collision: result dict now namespaced by task
    and model to avoid pandas column conflicts.
  - One meaningful plot per task (primary metric vs k, all models)
    instead of one plot per task × metric combination.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

OUT_DIR          = Path("topk")
SHAP_ROOT        = Path("shap_outputs")
DATA_PATH        = "data/data_feat.pkl"
SUPERCON_PATH    = "data/supercon_feat.pkl"

K_LIST           = [5, 6, 7, 8, 9, 10, 11, 12]
N_SPLITS         = 5
RANDOM_STATE     = 42
UNDERSAMPLE_SEEDS = [0, 7, 21, 42, 99]

# Primary metric shown in the summary plot for each task
PRIMARY_METRIC = {
    "egap":      "mae",
    "enthalpy":  "mae",
    "egap_type": "f1",
    "hm_class":  "f1",
    "supercon":  "mae",
}

TASKS = {
    "egap":      ("reg",     "Egap",                    DATA_PATH),
    "enthalpy":  ("reg",     "enthalpy_formation_atom", DATA_PATH),
    "egap_type": ("clf_bin", "Egap_type_numeric",       DATA_PATH),
    "hm_class":  ("clf_bin", "hm_class",                DATA_PATH),  # binary now
    "supercon":  ("reg",     "Tc",                      SUPERCON_PATH),
}
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_top_features(csv_path: str, k: int) -> list[str]:
    df = pd.read_csv(csv_path)
    if "feature" not in df.columns:
        df.columns = ["feature", "importance"]
    df = df.sort_values(by=df.columns[1], ascending=False)
    return df["feature"].head(k).tolist()


def reg_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "mae":  float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2":   float(r2_score(y, p)),
    }


def clf_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "f1":      float(f1_score(y, p, average="macro", zero_division=0)),
        "bal_acc": float(balanced_accuracy_score(y, p)),
    }


def undersample_strict(
    X: pd.DataFrame,
    y: pd.Series,
    minority_class: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    rng     = np.random.default_rng(seed)
    idx_min = list(y[y == minority_class].index)
    n_min   = len(idx_min)
    idx_all = list(idx_min)

    for cls in sorted(y.unique()):
        if cls == minority_class:
            continue
        idx_cls = y[y == cls].index
        chosen  = rng.choice(idx_cls, size=min(n_min, len(idx_cls)),
                             replace=False)
        idx_all.extend(chosen)

    idx_all = np.array(idx_all)
    rng.shuffle(idx_all)
    return X.loc[idx_all], y.loc[idx_all]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def lgbm_reg() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        learning_rate    = 0.05,
        num_leaves       = 64,
        n_estimators     = 5000,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        random_state     = RANDOM_STATE,
        verbose          = -1,
    )


def lgbm_clf(seed: int = RANDOM_STATE) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        learning_rate    = 0.05,
        num_leaves       = 64,
        n_estimators     = 5000,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        random_state     = seed,
        verbose          = -1,
    )


def _es_callbacks() -> list:
    return [lgb.early_stopping(200, verbose=False),
            lgb.log_evaluation(period=-1)]


# ══════════════════════════════════════════════════════════════════════════════
# CV RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def cv_regression(X: pd.DataFrame, y: pd.Series) -> list[dict]:
    """5-fold CV for regression. Eval set = fold validation (no inner split)."""
    cv           = KFold(n_splits=N_SPLITS, shuffle=True,
                         random_state=RANDOM_STATE)
    fold_metrics = []

    for tr, va in cv.split(X):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]

        m = lgbm_reg()
        m.fit(
            Xtr, ytr,
            eval_set   = [(Xva, yva)],
            eval_metric= "rmse",           # consistent with XGB side
            callbacks  = _es_callbacks(),
        )
        fold_metrics.append(reg_metrics(yva.values, m.predict(Xva)))

    return fold_metrics


def cv_binary(X: pd.DataFrame, y: pd.Series) -> list[dict]:
    """5-fold stratified CV for binary classification with seed ensemble."""
    cv           = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                   random_state=RANDOM_STATE)
    fold_metrics = []

    for tr, va in cv.split(X, y):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]

        probs_all = []
        for seed in UNDERSAMPLE_SEEDS:
            Xb, yb = undersample_strict(Xtr, ytr, 1, seed)
            m = lgbm_clf(seed)
            m.fit(
                Xb, yb,
                eval_set   = [(Xva, yva)],
                eval_metric= "binary_logloss",
                callbacks  = _es_callbacks(),
            )
            probs_all.append(m.predict_proba(Xva)[:, 1])

        preds = (np.mean(probs_all, axis=0) >= 0.5).astype(int)
        fold_metrics.append(clf_metrics(yva.values, preds))

    return fold_metrics


def cv_twostage(X: pd.DataFrame, y: pd.Series) -> list[dict]:
    """
    5-fold stratified CV for the two-stage half-metal classifier.
    Stage 1: binary (half-metal vs. not) with seed ensemble.
    Stage 2: conductor/insulator on non-half-metal samples.
    Threshold fixed at 0.5 for the top-k sensitivity analysis
    (threshold tuning is performed in the main hm_class pipeline).
    """
    cv           = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                   random_state=RANDOM_STATE)
    fold_metrics = []

    for tr, va in cv.split(X, y):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]

        # Stage 1 — half-metal detector
        ytr_s1    = (ytr == 2).astype(int)
        probs_s1  = []
        for seed in UNDERSAMPLE_SEEDS:
            Xb, yb = undersample_strict(Xtr, ytr_s1, 1, seed)
            m1 = lgbm_clf(seed)
            m1.fit(
                Xb, yb,
                eval_set   = [(Xva, yva_s1 := (yva == 2).astype(int))],
                eval_metric= "binary_logloss",
                callbacks  = _es_callbacks(),
            )
            probs_s1.append(m1.predict_proba(Xva)[:, 1])

        pred_s1 = np.mean(probs_s1, axis=0) >= 0.5

        # Stage 2 — conductor / insulator
        mask_tr = ytr != 2
        m2 = lgbm_clf()
        m2.fit(
            Xtr[mask_tr], ytr[mask_tr],
            eval_set   = [(Xva[~pred_s1], yva[~pred_s1])
                          if (~pred_s1).sum() > 1
                          else (Xtr[mask_tr].iloc[:1],
                                ytr[mask_tr].iloc[:1])],
            eval_metric= "multi_logloss",
            callbacks  = _es_callbacks(),
        )

        final          = m2.predict(Xva).copy()
        final[pred_s1] = 2
        fold_metrics.append(clf_metrics(yva.values, final))

    return fold_metrics


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

records = []

for task, (ttype, target, path) in TASKS.items():
    print(f"\n{'═' * 60}")
    print(f"  Task: {task}  |  type: {ttype}  |  target: {target}")
    print(f"{'═' * 60}")

    df_full = pd.read_pickle(path)

    # Apply task-specific filters
    if task == "egap":
        df_full = df_full[df_full["Egap"] > 0.1].reset_index(drop=True)
    if task == "supercon":
        df_full = df_full[df_full["Tc"] > 0].reset_index(drop=True)

    # Locate SHAP CSVs for this task
    shap_dir = SHAP_ROOT / task
    if task == "supercon":
        # Supercon SHAP files are named *_global.csv; fall back to all CSVs
        files = glob.glob(str(shap_dir / "*_global.csv"))
        if not files:
            files = glob.glob(str(shap_dir / "*.csv"))
    else:
        files = glob.glob(str(shap_dir / "*.csv"))

    if not files:
        print(f"  WARNING: no SHAP CSVs found in {shap_dir} — skipping.")
        continue

    for csv_path in sorted(files):
        model_name = Path(csv_path).stem
        print(f"\n  SHAP source: {model_name}")
        print(f"  {'k':>4}  {'primary metric':>16}  {'±std':>8}")
        print(f"  {'─' * 34}")

        for k in K_LIST:
            feats = load_top_features(csv_path, k)
            feats = [f for f in feats if f in df_full.columns]

            if len(feats) < 2:
                print(f"  {k:>4}  (fewer than 2 valid features — skip)")
                continue

            X = df_full[feats]
            y = df_full[target]

            # Binarize hm_class: 1 = half-metal (class 2), 0 = everything else
            if task == "hm_class":
                y = (y == 2).astype(int)

            if ttype == "reg":
                fold_metrics = cv_regression(X, y)
            elif ttype == "clf_bin":
                fold_metrics = cv_binary(X, y)
            else:
                raise ValueError(f"Unknown task type: {ttype}")

            metric_keys = list(fold_metrics[0].keys())
            mean_vals   = {mk: float(np.mean([fm[mk] for fm in fold_metrics]))
                           for mk in metric_keys}
            std_vals    = {mk: float(np.std( [fm[mk] for fm in fold_metrics]))
                           for mk in metric_keys}

            pm   = PRIMARY_METRIC[task]
            print(f"  {k:>4}  {mean_vals[pm]:>16.4f}  ±{std_vals[pm]:.4f}")

            records.append({
                "task":  task,
                "model": model_name,
                "k":     k,
                **{f"{mk}_mean": mean_vals[mk] for mk in metric_keys},
                **{f"{mk}_std":  std_vals[mk]  for mk in metric_keys},
            })

# ══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════

results_df = pd.DataFrame(records)
csv_out    = OUT_DIR / "results.csv"
results_df.to_csv(csv_out, index=False)
print(f"\n  Results saved → {csv_out}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS  — one per task (primary metric vs k, all models)
# ══════════════════════════════════════════════════════════════════════════════

METRIC_LABEL = {
    "mae":      "MAE",
    "rmse":     "RMSE",
    "r2":       "R²",
    "f1":       "Macro F1",
    "bal_acc":  "Balanced Accuracy",
}

# Whether lower is better for the primary metric (affects y-axis label)
LOWER_IS_BETTER = {"mae", "rmse"}

for task in results_df["task"].unique():
    dft    = results_df[results_df["task"] == task]
    pm     = PRIMARY_METRIC[task]
    models = dft["model"].unique()

    fig, ax = plt.subplots(figsize=(7, 4))

    for model in sorted(models):
        sub = dft[dft["model"] == model].sort_values("k")
        ax.errorbar(
            sub["k"],
            sub[f"{pm}_mean"],
            yerr=sub[f"{pm}_std"],
            marker="o", capsize=3, label=model,
        )

    ax.set_xlabel("k (number of top SHAP features)")
    ax.set_ylabel(METRIC_LABEL.get(pm, pm))
    direction = "↓ better" if pm in LOWER_IS_BETTER else "↑ better"
    ax.set_title(f"{task} — {METRIC_LABEL.get(pm, pm)} vs k  ({direction})",
                 fontsize=11)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(K_LIST)

    plt.tight_layout()
    fig_path = OUT_DIR / f"{task}_topk.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved → {fig_path}")

# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY — best k per task/model
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 70}")
print(f"  Best k per task (by primary metric)")
print(f"  {'Task':<14} {'Model':<28} {'Best k':>6}  {'Score':>10}")
print(f"  {'─' * 64}")

for task in results_df["task"].unique():
    pm  = PRIMARY_METRIC[task]
    dft = results_df[results_df["task"] == task]

    for model in sorted(dft["model"].unique()):
        sub = dft[dft["model"] == model]
        if pm in LOWER_IS_BETTER:
            best_row = sub.loc[sub[f"{pm}_mean"].idxmin()]
        else:
            best_row = sub.loc[sub[f"{pm}_mean"].idxmax()]

        print(f"  {task:<14} {model:<28} {int(best_row['k']):>6}"
              f"  {best_row[f'{pm}_mean']:>10.4f}")

print(f"{'═' * 70}")
print(f"\nDone → {OUT_DIR}/")