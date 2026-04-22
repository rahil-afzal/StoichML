#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py
────────
StoichML training pipeline — enthalpy and egap_type.

Tasks:
  enthalpy   formation enthalpy regression (eV/atom)
             LightGBM + XGBoost, 5-fold CV, full-dataset final fit

  egap_type  metal vs insulator classification
             5-seed ensemble undersampling (strict 1:1 balance)
             Seed-outer / fold-inner CV — one LightGBM model per seed

Usage:
    python train.py                      # run both tasks
    python train.py --task enthalpy
    python train.py --task egap_type
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

UNDERSAMPLE_SEEDS = [0, 7, 21, 42, 99]

TASKS = {
    "enthalpy": {
        "type":      "regression",
        "target":    "enthalpy_formation_atom",
        "transform": None,
        "data_path": "data/data_feat.pkl",
    },
    "egap_type": {
        "type":           "binary_ensemble",
        "target":         "Egap_type_numeric",
        "minority_class": 1,
        "data_path":      "data/data_feat.pkl",
    },
}
NON_FEATURE_COLS = [
    "compound", "compounds", "spacegroup_relax", "Egap", "Egap_type",
    "Egap_type_numeric", "enthalpy_formation_atom",
    "composition", "elements", "ratios", "hm_class", "Tc",
]


# ══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def load_features(task: str) -> list[str]:
    with open(FEATURES_JSON) as f:
        feats = json.load(f)
    task_feats = feats.get(task, [])
    if not task_feats:
        raise ValueError(
            f"No features found for task '{task}' in {FEATURES_JSON}. "
            "Run feature_selection.py first."
        )
    return task_feats


def transform_target(y: np.ndarray, mode: str | None) -> np.ndarray:
    if mode == "log1p":
        return np.log1p(np.clip(y, a_min=0, a_max=None))
    return y


def inverse_transform(y: np.ndarray, mode: str | None) -> np.ndarray:
    if mode == "log1p":
        return np.expm1(y)
    return y


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


def oof_best_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
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
    Must be identical to feature_selection.py.
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

def build_lgbm_regression(n_estimators: int = 5000) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        learning_rate=0.05,
        num_leaves=64,
        n_estimators=n_estimators,
        min_data_in_leaf=30,
        subsample=0.8,
        colsample_bytree=0.8,
        importance_type="gain",
        verbose=-1,
        random_state=RANDOM_STATE,
    )


def build_xgb_regression(
    n_estimators: int = 5000,
    early_stopping: bool = True,
) -> xgb.XGBRegressor:
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


def build_lgbm_binary(n_estimators: int = 5000) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.05,
        num_leaves=64,
        n_estimators=n_estimators,
        min_data_in_leaf=20,
        subsample=0.8,
        colsample_bytree=0.8,
        importance_type="gain",
        random_state=RANDOM_STATE,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TASK TRAINERS
# ══════════════════════════════════════════════════════════════════════════════

def train_regression(task_name: str, cfg: dict) -> None:
    df = pd.read_pickle(cfg["data_path"])
    print(f"\n  {len(df)} total samples")

    feats = load_features(task_name)
    X_df  = df[feats]
    y     = transform_target(df[cfg["target"]].values, cfg.get("transform"))

    print(f"  Features: {len(feats)}")
    print(f"  y range: [{y.min():.4f}, {y.max():.4f}]  mean: {y.mean():.4f}")

    cv    = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(cv.split(X_df))

    metrics_lgb    = []
    metrics_xgb    = []
    oof_lgb        = np.zeros(len(y))
    oof_xgb        = np.zeros(len(y))
    best_iters_lgb = []
    best_iters_xgb = []

    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        print(f"\n  ── Fold {fold}/{N_SPLITS} {'─' * 38}")

        Xtr_df, Xva_df = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        ytr,    yva    = y[tr_idx],          y[va_idx]

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
        xgbm.fit(Xtr_df, ytr, eval_set=[(Xva_df, yva)], verbose=False)

        best_iters_lgb.append(lgbm.best_iteration_)
        best_iters_xgb.append(xgbm.best_iteration)

        oof_lgb[va_idx] = lgbm.predict(Xva_df)
        oof_xgb[va_idx] = xgbm.predict(Xva_df)

        p_lgb = inverse_transform(oof_lgb[va_idx], cfg.get("transform"))
        p_xgb = inverse_transform(oof_xgb[va_idx], cfg.get("transform"))
        y_val = inverse_transform(yva,              cfg.get("transform"))

        m_lgb = regression_metrics(y_val, p_lgb)
        m_xgb = regression_metrics(y_val, p_xgb)
        metrics_lgb.append(m_lgb)
        metrics_xgb.append(m_xgb)

        print(f"  LGBM  iter={lgbm.best_iteration_:4d}  "
              f"MAE={m_lgb['mae']:.4f}  RMSE={m_lgb['rmse']:.4f}  R²={m_lgb['r2']:.4f}")
        print(f"  XGB   iter={xgbm.best_iteration:4d}  "
              f"MAE={m_xgb['mae']:.4f}  RMSE={m_xgb['rmse']:.4f}  R²={m_xgb['r2']:.4f}")

    y_orig    = inverse_transform(y,       cfg.get("transform"))
    oof_m_lgb = regression_metrics(y_orig, inverse_transform(oof_lgb, cfg.get("transform")))
    oof_m_xgb = regression_metrics(y_orig, inverse_transform(oof_xgb, cfg.get("transform")))
    oof_m_ens = regression_metrics(
        y_orig,
        inverse_transform((oof_lgb + oof_xgb) / 2, cfg.get("transform")),
    )

    print(f"\n  OOF (original units):")
    print(f"  LGBM      MAE={oof_m_lgb['mae']:.4f}  RMSE={oof_m_lgb['rmse']:.4f}  R²={oof_m_lgb['r2']:.4f}")
    print(f"  XGB       MAE={oof_m_xgb['mae']:.4f}  RMSE={oof_m_xgb['rmse']:.4f}  R²={oof_m_xgb['r2']:.4f}")
    print(f"  Ensemble  MAE={oof_m_ens['mae']:.4f}  RMSE={oof_m_ens['rmse']:.4f}  R²={oof_m_ens['r2']:.4f}")

    mean_iter_lgb = max(1, int(np.mean(best_iters_lgb)))
    mean_iter_xgb = max(1, int(np.mean(best_iters_xgb)))
    print(f"\n  Final fit — LGBM iter={mean_iter_lgb}  XGB iter={mean_iter_xgb}")

    final_lgb = build_lgbm_regression(n_estimators=mean_iter_lgb)
    final_xgb = build_xgb_regression(n_estimators=mean_iter_xgb, early_stopping=False)
    final_lgb.fit(X_df, y)
    final_xgb.fit(X_df, y)

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
    }
    save_json(results, os.path.join(task_dir, f"{task_name}_metrics.json"))

    print(f"\n  Saved → {task_dir}/")
    print(f"\n{'═' * 55}")
    print(f"  CV Summary  (mean ± std across {N_SPLITS} folds)")
    print(f"{'═' * 55}")
    for model_name, cv_m in results["cv_folds"].items():
        mae  = cv_m["mae"]
        rmse = cv_m["rmse"]
        r2   = cv_m["r2"]
        print(f"  {model_name.upper():<6}  "
              f"MAE={mae['mean']:.4f}±{mae['std']:.4f}  "
              f"RMSE={rmse['mean']:.4f}±{rmse['std']:.4f}  "
              f"R²={r2['mean']:.4f}±{r2['std']:.4f}")


def train_egap_type(task_name: str, cfg: dict) -> None:
    df    = pd.read_pickle(DATA_PATH)
    feats = load_features(task_name)

    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns],
                errors="ignore")[feats]
    y = df[cfg["target"]]
    mask = y.isin([0, 1])
    X    = X.loc[mask].reset_index(drop=True)
    y    = y.loc[mask].reset_index(drop=True)
    minority_class = cfg["minority_class"]
    print(f"\n  {len(y)} samples")
    print(f"  Class 0 (Conductor): {(y == 0).sum()}")
    print(f"  Class 1 (Insulator): {(y == 1).sum()}")
    print(f"  Ratio: {(y == 0).sum() / (y == 1).sum():.1f}:1")
    print(f"  Features: {len(feats)}")
    print(f"\n  Strategy: {N_SPLITS}-fold CV × {len(UNDERSAMPLE_SEEDS)} seeds "
          f"= {N_SPLITS * len(UNDERSAMPLE_SEEDS)} total models")

    cv    = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(cv.split(X, y))

    oof_probs_by_seed: dict[int, np.ndarray] = {}
    seed_cv_metrics:   dict[int, list[dict]] = {}
    best_iters:        dict[int, list[int]]  = {}
    final_models:      dict[int, lgb.LGBMClassifier] = {}
    y_oof = np.zeros(len(y))

    for seed in UNDERSAMPLE_SEEDS:
        print(f"\n{'═' * 55}")
        print(f"  Seed {seed}")
        print(f"{'═' * 55}")

        oof_prob     = np.zeros(len(y))
        fold_metrics = []
        iters        = []

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

    ensemble_prob = np.mean([oof_probs_by_seed[s] for s in UNDERSAMPLE_SEEDS], axis=0)
    thr_ensemble  = oof_best_threshold(y_oof, ensemble_prob)
    ens_oof_m     = binary_metrics(y_oof, ensemble_prob, thr=thr_ensemble)

    print(f"\n{'═' * 55}")
    print(f"  Ensemble OOF (mean of {len(UNDERSAMPLE_SEEDS)} seeds, "
          f"thr={thr_ensemble:.3f})")
    print(f"{'═' * 55}")
    for k, v in ens_oof_m.items():
        print(f"    {k}: {v:.4f}")

    print(f"\n  Final fit — full dataset, one model per seed")
    for seed in UNDERSAMPLE_SEEDS:
        mean_iter = max(1, int(np.mean(best_iters[seed])))
        print(f"  seed {seed:3d} | mean_iter={mean_iter}")

        X_bal, y_bal = undersample_strict(X, y, minority_class, seed)
        model = build_lgbm_binary(n_estimators=mean_iter)
        model.fit(X_bal, y_bal)
        final_models[seed] = model

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


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def predict_regression(
    X_new: pd.DataFrame,
    task_name: str,
    model_dir: str = OUT_DIR,
) -> np.ndarray:
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
    task_dir = os.path.join(model_dir, "egap_type")
    models   = [
        joblib.load(os.path.join(task_dir, f"egap_type_lgbm_seed{s}.pkl"))
        for s in UNDERSAMPLE_SEEDS
    ]
    thr_path = os.path.join(task_dir, "egap_type_threshold.json")
    thr      = json.load(open(thr_path))["ensemble_threshold"] if os.path.exists(thr_path) else 0.5
    prob     = np.mean([m.predict_proba(X_new)[:, 1] for m in models], axis=0)
    return (prob >= thr).astype(int), prob


# ══════════════════════════════════════════════════════════════════════════════
# DISPATCHER + CLI
# ══════════════════════════════════════════════════════════════════════════════

def train(task_name: str) -> None:
    cfg = TASKS[task_name]
    print(f"\n{'═' * 55}")
    print(f"  Task: {task_name}  |  type: {cfg['type']}")
    print(f"{'═' * 55}")

    if cfg["type"] == "regression":
        train_regression(task_name, cfg)
    elif cfg["type"] == "binary_ensemble":
        train_egap_type(task_name, cfg)
    else:
        raise ValueError(f"Unknown task type '{cfg['type']}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="StoichML training pipeline — enthalpy and egap_type.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python train.py                   # run both tasks\n"
            "  python train.py --task enthalpy\n"
            "  python train.py --task egap_type"
        ),
    )
    parser.add_argument(
        "--task",
        choices=list(TASKS.keys()),
        default=None,
        help="Task to train. If omitted, both tasks are run.",
    )
    args = parser.parse_args()

    tasks_to_run = [args.task] if args.task else list(TASKS.keys())
    for task_name in tasks_to_run:
        train(task_name)

    print(f"\n{'═' * 55}")
    print(f"  Done — trained: {', '.join(tasks_to_run)}")
    print(f"{'═' * 55}")