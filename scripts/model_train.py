#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    precision_recall_fscore_support
)

import lightgbm as lgb
import xgboost as xgb


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5

DATA_PATH = "data/data_feat.pkl"
FEATURES_JSON = "data/selected_features.json"
OUT_DIR = "models"

TASKS = {
    "enthalpy": {
        "target": "enthalpy_formation_atom",
        "type": "regression",
        "transform": None,
    },
    "egap": {
        "target": "Egap",
        "type": "regression",
        "transform": "log1p",
    },
    "egap_type": {
        "target": "Egap_type_numeric",
        "type": "binary",
    },
    "hm_class": {
        "target": "hm_class",
        "type": "multiclass",
        "num_class": 3,
        "class_weight": {0: 1, 1: 1, 2: 8}
    },
}


# ============================================================
# UTILITIES
# ============================================================

def load_features(task):
    with open(FEATURES_JSON) as f:
        feats = json.load(f)
    core = feats.get("core", [])
    task_feats = feats.get(task, [])
    return list(dict.fromkeys(core + task_feats))


def transform_target(y, mode):
    if mode == "log1p":
        return np.log1p(y)
    return y


def inverse_transform(y, mode):
    if mode == "log1p":
        return np.expm1(y)
    return y


def regression_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def summarize(metric_list):
    keys = metric_list[0].keys()
    return {
        k: {
            "mean": float(np.mean([m[k] for m in metric_list])),
            "std": float(np.std([m[k] for m in metric_list]))
        }
        for k in keys
    }


# ============================================================
# TRAINING
# ============================================================

def train(task_name):

    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}")

    cfg = TASKS[task_name]

    df = pd.read_pickle(DATA_PATH)
    feats = load_features(task_name)

    X = df[feats]
    y_raw = df[cfg["target"]]
    y = transform_target(y_raw, cfg.get("transform"))

    if cfg["type"] == "regression":
        cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    else:
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    metrics_lgb = []
    metrics_xgb = []

    oof_pred_lgb = np.zeros(len(y))
    oof_pred_xgb = np.zeros(len(y))

    # ============================================================
    # CROSS VALIDATION
    # ============================================================

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):

        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

        # --------------------------------------------------------
        # REGRESSION
        # --------------------------------------------------------
        if cfg["type"] == "regression":

            lgbm = lgb.LGBMRegressor(
                learning_rate=0.05,
                num_leaves=64,
                n_estimators=5000,
                random_state=RANDOM_STATE
            )

            xgbm = xgb.XGBRegressor(
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.9,
                n_estimators=5000,
                tree_method="hist",
                random_state=RANDOM_STATE
            )

            lgbm.fit(Xtr, ytr,
                     eval_set=[(Xva, yva)],
                     eval_metric="l2",
                     early_stopping_rounds=200,
                     verbose=False)

            xgbm.fit(Xtr, ytr,
                     eval_set=[(Xva, yva)],
                     eval_metric="rmse",
                     early_stopping_rounds=200,
                     verbose=False)

            p_l = inverse_transform(lgbm.predict(Xva), cfg.get("transform"))
            p_x = inverse_transform(xgbm.predict(Xva), cfg.get("transform"))
            y_real = inverse_transform(yva, cfg.get("transform"))

            oof_pred_lgb[va_idx] = p_l
            oof_pred_xgb[va_idx] = p_x

            metrics_lgb.append(regression_metrics(y_real, p_l))
            metrics_xgb.append(regression_metrics(y_real, p_x))

        # --------------------------------------------------------
        # BINARY CLASSIFICATION (BALANCED)
        # --------------------------------------------------------
        elif cfg["type"] == "binary":

            pos = (ytr == 1).sum()
            neg = (ytr == 0).sum()
            scale_pos_weight = neg / max(pos, 1)

            lgbm = lgb.LGBMClassifier(
                objective="binary",
                class_weight="balanced",
                learning_rate=0.05,
                num_leaves=64,
                n_estimators=5000,
                random_state=RANDOM_STATE
            )

            xgbm = xgb.XGBClassifier(
                objective="binary:logistic",
                scale_pos_weight=scale_pos_weight,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.9,
                n_estimators=5000,
                tree_method="hist",
                random_state=RANDOM_STATE
            )

            lgbm.fit(Xtr, ytr,
                     eval_set=[(Xva, yva)],
                     eval_metric="auc",
                     early_stopping_rounds=200,
                     verbose=False)

            xgbm.fit(Xtr, ytr,
                     eval_set=[(Xva, yva)],
                     eval_metric="auc",
                     early_stopping_rounds=200,
                     verbose=False)

            prob_l = lgbm.predict_proba(Xva)[:, 1]
            prob_x = xgbm.predict_proba(Xva)[:, 1]

            oof_pred_lgb[va_idx] = prob_l
            oof_pred_xgb[va_idx] = prob_x

            def best_threshold(y_true, prob):
                fpr, tpr, thr = roc_curve(y_true, prob)
                return thr[np.argmax(tpr - fpr)]

            thr_l = best_threshold(yva, prob_l)
            thr_x = best_threshold(yva, prob_x)

            pred_l = (prob_l >= thr_l).astype(int)
            pred_x = (prob_x >= thr_x).astype(int)

            metrics_lgb.append({
                "roc_auc": float(roc_auc_score(yva, prob_l)),
                "pr_auc": float(average_precision_score(yva, prob_l)),
                "macro_f1": float(f1_score(yva, pred_l, average="macro")),
                "balanced_acc": float(balanced_accuracy_score(yva, pred_l)),
            })

            metrics_xgb.append({
                "roc_auc": float(roc_auc_score(yva, prob_x)),
                "pr_auc": float(average_precision_score(yva, prob_x)),
                "macro_f1": float(f1_score(yva, pred_x, average="macro")),
                "balanced_acc": float(balanced_accuracy_score(yva, pred_x)),
            })

        # --------------------------------------------------------
        # MULTICLASS
        # --------------------------------------------------------
        else:

            weights = ytr.map(cfg["class_weight"]).values

            lgbm = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=cfg["num_class"],
                class_weight=cfg["class_weight"],
                learning_rate=0.05,
                num_leaves=64,
                n_estimators=5000,
                random_state=RANDOM_STATE
            )

            xgbm = xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=cfg["num_class"],
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.9,
                n_estimators=5000,
                tree_method="hist",
                random_state=RANDOM_STATE
            )

            lgbm.fit(Xtr, ytr,
                     eval_set=[(Xva, yva)],
                     eval_metric="multi_logloss",
                     early_stopping_rounds=200,
                     verbose=False)

            xgbm.fit(Xtr, ytr,
                     sample_weight=weights,
                     eval_set=[(Xva, yva)],
                     eval_metric="mlogloss",
                     early_stopping_rounds=200,
                     verbose=False)

            pred_l = lgbm.predict(Xva)
            pred_x = xgbm.predict(Xva)

            macro_l = f1_score(yva, pred_l, average="macro")
            macro_x = f1_score(yva, pred_x, average="macro")

            metrics_lgb.append({"macro_f1": float(macro_l)})
            metrics_xgb.append({"macro_f1": float(macro_x)})

    # ============================================================
    # AGGREGATE RESULTS
    # ============================================================

    results = {
        "lgbm": summarize(metrics_lgb),
        "xgb": summarize(metrics_xgb)
    }

    # ============================================================
    # FINAL FULL-DATA FIT
    # ============================================================

    final_lgb = lgbm.__class__(**lgbm.get_params())
    final_xgb = xgbm.__class__(**xgbm.get_params())

    final_lgb.fit(X, y)
    final_xgb.fit(X, y)

    os.makedirs(OUT_DIR, exist_ok=True)

    joblib.dump(final_lgb, f"{OUT_DIR}/{task_name}_lgbm.pkl")
    joblib.dump(final_xgb, f"{OUT_DIR}/{task_name}_xgb.pkl")

    with open(f"{OUT_DIR}/{task_name}_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print(json.dumps(results, indent=2))


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASKS.keys())
    args = parser.parse_args()

    train(args.task)
