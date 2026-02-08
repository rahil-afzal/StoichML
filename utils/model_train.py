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
    f1_score,
)

import lightgbm as lgb
import xgboost as xgb

# =====================
# Paths & constants
# =====================

DATA_PATH = "data/data_feat.pkl"
FEATURES_JSON = "data/selected_features.json"
OUT_DIR = "models"

RANDOM_STATE = 42
N_SPLITS = 5


# =====================
# TASK registry
# =====================

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
        "class_weight": "balanced",
    },
    "hm_class": {
        "target": "hm_class",
        "type": "multiclass",
        "num_class": 3,
        "class_weight": {0: 1, 1: 1, 2: 3},
    },
}


# =====================
# Helpers
# =====================

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


def regression_metrics(y, p):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
        "r2": float(r2_score(y, p)),
    }


# =====================
# Training
# =====================

def train(task_name):

    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}")

    cfg = TASKS[task_name]
    print(f"\n=== TASK: {task_name} ===")

    df = pd.read_pickle(DATA_PATH)
    feats = load_features(task_name)

    X = df[feats]
    y_raw = df[cfg["target"]]
    y = transform_target(y_raw, cfg.get("transform"))

    # CV splitter
    if cfg["type"] == "regression":
        cv = KFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
    else:
        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

    fold_metrics = {
        "lgbm": [],
        "xgb": [],
    }

    # =====================
    # Cross-validation
    # =====================

    for fold, (tr, va) in enumerate(cv.split(X, y)):
        print(f"Fold {fold+1}/{N_SPLITS}")

        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]

        # ----- LightGBM -----
        if cfg["type"] == "regression":
            lgbm = lgb.LGBMRegressor(
                learning_rate=0.05,
                num_leaves=64,
                min_data_in_leaf=50,
                feature_fraction=0.9,
                bagging_fraction=0.8,
                bagging_freq=1,
                n_estimators=500,
                random_state=RANDOM_STATE,
            )
        else:
            lgbm = lgb.LGBMClassifier(
                objective="binary" if cfg["type"] == "binary" else "multiclass",
                num_class=cfg.get("num_class"),
                learning_rate=0.05,
                num_leaves=64,
                min_data_in_leaf=50,
                n_estimators=500,
                feature_fraction=0.9,
                bagging_fraction=0.8,
                bagging_freq=1,
                random_state=RANDOM_STATE,
            )

        fit_kw = {}
        if "class_weight" in cfg:
            fit_kw["class_weight"] = cfg["class_weight"]

        lgbm.fit(Xtr, ytr, **fit_kw)

        # ----- XGBoost -----
        if cfg["type"] == "regression":
            xgbm = xgb.XGBRegressor(
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.9,
                n_estimators=2000,
                random_state=RANDOM_STATE,
                tree_method="hist",
            )
        else:
            xgbm = xgb.XGBClassifier(
                objective="binary:logistic" if cfg["type"] == "binary" else "multi:softprob",
                num_class=cfg.get("num_class"),
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.9,
                n_estimators=2000,
                random_state=RANDOM_STATE,
                tree_method="hist",
            )

        if cfg["type"] == "binary" and cfg.get("class_weight") == "balanced":
            pos = (ytr == 1).sum()
            neg = (ytr == 0).sum()
            xgbm.set_params(scale_pos_weight=neg / max(pos, 1))

        xgbm.fit(Xtr, ytr, verbose=False)

        # ----- Evaluation -----
        if cfg["type"] == "regression":
            yva_real = inverse_transform(yva, cfg.get("transform"))

            p_l = inverse_transform(lgbm.predict(Xva), cfg.get("transform"))
            p_x = inverse_transform(xgbm.predict(Xva), cfg.get("transform"))

            fold_metrics["lgbm"].append(regression_metrics(yva_real, p_l))
            fold_metrics["xgb"].append(regression_metrics(yva_real, p_x))

        elif cfg["type"] == "binary":
            p_l = lgbm.predict_proba(Xva)[:, 1]
            p_x = xgbm.predict_proba(Xva)[:, 1]

            fold_metrics["lgbm"].append(
                {"auc": float(roc_auc_score(yva, p_l))}
            )
            fold_metrics["xgb"].append(
                {"auc": float(roc_auc_score(yva, p_x))}
            )

        else:
            p_l = lgbm.predict(Xva)
            p_x = xgbm.predict(Xva)

            fold_metrics["lgbm"].append(
                {"macro_f1": float(f1_score(yva, p_l, average="macro"))}
            )
            fold_metrics["xgb"].append(
                {"macro_f1": float(f1_score(yva, p_x, average="macro"))}
            )

    # =====================
    # Aggregate CV metrics
    # =====================

    def average(metrics):
        return {
            k: float(np.mean([m[k] for m in metrics]))
            for k in metrics[0]
        }

    cv_mean = {
        "lgbm": average(fold_metrics["lgbm"]),
        "xgb": average(fold_metrics["xgb"]),
    }

    # =====================
    # Final full-data fit
    # =====================

    final_lgbm = lgbm.__class__(**lgbm.get_params())
    final_xgbm = xgbm.__class__(**xgbm.get_params())

    final_lgbm.fit(X, y, **fit_kw)
    final_xgbm.fit(X, y)

    full_metrics = {}

    if cfg["type"] == "regression":
        y_real = inverse_transform(y, cfg.get("transform"))
        p_l = inverse_transform(final_lgbm.predict(X), cfg.get("transform"))
        p_x = inverse_transform(final_xgbm.predict(X), cfg.get("transform"))

        full_metrics["lgbm"] = regression_metrics(y_real, p_l)
        full_metrics["xgb"] = regression_metrics(y_real, p_x)

    # =====================
    # Save everything
    # =====================

    os.makedirs(OUT_DIR, exist_ok=True)

    joblib.dump(final_lgbm, f"{OUT_DIR}/{task_name}_lgbm.pkl")
    joblib.dump(final_xgbm, f"{OUT_DIR}/{task_name}_xgb.pkl")

    results = {
        "cv_folds": fold_metrics,
        "cv_mean": cv_mean,
        "full_fit": full_metrics,
    }

    with open(f"{OUT_DIR}/{task_name}_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nSaved models and metrics.")
    print(json.dumps(cv_mean, indent=2))


# =====================
# CLI
# =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASKS.keys())
    args = parser.parse_args()

    train(args.task)
