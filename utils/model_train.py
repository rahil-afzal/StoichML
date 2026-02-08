import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
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


def regression_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def hm_metrics(y_true, y_pred, labels):
    macro = f1_score(y_true, y_pred, average="macro")
    weighted = f1_score(y_true, y_pred, average="weighted")

    p, r, f, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "macro_f1": float(macro),
        "weighted_f1": float(weighted),
        "per_class_f1": {str(lbl): float(f[i]) for i, lbl in enumerate(labels)},
        "confusion_matrix": cm.tolist(),
    }

# =====================
# Training
# =====================

def train(task_name):
    cfg = TASKS[task_name]
    print(f"\n=== TASK: {task_name} ===")

    df = pd.read_pickle(DATA_PATH)
    feats = load_features(task_name)

    X = df[feats]
    y_raw = df[cfg["target"]]
    y = transform_target(y_raw, cfg.get("transform"))

    if cfg["type"] == "regression":
        cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    else:
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_metrics = {"lgbm": [], "xgb": []}

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        print(f"Fold {fold + 1}/{N_SPLITS}")

        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

        # ---------- LightGBM ----------
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
                class_weight=cfg.get("class_weight"),
                learning_rate=0.05,
                num_leaves=64,
                min_data_in_leaf=50,
                feature_fraction=0.9,
                bagging_fraction=0.8,
                bagging_freq=1,
                n_estimators=500,
                random_state=RANDOM_STATE,
            )

        lgbm.fit(Xtr, ytr)

        # ---------- XGBoost ----------
        sample_weight = None

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

            elif cfg["type"] == "multiclass":
                sample_weight = ytr.map(cfg["class_weight"]).values

        xgbm.fit(Xtr, ytr, sample_weight=sample_weight, verbose=False)

        # ---------- Evaluation ----------
        if cfg["type"] == "regression":
            yva_real = inverse_transform(yva, cfg.get("transform"))
            fold_metrics["lgbm"].append(
                regression_metrics(yva_real, inverse_transform(lgbm.predict(Xva), cfg.get("transform")))
            )
            fold_metrics["xgb"].append(
                regression_metrics(yva_real, inverse_transform(xgbm.predict(Xva), cfg.get("transform")))
            )

        elif cfg["type"] == "binary":
            fold_metrics["lgbm"].append(
                {"auc": float(roc_auc_score(yva, lgbm.predict_proba(Xva)[:, 1]))}
            )
            fold_metrics["xgb"].append(
                {"auc": float(roc_auc_score(yva, xgbm.predict_proba(Xva)[:, 1]))}
            )

        else:  # hm_class
            labels = list(range(cfg["num_class"]))
            fold_metrics["lgbm"].append(
                hm_metrics(yva, lgbm.predict(Xva), labels)
            )
            fold_metrics["xgb"].append(
                hm_metrics(yva, xgbm.predict(Xva), labels)
            )

    # ---------- Aggregate ----------
    def mean_metric(ms, key):
        return float(np.mean([m[key] for m in ms]))

    if cfg["type"] == "multiclass":
        cv_mean = {
            "lgbm_macro_f1": mean_metric(fold_metrics["lgbm"], "macro_f1"),
            "xgb_macro_f1": mean_metric(fold_metrics["xgb"], "macro_f1"),
        }
    else:
        cv_mean = {
            "lgbm": {k: mean_metric(fold_metrics["lgbm"], k) for k in fold_metrics["lgbm"][0]},
            "xgb": {k: mean_metric(fold_metrics["xgb"], k) for k in fold_metrics["xgb"][0]},
        }

    # ---------- Save ----------
    os.makedirs(OUT_DIR, exist_ok=True)

    joblib.dump(lgbm, f"{OUT_DIR}/{task_name}_lgbm.pkl")
    joblib.dump(xgbm, f"{OUT_DIR}/{task_name}_xgb.pkl")

    with open(f"{OUT_DIR}/{task_name}_metrics.json", "w") as f:
        json.dump(
            {
                "cv_folds": fold_metrics,
                "cv_mean": cv_mean,
            },
            f,
            indent=4,
        )

    print(json.dumps(cv_mean, indent=2))


# =====================
# CLI
# =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASKS.keys())
    args = parser.parse_args()
    train(args.task)
