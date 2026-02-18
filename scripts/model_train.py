#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_curve,
    auc,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt

# =====================
# Config & Constants
# =====================
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
N_SPLITS = 5
DPI = 400
TOP_K = 8  # for SHAP/top features if needed

DATA_PATH = "data/data_feat.pkl"
FEATURES_JSON = "data/selected_features.json"
OUT_DIR = "models"

TASKS = {
    "enthalpy": {"target": "enthalpy_formation_atom", "type": "regression", "transform": None},
    "egap": {"target": "Egap", "type": "regression", "transform": "log1p"},
    "egap_type": {"target": "Egap_type_numeric", "type": "binary"},
    "hm_class": {"target": "hm_class", "type": "multiclass", "num_class": 3, "class_weight": {0:1,1:1,2:8}}
}

# =====================
# Helper Functions
# =====================

def load_features(task):
    with open(FEATURES_JSON) as f:
        feats = json.load(f)
    core = feats.get("core", [])
    task_feats = feats.get(task, [])
    return list(dict.fromkeys(core + task_feats))


def transform_target(y, mode):
    return np.log1p(y) if mode == "log1p" else y

def inverse_transform(y, mode):
    return np.expm1(y) if mode == "log1p" else y

def regression_metrics(y_true, y_pred):
    return {"rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred))}

def hm_metrics(y_true, y_pred, labels):
    macro = f1_score(y_true, y_pred, average="macro")
    weighted = f1_score(y_true, y_pred, average="weighted")
    _, _, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {"macro_f1": float(macro), "weighted_f1": float(weighted),
            "per_class_f1": {str(lbl): float(f[i]) for i,lbl in enumerate(labels)},
            "confusion_matrix": cm.tolist()}

# =====================
# Visualization Helpers
# =====================

def ensure_img_dir(task_dir):
    img_dir = os.path.join(task_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    return img_dir

def plot_regression_cv(y_true, y_pred, metrics_mean, model_name, task_name, img_dir):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], linestyle="--", color="black")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{task_name} - {model_name}")
    text = "\n".join([f"RMSE={metrics_mean['rmse']:.4f}", f"MAE={metrics_mean['mae']:.4f}", f"R2={metrics_mean['r2']:.4f}"])
    plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, verticalalignment="top")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, f"{task_name}_{model_name}_cv_scatter.png"), dpi=DPI)
    plt.close()

def plot_binary_roc(y_true, y_prob, model_name, task_name, img_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.plot([0,1],[0,1], linestyle="--", color="black")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{task_name} - {model_name} ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, f"{task_name}_{model_name}_roc.png"), dpi=DPI)
    plt.close()

def plot_confusion_matrix(cm, model_name, task_name, img_dir):
    plt.figure(figsize=(6,5))
    plt.imshow(cm)
    plt.title(f"{task_name} - {model_name} Confusion Matrix")
    plt.colorbar()
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, cm[i][j], ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, f"{task_name}_{model_name}_confusion.png"), dpi=DPI)
    plt.close()

def plot_macro_f1_bar(fold_metrics, model_name, task_name, img_dir):
    f1_scores = [m["macro_f1"] for m in fold_metrics]
    plt.figure(figsize=(6,4))
    plt.bar(range(1,len(f1_scores)+1), f1_scores)
    plt.xlabel("Fold")
    plt.ylabel("Macro F1")
    plt.title(f"{task_name} - {model_name} Macro F1 per Fold")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, f"{task_name}_{model_name}_macro_f1.png"), dpi=DPI)
    plt.close()

# =====================
# Training Routine
# =====================

def train(task_name):
    cfg = TASKS[task_name]
    print(f"\n=== TASK: {task_name} ===")

    df = pd.read_pickle(DATA_PATH)
    feats = load_features(task_name)
    X = df[feats]
    y_raw = df[cfg["target"]]
    y = transform_target(y_raw, cfg.get("transform"))

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE) \
        if cfg["type"] == "regression" else StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_metrics = {"lgbm": [], "xgb": []}
    all_preds = {"lgbm": [], "xgb": []}
    all_probs = {"lgbm": [], "xgb": []} if cfg["type"]=="binary" else None
    all_true = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        print(f"Fold {fold+1}/{N_SPLITS}")
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

        # --- LightGBM ---
        if cfg["type"]=="regression":
            lgbm = lgb.LGBMRegressor(learning_rate=0.05, num_leaves=64, min_data_in_leaf=50,
                                     n_estimators=500, random_state=RANDOM_STATE)
        else:
            lgbm = lgb.LGBMClassifier(objective="binary" if cfg["type"]=="binary" else "multiclass",
                                      num_class=cfg.get("num_class"), class_weight=cfg.get("class_weight"),
                                      learning_rate=0.05, num_leaves=64, min_data_in_leaf=50, n_estimators=500,
                                      random_state=RANDOM_STATE)
        lgbm.fit(Xtr, ytr)

        # --- XGBoost ---
        sample_weight = None
        if cfg["type"]=="regression":
            xgbm = xgb.XGBRegressor(learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.9,
                                    n_estimators=2000, random_state=RANDOM_STATE, tree_method="hist")
        else:
            xgbm = xgb.XGBClassifier(objective="binary:logistic" if cfg["type"]=="binary" else "multi:softprob",
                                     num_class=cfg.get("num_class"), learning_rate=0.05, max_depth=6,
                                     subsample=0.8, colsample_bytree=0.9, n_estimators=2000,
                                     random_state=RANDOM_STATE, tree_method="hist")
            if cfg["type"]=="binary" and cfg.get("class_weight")=="balanced":
                pos = (ytr==1).sum()
                neg = (ytr==0).sum()
                xgbm.set_params(scale_pos_weight=neg/max(pos,1))
            elif cfg["type"]=="multiclass":
                sample_weight = ytr.map(cfg["class_weight"]).values

        xgbm.fit(Xtr, ytr, sample_weight=sample_weight, verbose=False)

        # --- Evaluation ---
        if cfg["type"]=="regression":
            yva_real = inverse_transform(yva, cfg.get("transform"))
            lgb_pred = inverse_transform(lgbm.predict(Xva), cfg.get("transform"))
            xgb_pred = inverse_transform(xgbm.predict(Xva), cfg.get("transform"))
            fold_metrics["lgbm"].append(regression_metrics(yva_real, lgb_pred))
            fold_metrics["xgb"].append(regression_metrics(yva_real, xgb_pred))
            all_true.extend(yva_real)
            all_preds["lgbm"].extend(lgb_pred)
            all_preds["xgb"].extend(xgb_pred)

        elif cfg["type"]=="binary":
            lgb_prob = lgbm.predict_proba(Xva)[:,1]
            xgb_prob = xgbm.predict_proba(Xva)[:,1]
            fold_metrics["lgbm"].append({"auc": float(auc(*roc_curve(yva,lgb_prob)[:2]))})
            fold_metrics["xgb"].append({"auc": float(auc(*roc_curve(yva,xgb_prob)[:2]))})
            all_true.extend(yva)
            all_probs["lgbm"].extend(lgb_prob)
            all_probs["xgb"].extend(xgb_prob)

        else:
            labels = list(range(cfg["num_class"]))
            lgb_pred = lgbm.predict(Xva)
            xgb_pred = xgbm.predict(Xva)
            fold_metrics["lgbm"].append(hm_metrics(yva, lgb_pred, labels))
            fold_metrics["xgb"].append(hm_metrics(yva, xgb_pred, labels))
            all_true.extend(yva)
            all_preds["lgbm"].extend(lgb_pred)
            all_preds["xgb"].extend(xgb_pred)

    # --- Aggregate Metrics ---
    def mean_metric(ms, key):
        return float(np.mean([m[key] for m in ms]))

    if cfg["type"]=="multiclass":
        cv_mean = {
            "lgbm_macro_f1": mean_metric(fold_metrics["lgbm"], "macro_f1"),
            "xgb_macro_f1": mean_metric(fold_metrics["xgb"], "macro_f1")
        }
    else:
        cv_mean = {
            "lgbm": {k: mean_metric(fold_metrics["lgbm"], k) for k in fold_metrics["lgbm"][0]},
            "xgb": {k: mean_metric(fold_metrics["xgb"], k) for k in fold_metrics["xgb"][0]}
        }

    # --- Save Models & Metrics ---
    task_dir = os.path.join(OUT_DIR, task_name)
    os.makedirs(task_dir, exist_ok=True)
    joblib.dump(lgbm, os.path.join(task_dir, f"{task_name}_lgbm.pkl"))
    joblib.dump(xgbm, os.path.join(task_dir, f"{task_name}_xgb.pkl"))
    with open(os.path.join(task_dir, f"{task_name}_metrics.json"), "w") as f:
        json.dump({"cv_folds": fold_metrics, "cv_mean": cv_mean}, f, indent=4)

    # --- Visualizations ---
    img_dir = ensure_img_dir(task_dir)
    if cfg["type"]=="regression":
        plot_regression_cv(all_true, all_preds["lgbm"], cv_mean["lgbm"], "lgbm", task_name, img_dir)
        plot_regression_cv(all_true, all_preds["xgb"], cv_mean["xgb"], "xgb", task_name, img_dir)
    elif cfg["type"]=="binary":
        plot_binary_roc(all_true, all_probs["lgbm"], "lgbm", task_name, img_dir)
        plot_binary_roc(all_true, all_probs["xgb"], "xgb", task_name, img_dir)
    else:
        plot_confusion_matrix(confusion_matrix(all_true, all_preds["lgbm"]), "lgbm", task_name, img_dir)
        plot_confusion_matrix(confusion_matrix(all_true, all_preds["xgb"]), "xgb", task_name, img_dir)
        plot_macro_f1_bar(fold_metrics["lgbm"], "lgbm", task_name, img_dir)
        plot_macro_f1_bar(fold_metrics["xgb"], "xgb", task_name, img_dir)

    print(json.dumps(cv_mean, indent=2))

# =====================
# CLI Entry
# =====================
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(TASKS.keys())+["all"])
    args = parser.parse_args()
    if args.task=="all":
        for task in TASKS: train(task)
    else:
        train(args.task)
