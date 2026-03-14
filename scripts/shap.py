#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
shap_analysis.py
────────────────
Unified SHAP analysis for all StoichML tasks.

For each task: SHAP beeswarm (summary) + bar (mean |SHAP|) plots.
Seed-ensemble tasks (egap_type, hm_class stage 1) average SHAP values
across all seed models for stability.

Tasks
  enthalpy    regression       single LGBM model
  egap        regression       single LGBM model
  egap_type   binary           mean SHAP across 5-seed LGBM ensemble
  hm_class    binary (stage 1) mean SHAP across 5-seed LGBM ensemble
                               stage 1 only — half-metal vs not

Outputs  shap_outputs/{task}/
  shap_summary_{task}.png     beeswarm coloured by feature value
  shap_bar_{task}.png         mean |SHAP| bar chart

Usage
  python shap_analysis.py --task enthalpy
  python shap_analysis.py --task all
  python shap_analysis.py --task all --max_samples 5000
"""

import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

RANDOM_STATE   = 42
DATA_PATH      = "data/data_feat.pkl"
FEATURES_JSON  = "data/selected_features.json"
MODEL_DIR      = "models"
OUT_DIR        = "shap_outputs"

MAX_SAMPLES    = 10000
TOP_K          = 20
DPI            = 150

UNDERSAMPLE_SEEDS = [0, 7, 21, 42, 99]

# Task registry
# type:       how to extract SHAP values
# model_path: callable(seed=None) -> path string
TASKS = {
    "enthalpy": {
        "type":         "regression",
        "features_key": "enthalpy",
        "model_paths":  lambda: [os.path.join(MODEL_DIR, "enthalpy", "enthalpy_lgbm.pkl")],
    },
    "egap": {
        "type":         "regression",
        "features_key": "egap",
        "model_paths":  lambda: [os.path.join(MODEL_DIR, "egap", "egap_lgbm.pkl")],
    },
    "egap_type": {
        "type":         "binary",
        "features_key": "egap_type",
        "model_paths":  lambda: [
            os.path.join(MODEL_DIR, "egap_type", f"egap_type_lgbm_seed{s}.pkl")
            for s in UNDERSAMPLE_SEEDS
        ],
    },
    "hm_class": {
        "type":         "binary",          # stage 1 only — half-metal vs not
        "features_key": "hm_class",
        "model_paths":  lambda: [
            os.path.join(MODEL_DIR, "hm_class", f"stage1_lgbm_seed{s}.pkl")
            for s in UNDERSAMPLE_SEEDS
        ],
        "shap_class":   1,                 # class 1 = half-metal
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def load_features(task: str) -> list[str]:
    with open(FEATURES_JSON) as f:
        feats = json.load(f)
    key = TASKS[task]["features_key"]
    task_feats = feats.get(key, [])
    if not task_feats:
        raise ValueError(f"No features found for '{key}' in {FEATURES_JSON}.")
    return task_feats


def extract_shap(raw_sv, task_type: str, shap_class: int = 1) -> np.ndarray:
    """
    Normalise SHAP output to (n_samples, n_features) regardless of
    how TreeExplainer returned it (list, 2D, 3D).
    """
    if task_type == "regression":
        return np.array(raw_sv)

    if task_type == "binary":
        if isinstance(raw_sv, list):
            return np.array(raw_sv[shap_class])
        if raw_sv.ndim == 3:
            # (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
            if raw_sv.shape[-1] == 2:
                return raw_sv[:, :, shap_class]
            return raw_sv[shap_class]
        return np.array(raw_sv)

    raise ValueError(f"Unknown task type: {task_type}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_beeswarm(shap_vals: np.ndarray,
                  X: pd.DataFrame,
                  title: str,
                  fname: str,
                  top_k: int = TOP_K) -> None:
    """SHAP beeswarm plot — shows direction and magnitude of each feature."""
    mean_abs   = np.abs(shap_vals).mean(axis=0)
    top_idx    = np.argsort(mean_abs)[-top_k:][::-1]
    top_feats  = X.columns[top_idx]
    X_top      = X[top_feats]
    sv_top     = shap_vals[:, top_idx]

    fig, ax = plt.subplots(figsize=(8, max(5, top_k * 0.32)))
    shap.summary_plot(sv_top, X_top, show=False, max_display=top_k, plot_size=None)
    plt.title(title, fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout()
    fig.savefig(fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fname}")


def plot_bar(shap_vals: np.ndarray,
             X: pd.DataFrame,
             title: str,
             fname: str,
             top_k: int = TOP_K) -> None:
    """Mean |SHAP| bar chart."""
    mean_abs   = np.abs(shap_vals).mean(axis=0)
    top_idx    = np.argsort(mean_abs)[-top_k:][::-1]
    top_feats  = X.columns[top_idx]
    X_top      = X[top_feats]
    sv_top     = shap_vals[:, top_idx]

    fig, ax = plt.subplots(figsize=(8, max(5, top_k * 0.32)))
    shap.summary_plot(sv_top, X_top, plot_type="bar", show=False,
                      max_display=top_k, plot_size=None)
    plt.title(title, fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout()
    fig.savefig(fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS ROUTINE
# ══════════════════════════════════════════════════════════════════════════════

def run_shap(task_name: str, max_samples: int = MAX_SAMPLES) -> None:
    cfg = TASKS[task_name]

    print(f"\n{'═' * 55}")
    print(f"  SHAP — {task_name}")
    print(f"{'═' * 55}")

    # Load data
    df    = pd.read_pickle(DATA_PATH)
    feats = load_features(task_name)

    # For egap, filter to insulators only (matches training distribution)
    if task_name == "egap":
        df = df[df["Egap"] > 0.1].reset_index(drop=True)

    X = df[feats]

    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"  Sampled {max_samples} / {len(df)} rows")
    else:
        print(f"  Using all {len(X)} rows")

    # Load models
    model_paths = cfg["model_paths"]()
    models = []
    for path in model_paths:
        if not os.path.exists(path):
            print(f"  WARNING: model not found — {path}")
            continue
        models.append(joblib.load(path))

    if not models:
        print(f"  No models found for {task_name} — skipping.")
        return

    print(f"  Loaded {len(models)} model(s)")

    # Compute SHAP values — average across models if seed ensemble
    task_type  = cfg["type"]
    shap_class = cfg.get("shap_class", 1)

    all_shap = []
    for i, model in enumerate(models, 1):
        print(f"  Computing SHAP  [{i}/{len(models)}] ...", end="\r")
        explainer = shap.TreeExplainer(model)
        raw_sv    = explainer.shap_values(X)
        sv        = extract_shap(raw_sv, task_type, shap_class)
        all_shap.append(sv)

    shap_vals = np.mean(all_shap, axis=0)   # (n_samples, n_features)
    print(f"  SHAP computed — shape {shap_vals.shape}          ")

    # Output directory
    out_path = os.path.join(OUT_DIR, task_name)
    os.makedirs(out_path, exist_ok=True)

    # Titles
    type_label = {
        "enthalpy":  "Formation Enthalpy Regression",
        "egap":      "Band Gap Regression (Egap > 0.1 eV)",
        "egap_type": "Metal vs Insulator Classification",
        "hm_class":  "Half-metal Detector (Stage 1)",
    }
    n_models_str = f"{len(models)} seed model avg" if len(models) > 1 else "single model"
    title_base   = f"{type_label[task_name]}  ({n_models_str})"

    # Beeswarm
    plot_beeswarm(
        shap_vals, X,
        title = f"SHAP Summary — {title_base}",
        fname = os.path.join(out_path, f"shap_summary_{task_name}.png"),
    )

    # Bar
    plot_bar(
        shap_vals, X,
        title = f"Mean |SHAP| — {title_base}",
        fname = os.path.join(out_path, f"shap_bar_{task_name}.png"),
    )

    print(f"  Done — {task_name}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified SHAP analysis for StoichML tasks.",
        epilog=(
            "Examples:\n"
            "  python shap_analysis.py --task enthalpy\n"
            "  python shap_analysis.py --task all\n"
            "  python shap_analysis.py --task all --max_samples 5000"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=list(TASKS.keys()) + ["all"],
        help="Task to analyse, or 'all' to run all tasks.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=MAX_SAMPLES,
        help=f"Max samples for SHAP computation (default: {MAX_SAMPLES}).",
    )
    args = parser.parse_args()

    tasks_to_run = list(TASKS.keys()) if args.task == "all" else [args.task]

    for task in tasks_to_run:
        run_shap(task, max_samples=args.max_samples)

    print(f"\n{'═' * 55}")
    print(f"  All outputs in: {OUT_DIR}/")
    print(f"{'═' * 55}")