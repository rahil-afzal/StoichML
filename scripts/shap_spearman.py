#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
shap_spearman.py
─────────────────

  1. Recomputing mean |SHAP| per feature with TreeExplainer
     (sanity check against your existing shap_outputs/{task}/
     feature_importance_{task} files).
  2. Computing the full pairwise Spearman correlation matrix among
     each task's *selected* feature set (not just the top 5 — this
     lets you see if a top-5 feature is secretly redundant with a
     lower-ranked feature too, which the top-5-only view would miss).
  3. Grouping the top-5 SHAP features into "correlated groups" via
     union-find on edges where |Spearman r| > CORR_THRESHOLD.


Usage
─────
    python shap_spearman.py --task enthalpy
    python shap_spearman.py --task all
    python shap_spearman.py --task all --threshold 0.7

"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG 
# ══════════════════════════════════════════════════════════════════════════════

RANDOM_STATE  = 42
N_SHAP_SAMPLE = 10000          # matches "up to 10,000 compounds" in the paper
CORR_THRESHOLD_DEFAULT = 0.8    # |Spearman r| above this = "correlated group"

DATA_PATH_AFLOW    = "data/data_feat.pkl"
DATA_PATH_SUPERCON = "data/supercon_feat.pkl"
FEATURES_JSON       = "data/selected_features.json"
MODEL_DIR            = "models"
OUT_DIR               = "shap_outputs"          # matches your existing convention

UNDERSAMPLE_SEEDS = [0, 7, 21, 42, 99]


# ══════════════════════════════════════════════════════════════════════════════
# GENERIC UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def compute_shap_mean_abs(
    model,
    X: pd.DataFrame,
    n_sample: int = N_SHAP_SAMPLE,
    seed: int = RANDOM_STATE,
) -> pd.Series:
    """Mean |SHAP| per feature from TreeExplainer, on a random subsample."""
    rng   = np.random.default_rng(seed)
    n     = min(n_sample, len(X))
    idx   = rng.choice(len(X), size=n, replace=False)
    X_sub = X.iloc[idx]

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sub)
    if isinstance(sv, list):
        sv = sv[-1]
    mean_abs = np.abs(sv).mean(axis=0)
    return pd.Series(mean_abs, index=X.columns).sort_values(ascending=False)


def spearman_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Spearman correlation matrix among all columns of X."""
    return X.corr(method="spearman")


def correlated_partners(
    corr: pd.DataFrame,
    features: list[str],
    threshold: float,
) -> dict[str, list[str]]:
    """
    For each feature in `features`, list which OTHER features in the
    same list correlate with it above `threshold` (|Spearman r|).
    """
    partners = {}
    for f in features:
        if f not in corr.columns:
            partners[f] = []
            continue
        others = [g for g in features if g != f and g in corr.columns]
        row = corr.loc[f, others]
        partners[f] = row[row.abs() > threshold].index.tolist()
    return partners


def union_find_groups(features: list[str], partners: dict[str, list[str]]) -> dict[str, int]:
    """Assign each feature a group ID via union-find on correlated pairs."""
    parent = {f: f for f in features}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for f, plist in partners.items():
        for p in plist:
            union(f, p)

    roots        = {f: find(f) for f in features}
    unique_roots = {r: i for i, r in enumerate(sorted(set(roots.values())))}
    return {f: unique_roots[r] for f, r in roots.items()}


def report_task(
    task_name: str,
    X: pd.DataFrame,
    model,
    top_features: list[str],
    threshold: float = CORR_THRESHOLD_DEFAULT,
    save: bool = True,
) -> pd.DataFrame:
    """
    Full per-task report: SHAP mean |SHAP|, Spearman correlation matrix
    among ALL selected features, and correlated-group assignment
    restricted to `top_features` (e.g. the Table 17 top-5 list).
    """
    print(f"\n{'=' * 70}\n  Task: {task_name}  (N={len(X)}, features={X.shape[1]})\n{'=' * 70}")

    shap_vals = compute_shap_mean_abs(model, X)
    print("\nMean |SHAP| (top 10, recomputed — compare against your")
    print(f"  shap_outputs/{task_name}/feature_importance_{task_name} file):")
    print(shap_vals.head(10).round(4).to_string())

    missing = [f for f in top_features if f not in X.columns]
    if missing:
        print(f"\n  WARNING: {missing} not found in this task's selected "
              f"feature set — check the top5 list matches Table 4's "
              f"selected features for this task.")

    corr     = spearman_matrix(X)
    partners = correlated_partners(corr, top_features, threshold)
    groups   = union_find_groups(top_features, partners)

    rows = []
    for f in top_features:
        rows.append({
            "feature":          f,
            "mean_abs_shap":    round(float(shap_vals.get(f, float("nan"))), 4),
            "correlated_with":  ", ".join(partners[f]) if partners[f] else "uncorrelated",
            "group_id":         groups[f],
        })
    report = pd.DataFrame(rows)
    print(f"\nCorrelated-group assignment (|Spearman r| > {threshold}):")
    print(report.to_string(index=False))

    if save:
        out_dir = Path(OUT_DIR) / task_name
        out_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(out_dir / f"spearman_groups_{task_name}.json",
                        orient="records", indent=2)
        corr.to_csv(out_dir / f"spearman_matrix_{task_name}.csv")
        print(f"\nSaved -> {out_dir}/spearman_groups_{task_name}.json")
        print(f"Saved -> {out_dir}/spearman_matrix_{task_name}.csv  "
              f"(full {corr.shape[0]}x{corr.shape[1]} matrix, all selected features)")

    return report


# ══════════════════════════════════════════════════════════════════════════════
# PER-TASK LOADERS
# (data/model paths and top-5 lists match Table 17 / Table 4 in the paper —
#  EDIT the model filenames below if your saved checkpoints differ.)
# ══════════════════════════════════════════════════════════════════════════════

def load_features_json(task_key: str) -> list[str]:
    with open(FEATURES_JSON) as f:
        feats = json.load(f)
    return feats[task_key]


def run_enthalpy(threshold: float) -> pd.DataFrame:
    df    = pd.read_pickle(DATA_PATH_AFLOW)
    feats = load_features_json("enthalpy")
    X     = df[feats]
    model = joblib.load(os.path.join(MODEL_DIR, "enthalpy", "enthalpy_lgbm.pkl"))
    top5  = ["chi_mad", "n_atoms", "chi_min", "Tm_max", "dhalf_mad"]
    return report_task("enthalpy", X, model, top5, threshold)


def run_egap_dft(threshold: float) -> pd.DataFrame:
    df    = pd.read_pickle(DATA_PATH_AFLOW)
    df    = df[df["Egap"] > 0.1].reset_index(drop=True)
    feats = load_features_json("egap")
    X     = df[feats]
    model = joblib.load(os.path.join(MODEL_DIR, "egap", "egap_lgbm.pkl"))
    top5  = ["period_hmean", "period_gmean", "pcount_std", "Z_mad", "tm_frac"]
    return report_task("egap_dft", X, model, top5, threshold)


def run_egap_type(threshold: float) -> pd.DataFrame:
    df    = pd.read_pickle(DATA_PATH_AFLOW)
    feats = load_features_json("egap_type")
    mask  = df["Egap_type_numeric"].isin([0, 1])
    X     = df.loc[mask, feats].reset_index(drop=True)

    shap_per_seed = [
       compute_shap_mean_abs(
           joblib.load(os.path.join(MODEL_DIR, "egap_type", f"egap_type_lgbm_seed{s}.pkl")), X)
       for s in UNDERSAMPLE_SEEDS
    ]
    shap_vals = pd.concat(shap_per_seed, axis=1).mean(axis=1)
    model = joblib.load(os.path.join(MODEL_DIR, "egap_type", "egap_type_lgbm_seed0.pkl"))
    top5  = ["tm_frac", "n_atoms", "chi_mean", "Tb_gmean", "Tm_mean"]
    return report_task("egap_type", X, model, top5, threshold)


def run_hm_class_stage1(threshold: float) -> pd.DataFrame:
    df    = pd.read_pickle(DATA_PATH_AFLOW)
    feats = load_features_json("hm_class")
    X     = df[feats]
    # Same seed-averaging note as egap_type applies here if you want the
    # mean|SHAP| column to match Table 17 exactly.
    model = joblib.load(os.path.join(MODEL_DIR, "hm_class", "detector_lgb_seed0.pkl"))
    top5  = ["chi_mad", "Z_hmean", "Z_gmean", "n_atoms", "kappa_hmean"]
    return report_task("hm_class", X, model, top5, threshold)


def run_supercon(threshold: float) -> pd.DataFrame:
    df    = pd.read_pickle(DATA_PATH_SUPERCON)
    df    = df[df["Tc"] > 0.1].reset_index(drop=True)
    feats = load_features_json("supercon")
    X     = df[feats]
    model = joblib.load(os.path.join(MODEL_DIR, "supercon", "supercon_lgbm.pkl"))
    top5  = ["volume_mad", "kappa_std", "volume_max", "delta_chi", "kappa_mad"]
    return report_task("supercon", X, model, top5, threshold)


TASKS = {
    "enthalpy":  run_enthalpy,
    "egap":  run_egap_dft,
    "egap_type": run_egap_type,
    "hm_class":  run_hm_class_stage1,
    "supercon":  run_supercon,
}


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=list(TASKS) + ["all"], default="all",
                   help="Which task to run (default: all five tasks).")
    p.add_argument("--threshold", type=float, default=CORR_THRESHOLD_DEFAULT,
                   help=f"|Spearman r| cutoff for grouping (default {CORR_THRESHOLD_DEFAULT}).")
    args = p.parse_args()

    tasks_to_run = list(TASKS) if args.task == "all" else [args.task]

    reports = {}
    for t in tasks_to_run:
        reports[t] = TASKS[t](args.threshold)

    print(f"\n{'=' * 70}\n  Done. {len(tasks_to_run)} task(s) processed.\n{'=' * 70}")


if __name__ == "__main__":
    main()