#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
shap_supercon.py
────────────────
SHAP analysis for the superconductivity Tc regression model.

Two levels of analysis:
  Global   SHAP on all Tc > 0 samples — between-family signal
  Per-bin  SHAP separately on each Tc bin — reveals composition ceiling

Outputs  shap_outputs/supercon/
  shap_summary_global.png
  shap_bar_global.png
  shap_dependence_{feat}_global.png   (top 3)
  feature_importance_global.csv

  shap_summary_{bin}.png              (one per bin)
  shap_bar_{bin}.png
  feature_importance_{bin}.csv

Usage
  python -m scripts.shap_supercon
  python -m scripts.shap_supercon --max_samples 5000
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

RANDOM_STATE  = 42
DATA_PATH     = "data/supercon_feat.pkl"
FEATURES_JSON = "data/selected_features.json"
MODEL_PATH    = "models/supercon/supercon_lgbm.pkl"
OUT_DIR       = "shap_outputs/supercon"

MAX_SAMPLES   = 10000
TOP_K         = 10
DPI           = 150
MIN_BIN_SIZE  = 100   # skip bins with fewer samples

BINS      = [0, 10, 20, 40, 77, 100, 200]
BIN_NAMES = ["0–10", "10–20", "20–40", "40–77", "77–100", "100+"]

# Safe names for filenames
BIN_FNAMES = ["0_10", "10_20", "20_40", "40_77", "77_100", "100plus"]


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_beeswarm(
    shap_vals: np.ndarray,
    X: pd.DataFrame,
    title: str,
    fname: str,
    top_k: int = TOP_K,
) -> None:
    mean_abs  = np.abs(shap_vals).mean(axis=0)
    top_idx   = np.argsort(mean_abs)[-top_k:][::-1]
    top_feats = X.columns[top_idx]
    X_top     = X[top_feats]
    sv_top    = shap_vals[:, top_idx]

    fig, ax = plt.subplots(figsize=(8, max(5, top_k * 0.35)))
    shap.summary_plot(sv_top, X_top, show=False,
                      max_display=top_k, plot_size=None)
    plt.title(title, fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout()
    fig.savefig(fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fname}")


def plot_bar(
    shap_vals: np.ndarray,
    X: pd.DataFrame,
    title: str,
    fname: str,
    top_k: int = TOP_K,
) -> None:
    mean_abs  = np.abs(shap_vals).mean(axis=0)
    top_idx   = np.argsort(mean_abs)[-top_k:][::-1]
    top_feats = X.columns[top_idx]
    X_top     = X[top_feats]
    sv_top    = shap_vals[:, top_idx]

    fig, ax = plt.subplots(figsize=(8, max(5, top_k * 0.35)))
    shap.summary_plot(sv_top, X_top, plot_type="bar", show=False,
                      max_display=top_k, plot_size=None)
    plt.title(title, fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout()
    fig.savefig(fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fname}")


def plot_dependence(
    shap_vals: np.ndarray,
    X: pd.DataFrame,
    title_prefix: str,
    out_dir: str,
    tag: str,
    top_n: int = 3,
) -> None:
    mean_abs  = np.abs(shap_vals).mean(axis=0)
    top_idx   = np.argsort(mean_abs)[-top_n:][::-1]
    top_feats = [X.columns[i] for i in top_idx]

    for feat in top_feats:
        fig, ax = plt.subplots(figsize=(7, 5))
        shap.dependence_plot(feat, shap_vals, X, ax=ax, show=False)
        ax.set_title(f"{title_prefix} — {feat}",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        fname = os.path.join(out_dir,
                             f"shap_dependence_{feat}_{tag}.png")
        fig.savefig(fname, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {fname}")


def save_importance_csv(
    shap_vals: np.ndarray,
    X: pd.DataFrame,
    tag: str,
    out_dir: str,
    top_k: int = TOP_K,
) -> None:
    mean_abs = np.abs(shap_vals).mean(axis=0)
    std_abs  = np.abs(shap_vals).std(axis=0)

    importance = pd.DataFrame({
        "feature":       X.columns,
        "mean_abs_shap": mean_abs,
        "std_abs_shap":  std_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    importance.index     += 1
    importance.index.name = "rank"

    csv_path = os.path.join(out_dir, f"feature_importance_{tag}.csv")
    importance.to_csv(csv_path)
    print(f"  Saved → {csv_path}")

    print(f"\n  Top {top_k} features — {tag}")
    print(f"  {'Rank':<6} {'Feature':<35} {'Mean |SHAP|':>12} {'Std':>10}")
    print(f"  {'─' * 67}")
    for rank, row in importance.head(top_k).iterrows():
        print(f"  {rank:<6} {row['feature']:<35} "
              f"{row['mean_abs_shap']:>12.5f} "
              f"{row['std_abs_shap']:>10.5f}")


def plot_bin_comparison(
    bin_top_features: dict,
    out_dir: str,
    top_n: int = 8,
) -> None:
    """
    Heatmap of mean |SHAP| for top features across all bins.
    Shows how feature importance shifts across the Tc range.
    """
    # Collect all features that appear in top_n of any bin
    all_feats = set()
    for stats in bin_top_features.values():
        all_feats.update(list(stats["feature"])[:top_n])
    all_feats = sorted(all_feats)

    bin_labels = list(bin_top_features.keys())
    matrix     = np.zeros((len(all_feats), len(bin_labels)))

    for j, label in enumerate(bin_labels):
        stats = bin_top_features[label].set_index("feature")
        for i, feat in enumerate(all_feats):
            if feat in stats.index:
                matrix[i, j] = stats.loc[feat, "mean_abs_shap"]

    fig, ax = plt.subplots(figsize=(max(8, len(bin_labels) * 1.5),
                                    max(6, len(all_feats) * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels([f"Tc {b} K" for b in bin_labels],
                       rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(all_feats)))
    ax.set_yticklabels(all_feats, fontsize=9)
    plt.colorbar(im, ax=ax, label="Mean |SHAP|")
    ax.set_title("Feature Importance Across Tc Bins\n"
                 "(shows how composition signal shifts with Tc range)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fname = os.path.join(out_dir, "shap_bin_comparison_heatmap.png")
    fig.savefig(fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fname}")


def plot_shap_magnitude_by_bin(
    bin_mean_shap: dict,
    out_dir: str,
) -> None:
    """
    Bar chart of mean |SHAP| magnitude per bin — directly visualises
    the composition ceiling: bins where SHAP is near zero have no
    composition signal for within-bin Tc prediction.
    """
    labels = list(bin_mean_shap.keys())
    values = list(bin_mean_shap.values())
    colors = ["#d73027" if l in ["77–100", "100+"] else "steelblue"
              for l in labels]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Tc Bin (K)")
    ax.set_ylabel("Mean |SHAP| (all features)")
    ax.set_title("Mean SHAP Magnitude per Tc Bin\n"
                 "(red = within-bin R² < 0 — composition cannot predict Tc here)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fname = os.path.join(out_dir, "shap_magnitude_by_bin.png")
    fig.savefig(fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(max_samples: int = MAX_SAMPLES) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    df = pd.read_pickle(DATA_PATH)
    df = df[df["Tc"] >= 0.01].reset_index(drop=True)

    with open(FEATURES_JSON) as f:
        feats = json.load(f)["supercon"]

    X_full = df[feats]
    y_full = df["Tc"].values

    print(f"\n{'═' * 55}")
    print(f"  SHAP — supercon")
    print(f"{'═' * 55}")
    print(f"  Samples: {len(y_full)}   Features: {len(feats)}")

    # ── Load model ────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"  Model not found: {MODEL_PATH}")
        print(f"  Run model_supercon.py first.")
        return

    model     = joblib.load(MODEL_PATH)
    explainer = shap.TreeExplainer(model)
    print(f"  Loaded → {MODEL_PATH}")

    # ── Global SHAP ───────────────────────────────────────────────────────
    print(f"\n{'─' * 55}")
    print(f"  Global SHAP (all Tc > 0)")
    print(f"{'─' * 55}")

    if len(X_full) > max_samples:
        idx_sample = np.random.default_rng(RANDOM_STATE).choice(
            len(X_full), max_samples, replace=False
        )
        X_global = X_full.iloc[idx_sample].reset_index(drop=True)
        print(f"  Sampled {max_samples} / {len(X_full)} rows")
    else:
        X_global = X_full.reset_index(drop=True)
        print(f"  Using all {len(X_global)} rows")

    print(f"  Computing SHAP ...")
    sv_global = np.array(explainer.shap_values(X_global))
    print(f"  SHAP shape: {sv_global.shape}")

    title_global = "Superconductivity Tc  — Global (all Tc > 0 K)"

    plot_beeswarm(sv_global, X_global, title_global,
                  os.path.join(OUT_DIR, "shap_summary_global.png"))
    plot_bar(sv_global, X_global, title_global,
             os.path.join(OUT_DIR, "shap_bar_global.png"))
    plot_dependence(sv_global, X_global, "Global",
                    OUT_DIR, "global")
    save_importance_csv(sv_global, X_global, "global", OUT_DIR)

    # ── Per-bin SHAP ──────────────────────────────────────────────────────
    print(f"\n{'─' * 55}")
    print(f"  Per-bin SHAP")
    print(f"{'─' * 55}")

    bin_idx        = np.clip(
        np.digitize(y_full, BINS) - 1, 0, len(BIN_NAMES) - 1
    )
    bin_top_feats  = {}
    bin_mean_shap  = {}

    for i, (label, fname_tag) in enumerate(zip(BIN_NAMES, BIN_FNAMES)):
        mask = bin_idx == i
        n    = mask.sum()

        print(f"\n  Bin {label} K  (n={n})")

        if n < MIN_BIN_SIZE:
            print(f"  Skipping — fewer than {MIN_BIN_SIZE} samples")
            continue

        X_bin = X_full[mask].reset_index(drop=True)
        y_bin = y_full[mask]

        # Subsample large bins
        if len(X_bin) > max_samples:
            idx_s = np.random.default_rng(RANDOM_STATE).choice(
                len(X_bin), max_samples, replace=False
            )
            X_bin = X_bin.iloc[idx_s].reset_index(drop=True)
            print(f"  Sampled {max_samples} / {n} rows")

        print(f"  Computing SHAP ...")
        sv_bin = np.array(explainer.shap_values(X_bin))
        print(f"  SHAP shape: {sv_bin.shape}")

        title_bin = (f"Superconductivity Tc — Bin {label} K  "
                     f"(n={n}, {'within-bin R²<0' if label in ['77–100','100+','0–10','10–20','20–40'] else 'R²≈0'})")

        plot_beeswarm(sv_bin, X_bin, title_bin,
                      os.path.join(OUT_DIR, f"shap_summary_{fname_tag}.png"))
        plot_bar(sv_bin, X_bin, title_bin,
                 os.path.join(OUT_DIR, f"shap_bar_{fname_tag}.png"))
        save_importance_csv(sv_bin, X_bin, fname_tag, OUT_DIR)

        # Store for comparison plots
        mean_abs = np.abs(sv_bin).mean(axis=0)
        std_abs  = np.abs(sv_bin).std(axis=0)
        bin_top_feats[label] = pd.DataFrame({
            "feature":       X_bin.columns,
            "mean_abs_shap": mean_abs,
            "std_abs_shap":  std_abs,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        bin_mean_shap[label] = float(mean_abs.mean())

    # ── Cross-bin comparison figures ──────────────────────────────────────
    if len(bin_top_feats) >= 2:
        print(f"\n  Cross-bin comparison figures ...")
        plot_bin_comparison(bin_top_feats, OUT_DIR)
        plot_shap_magnitude_by_bin(bin_mean_shap, OUT_DIR)

    print(f"\n{'═' * 55}")
    print(f"  Done.  All outputs in: {OUT_DIR}/")
    print(f"{'═' * 55}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SHAP analysis for superconductivity Tc model.",
    )
    parser.add_argument(
        "--max_samples", type=int, default=MAX_SAMPLES,
        help=f"Max samples per SHAP computation (default: {MAX_SAMPLES}).",
    )
    args = parser.parse_args()
    main(max_samples=args.max_samples)