#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model_egap.py
─────────────
Band gap regression benchmark — full comparison table.

Models evaluated in one run:
  SVR RBF          (reference, filter > 0)
  LGBM standard    (reference, filter > 0)
  XGB  standard    (reference, filter > 0)
  LGBM deep        (filter > egap_filter, default 0.1)
  XGB  deep        (filter > egap_filter, default 0.1)
  Ensemble         (LGBM deep + XGB deep)

Outputs  models/egap/
  benchmark_metrics.json
  images/
    comparison_table.png
    ensemble_error_diagnostics.png
    lgbm_deep_error_diagnostics.png
    xgb_deep_error_diagnostics.png
    feature_importance.png

Usage
  python -m scripts.model_egap
  python -m scripts.model_egap --egap_filter 0.1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

OUT_DIR = Path("models/egap")
IMG_DIR = OUT_DIR / "images"

BINS       = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 9.5]
BIN_LABELS = ["0–0.5", "0.5–1", "1–2", "2–4", "4–6", "6–9.5"]

RANDOM_STATE = 42
N_SPLITS     = 5


# ══════════════════════════════════════════════════════════════════════════════
# MODEL BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_lgbm_standard(n_estimators: int = 5000) -> lgb.LGBMRegressor:
    """Standard LGBM — reference model, trained on Egap > 0."""
    return lgb.LGBMRegressor(
        learning_rate    = 0.05,
        num_leaves       = 64,
        n_estimators     = n_estimators,
        min_data_in_leaf = 30,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        importance_type  = "gain",
        random_state     = RANDOM_STATE,
        verbose          = -1,
    )


def build_xgb_standard(
    n_estimators: int = 5000,
    early_stopping: bool = True,
) -> xgb.XGBRegressor:
    """Standard XGB — reference model, trained on Egap > 0."""
    return xgb.XGBRegressor(
        learning_rate         = 0.05,
        max_depth             = 6,
        subsample             = 0.8,
        colsample_bytree      = 0.9,
        n_estimators          = n_estimators,
        tree_method           = "hist",
        random_state          = RANDOM_STATE,
        eval_metric           = "rmse",
        early_stopping_rounds = 200 if early_stopping else None,
    )


def build_lgbm_deep(n_estimators: int = 5000) -> lgb.LGBMRegressor:
    """Deep LGBM — task-specific hyperparameters, trained on Egap > filter."""
    return lgb.LGBMRegressor(
        learning_rate    = 0.03,
        num_leaves       = 127,
        n_estimators     = n_estimators,
        min_data_in_leaf = 10,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        reg_alpha        = 0.05,
        reg_lambda       = 0.1,
        importance_type  = "gain",
        random_state     = RANDOM_STATE,
        verbose          = -1,
    )


def build_xgb_deep(
    n_estimators: int = 5000,
    early_stopping: bool = True,
) -> xgb.XGBRegressor:
    """Deep XGB — task-specific hyperparameters, trained on Egap > filter."""
    return xgb.XGBRegressor(
        learning_rate         = 0.03,
        max_depth             = 8,
        min_child_weight      = 3,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        reg_alpha             = 0.05,
        reg_lambda            = 1.5,
        gamma                 = 0.1,
        n_estimators          = n_estimators,
        tree_method           = "hist",
        random_state          = RANDOM_STATE,
        eval_metric           = "rmse",
        early_stopping_rounds = 200 if early_stopping else None,
    )


def build_svr() -> Pipeline:
    """SVR RBF with standard scaling — reference model."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svr",    SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1)),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def bin_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict]:
    errors  = y_pred - y_true
    bin_idx = np.clip(
        np.digitize(y_true, BINS) - 1, 0, len(BIN_LABELS) - 1
    )
    stats = []
    for i, label in enumerate(BIN_LABELS):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        n   = int(mask.sum())
        e   = errors[mask]
        r2b = (float(r2_score(y_true[mask], y_pred[mask]))
               if n > 1 else float("nan"))
        stats.append({
            "bin":      label,
            "n":        n,
            "mean_err": float(e.mean()),
            "mae":      float(np.abs(e).mean()),
            "rmse":     float(np.sqrt((e ** 2).mean())),
            "r2":       r2b,
        })
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def run_cv_gbm(
    X: pd.DataFrame,
    y: np.ndarray,
    lgbm_builder,
    xgb_builder,
    label: str,
) -> dict:
    """5-fold CV for LGBM + XGB pair. Returns OOF predictions and metrics."""
    cv    = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(cv.split(X))

    oof_lgb          = np.zeros(len(y))
    oof_xgb          = np.zeros(len(y))
    fold_metrics     = []
    best_iters_lgb   = []
    best_iters_xgb   = []
    feat_importances = []

    print(f"\n  {label}")
    print(f"  {'─' * 70}")
    print(f"  {'Fold':<6} {'LGBM iter':>9}  {'MAE':>7} {'RMSE':>7} {'R²':>7}  "
          f"{'XGB iter':>8}  {'MAE':>7} {'RMSE':>7} {'R²':>7}")
    print(f"  {'─' * 70}")

    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        t0 = time.perf_counter()
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y[tr_idx],      y[va_idx]

        lgbm = lgbm_builder()
        xgbm = xgb_builder(early_stopping=True)

        lgbm.fit(
            Xtr, ytr,
            eval_set   = [(Xva, yva)],
            eval_metric= "l2",
            callbacks  = [
                lgb.early_stopping(200, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        xgbm.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

        best_iters_lgb.append(lgbm.best_iteration_)
        best_iters_xgb.append(xgbm.best_iteration)
        feat_importances.append(lgbm.feature_importances_)

        oof_lgb[va_idx] = lgbm.predict(Xva)
        oof_xgb[va_idx] = xgbm.predict(Xva)

        m_lgb = compute_metrics(yva, oof_lgb[va_idx])
        m_xgb = compute_metrics(yva, oof_xgb[va_idx])
        fold_metrics.append({"lgbm": m_lgb, "xgb": m_xgb})

        print(
            f"  {fold}/{N_SPLITS}    "
            f"{lgbm.best_iteration_:9d}  "
            f"{m_lgb['mae']:7.4f} {m_lgb['rmse']:7.4f} {m_lgb['r2']:7.4f}  "
            f"{xgbm.best_iteration:8d}  "
            f"{m_xgb['mae']:7.4f} {m_xgb['rmse']:7.4f} {m_xgb['r2']:7.4f}  "
            f"({time.perf_counter()-t0:.0f}s)"
        )

    return dict(
        oof_lgb          = oof_lgb,
        oof_xgb          = oof_xgb,
        oof_ens          = (oof_lgb + oof_xgb) / 2.0,
        fold_metrics     = fold_metrics,
        best_iters_lgb   = best_iters_lgb,
        best_iters_xgb   = best_iters_xgb,
        feat_importances = feat_importances,
        feature_names    = list(X.columns),
    )


def run_cv_svr(
    X: pd.DataFrame,
    y: np.ndarray,
) -> np.ndarray:
    """5-fold CV for SVR RBF. Returns OOF predictions only."""
    cv    = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof   = np.zeros(len(y))

    print(f"\n  SVR RBF (reference)  — this may take a few minutes ...")

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X), 1):
        t0 = time.perf_counter()
        model = build_svr()
        model.fit(X.iloc[tr_idx].values, y[tr_idx])
        oof[va_idx] = model.predict(X.iloc[va_idx].values)
        m = compute_metrics(y[va_idx], oof[va_idx])
        print(f"  fold {fold}/{N_SPLITS}  "
              f"MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R²={m['r2']:.4f}  "
              f"({time.perf_counter()-t0:.0f}s)")

    return oof


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY PRINTERS
# ══════════════════════════════════════════════════════════════════════════════

def print_comparison_table(results: dict) -> None:
    """Print full comparison table — all models, OOF metrics."""
    print(f"\n{'═' * 72}")
    print(f"  {'Model':<38} {'Filter':>8}  "
          f"{'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print(f"  {'─' * 66}")
    for row in results["comparison"]:
        print(f"  {row['model']:<38} {row['filter']:>8}  "
              f"{row['mae']:>8.4f} {row['rmse']:>8.4f} {row['r2']:>8.4f}")
    print(f"{'═' * 72}")


def print_bin_table(
    y: np.ndarray,
    oof_dict: dict,
    title: str,
) -> dict:
    """Print per-bin analysis for a set of OOF predictions."""
    print(f"\n  Per-bin OOF — {title}")
    print(f"  {'Bin (eV)':<12} {'N':>6} {'Mean err':>10} "
          f"{'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print(f"  {'─' * 60}")

    all_bins = {}
    for name, oof in oof_dict.items():
        stats = bin_analysis(y, oof)
        all_bins[name] = stats
        print(f"\n  {name}")
        for s in stats:
            print(f"  {s['bin']:<12} {s['n']:>6} "
                  f"{s['mean_err']:>+10.4f} "
                  f"{s['mae']:>8.4f} {s['rmse']:>8.4f} "
                  f"{s['r2']:>8.4f}")
    return all_bins


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison_table(results: dict) -> None:
    """Render comparison table as a clean matplotlib figure."""
    rows   = results["comparison"]
    cols   = ["Model", "Filter", "MAE (eV)", "RMSE (eV)", "R²"]
    data   = [[r["model"], r["filter"],
               f"{r['mae']:.4f}", f"{r['rmse']:.4f}", f"{r['r2']:.4f}"]
              for r in rows]

    fig, ax = plt.subplots(figsize=(13, max(4, len(rows) * 0.55 + 1.5)))
    ax.axis("off")
    tbl = ax.table(
        cellText   = data,
        colLabels  = cols,
        cellLoc    = "center",
        loc        = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)

    # Header styling
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight ensemble row
    for i, row in enumerate(rows, 1):
        if "Ensemble" in row["model"]:
            for j in range(len(cols)):
                tbl[i, j].set_facecolor("#d5e8d4")
        elif "reference" in row["model"].lower() or "SVR" in row["model"]:
            for j in range(len(cols)):
                tbl[i, j].set_facecolor("#f5f5f5")

    ax.set_title("Band Gap Regression — Model Comparison (OOF, 5-fold CV)",
                 fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    fname = IMG_DIR / "comparison_table.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fname}")


def plot_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bin_stats: list[dict],
    title_prefix: str,
    out_path: Path,
    egap_filter: float,
) -> None:
    errors     = y_pred - y_true
    abs_errors = np.abs(errors)
    m          = compute_metrics(y_true, y_pred)
    near       = errors[y_true < 0.5]
    rest       = errors[y_true >= 0.5]

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(
        f"{title_prefix} — Band Gap Error Diagnostics  "
        f"(Egap > {egap_filter} eV)\n"
        f"OOF  MAE={m['mae']:.4f} eV   RMSE={m['rmse']:.4f} eV   "
        f"R²={m['r2']:.4f}",
        fontsize=13, fontweight="bold",
    )

    # A — parity
    ax = axes[0, 0]
    sc = ax.scatter(
        y_true, y_pred, c=abs_errors, cmap="RdYlGn_r",
        norm=mcolors.Normalize(vmin=0, vmax=2),
        alpha=0.35, s=7, rasterized=True,
    )
    lim = [0, max(y_true.max(), y_pred.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1, label="perfect")
    ax.axvspan(0, 0.5, alpha=0.08, color="red")
    ax.axvline(0.5, color="red", lw=0.8, ls=":",
               label="0.5 eV boundary")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("DFT Band Gap (eV)")
    ax.set_ylabel("Predicted Band Gap (eV)")
    ax.set_title("A — Parity (colour = |error|)")
    plt.colorbar(sc, ax=ax, label="|error| (eV)")
    ax.legend(fontsize=8)

    # B — signed residuals
    ax = axes[0, 1]
    ax.scatter(y_true, errors, alpha=0.25, s=7,
               color="steelblue", rasterized=True)
    ax.axhline(0,   color="k",   lw=1,   ls="--")
    ax.axvline(0.5, color="red", lw=0.8, ls=":",
               label="0.5 eV boundary")
    ax.set_xlabel("DFT Band Gap (eV)")
    ax.set_ylabel("Residual: Predicted − DFT (eV)")
    ax.set_title("B — Signed Residuals vs DFT Gap")
    ax.legend(fontsize=8)
    near_mask = y_true < 0.5
    if near_mask.sum() > 0:
        bias = errors[near_mask].mean()
        ax.annotate(
            f"Egap < 0.5 eV\nbias: {bias:+.3f} eV",
            xy=(0.25, bias), xytext=(2.0, bias + 0.7),
            arrowprops=dict(arrowstyle="->", color="red"),
            color="red", fontsize=9,
        )

    # C — per-bin MAE
    ax = axes[1, 0]
    labels = [s["bin"] for s in bin_stats]
    maes   = [s["mae"] for s in bin_stats]
    counts = [s["n"]   for s in bin_stats]
    colors = ["#d73027" if l == "0–0.5" else "steelblue"
              for l in labels]
    bars = ax.bar(labels, maes, color=colors,
                  edgecolor="white", width=0.6)
    ax.axhline(m["mae"], color="k", ls="--", lw=1,
               label=f"Overall MAE = {m['mae']:.4f} eV")
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"n={n}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("DFT Band Gap Bin (eV)")
    ax.set_ylabel("MAE (eV)")
    ax.set_title("C — MAE by Band Gap Bin\n(red = near-zero region)")
    ax.legend(fontsize=8)

    # D — residual distributions
    ax = axes[1, 1]
    ax.hist(rest, bins=60, alpha=0.6, color="steelblue",
            label=f"Egap ≥ 0.5 eV  (n={len(rest)})", density=True)
    ax.hist(near, bins=max(10, len(near) // 5), alpha=0.7,
            color="red",
            label=f"Egap < 0.5 eV  (n={len(near)})", density=True)
    ax.axvline(0,            color="k",        lw=1,   ls="--")
    ax.axvline(near.mean(),  color="red",       lw=1.5, ls="-",
               label=f"Near-zero mean: {near.mean():+.3f} eV")
    ax.axvline(rest.mean(),  color="steelblue", lw=1.5, ls="-",
               label=f"Rest mean: {rest.mean():+.3f} eV")
    ax.set_xlabel("Residual: Predicted − DFT (eV)")
    ax.set_ylabel("Density")
    ax.set_title("D — Residual Distribution: Near-zero vs Rest")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_bin_comparison(
    y_ref: np.ndarray,
    oof_ref: np.ndarray,
    y_deep: np.ndarray,
    oof_ens: np.ndarray,
    oof_svr: np.ndarray,
) -> None:
    """
    Side-by-side per-bin MAE for SVR, standard LGBM, and deep ensemble.
    Shows the improvement trajectory across models.
    """
    models = {
        "SVR RBF (ref, >0)":      (y_ref,  oof_svr),
        "LGBM standard (ref, >0)": (y_ref,  oof_ref),
        "Ensemble deep (>filter)": (y_deep, oof_ens),
    }
    colors = ["#e74c3c", "#f39c12", "#27ae60"]

    fig, ax = plt.subplots(figsize=(12, 5))
    x       = np.arange(len(BIN_LABELS))
    width   = 0.25

    for i, (name, (yt, yp)) in enumerate(models.items()):
        stats = bin_analysis(yt, yp)
        maes  = [s["mae"] for s in stats]
        ax.bar(x[:len(maes)] + i * width, maes,
               width=width, label=name,
               color=colors[i], edgecolor="white", alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(BIN_LABELS)
    ax.set_xlabel("DFT Band Gap Bin (eV)")
    ax.set_ylabel("MAE (eV)")
    ax.set_title("Per-Bin MAE Comparison — Reference vs Deep Ensemble",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fname = IMG_DIR / "bin_mae_comparison.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fname}")


def plot_feature_importance(
    feat_importances: list[np.ndarray],
    feature_names: list[str],
    out_path: Path,
    top_n: int = 10,
) -> None:
    mean_imp = np.mean(feat_importances, axis=0)
    std_imp  = np.std(feat_importances,  axis=0)
    idx      = np.argsort(mean_imp)[-top_n:][::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.28)))
    y_pos   = np.arange(top_n)
    ax.barh(y_pos,
            mean_imp[idx][::-1],
            xerr=std_imp[idx][::-1],
            color="steelblue", ecolor="grey",
            alpha=0.85, height=0.7, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [feature_names[i] for i in idx[::-1]], fontsize=9
    )
    ax.set_xlabel("Mean Gain Importance (± std over folds)")
    ax.set_title(
        f"LGBM Deep Feature Importance — Top {top_n}\n"
        f"(mean ± std, {len(feat_importances)} folds)"
    )
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE METRICS
# ══════════════════════════════════════════════════════════════════════════════

def save_metrics(
    results: dict,
    cv_ref: dict,
    cv_deep: dict,
    egap_filter: float,
) -> None:
    payload = {
        "egap_filter":      egap_filter,
        "comparison":       results["comparison"],
        "bins_ref":         results["bins_ref"],
        "bins_deep":        results["bins_deep"],
        "cv_ref_folds":     cv_ref["fold_metrics"],
        "cv_deep_folds":    cv_deep["fold_metrics"],
        "best_iters_ref":  {
            "lgbm": cv_ref["best_iters_lgb"],
            "xgb":  cv_ref["best_iters_xgb"],
        },
        "best_iters_deep": {
            "lgbm": cv_deep["best_iters_lgb"],
            "xgb":  cv_deep["best_iters_xgb"],
        },
    }
    out_path = OUT_DIR / "benchmark_metrics.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Band gap regression — full model comparison."
    )
    p.add_argument("--feat_path",   default="data/data_feat.pkl")
    p.add_argument("--feats_json",  default="data/selected_features.json")
    p.add_argument("--egap_filter", type=float, default=0.1,
                   help="Deep model filter threshold (default 0.1 eV)")
    p.add_argument("--skip_svr",    action="store_true",
                   help="Skip SVR reference (slow on large datasets)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * 60}")
    print(f"  Band Gap Benchmark — Full Comparison")
    print(f"  Reference filter: Egap > 0")
    print(f"  Deep filter:      Egap > {args.egap_filter} eV")
    print(f"{'═' * 60}")

    # Load features
    with open(args.feats_json) as f:
        feats = json.load(f)["egap"]

    # ── Reference dataset: Egap > 0 ──────────────────────────────────────
    df_ref = pd.read_pickle(args.feat_path)
    df_ref = df_ref[df_ref["Egap"] > 0].reset_index(drop=True)
    X_ref  = df_ref[feats]
    y_ref  = df_ref["Egap"].values

    print(f"\n  Reference dataset (Egap > 0):  {len(y_ref)} samples")

    # ── Deep dataset: Egap > filter ───────────────────────────────────────
    df_deep = pd.read_pickle(args.feat_path)
    df_deep = df_deep[df_deep["Egap"] > args.egap_filter].reset_index(
        drop=True
    )
    X_deep  = df_deep[feats]
    y_deep  = df_deep["Egap"].values

    print(f"  Deep dataset    (Egap > {args.egap_filter}): "
          f"{len(y_deep)} samples")
    print(f"  Features: {len(feats)}")

    # ── SVR reference ─────────────────────────────────────────────────────
    if not args.skip_svr:
        print(f"\n{'─' * 60}")
        print(f"  Running SVR RBF reference ...")
        oof_svr = run_cv_svr(X_ref, y_ref)
        m_svr   = compute_metrics(y_ref, oof_svr)
    else:
        print(f"\n  SVR skipped (--skip_svr)")
        oof_svr = None
        m_svr   = {"mae": float("nan"), "rmse": float("nan"),
                   "r2": float("nan")}

    # ── Standard reference models (Egap > 0) ─────────────────────────────
    print(f"\n{'─' * 60}")
    cv_ref = run_cv_gbm(
        X_ref, y_ref,
        build_lgbm_standard, build_xgb_standard,
        "LGBM standard + XGB standard  (reference, Egap > 0)",
    )
    m_lgbm_ref = compute_metrics(y_ref, cv_ref["oof_lgb"])
    m_xgb_ref  = compute_metrics(y_ref, cv_ref["oof_xgb"])

    # ── Deep models (Egap > filter) ───────────────────────────────────────
    print(f"\n{'─' * 60}")
    cv_deep = run_cv_gbm(
        X_deep, y_deep,
        build_lgbm_deep, build_xgb_deep,
        f"LGBM deep + XGB deep  (Egap > {args.egap_filter} eV)",
    )
    m_lgbm_deep = compute_metrics(y_deep, cv_deep["oof_lgb"])
    m_xgb_deep  = compute_metrics(y_deep, cv_deep["oof_xgb"])
    m_ens_deep  = compute_metrics(y_deep, cv_deep["oof_ens"])

    # ── Build comparison table ────────────────────────────────────────────
    comparison = [
        {"model": "SVR RBF (reference)",
         "filter": "> 0",   **m_svr},
        {"model": "LGBM standard (reference)",
         "filter": "> 0",   **m_lgbm_ref},
        {"model": "XGB standard (reference)",
         "filter": "> 0",   **m_xgb_ref},
        {"model": f"LGBM deep",
         "filter": f"> {args.egap_filter}", **m_lgbm_deep},
        {"model": f"XGB deep",
         "filter": f"> {args.egap_filter}", **m_xgb_deep},
        {"model": "Ensemble (LGBM deep + XGB deep)",
         "filter": f"> {args.egap_filter}", **m_ens_deep},
    ]

    results = {"comparison": comparison}

    print_comparison_table(results)

    # ── Per-bin analysis ──────────────────────────────────────────────────
    bins_ref = print_bin_table(
        y_ref,
        {
            "LGBM standard": cv_ref["oof_lgb"],
            "XGB standard":  cv_ref["oof_xgb"],
        },
        title="Reference models (Egap > 0)",
    )
    if oof_svr is not None:
        bins_ref["SVR RBF"] = bin_analysis(y_ref, oof_svr)

    bins_deep = print_bin_table(
        y_deep,
        {
            "LGBM deep": cv_deep["oof_lgb"],
            "XGB deep":  cv_deep["oof_xgb"],
            "Ensemble":  cv_deep["oof_ens"],
        },
        title=f"Deep models (Egap > {args.egap_filter} eV)",
    )
    results["bins_ref"]  = bins_ref
    results["bins_deep"] = bins_deep

    # ── Figures ───────────────────────────────────────────────────────────
    print(f"\n  Saving figures → {IMG_DIR}/")

    plot_comparison_table(results)

    for label, oof, y_use, bstats, fname in [
        ("LGBM deep",  cv_deep["oof_lgb"], y_deep,
         bins_deep["LGBM deep"],
         "lgbm_deep_error_diagnostics.png"),
        ("XGB deep",   cv_deep["oof_xgb"], y_deep,
         bins_deep["XGB deep"],
         "xgb_deep_error_diagnostics.png"),
        ("Ensemble",   cv_deep["oof_ens"], y_deep,
         bins_deep["Ensemble"],
         "ensemble_error_diagnostics.png"),
    ]:
        plot_diagnostics(
            y_use, oof, bstats, label,
            IMG_DIR / fname, args.egap_filter,
        )

    if oof_svr is not None:
        plot_bin_comparison(
            y_ref,  cv_ref["oof_lgb"],
            y_deep, cv_deep["oof_ens"],
            oof_svr,
        )

    plot_feature_importance(
        cv_deep["feat_importances"],
        cv_deep["feature_names"],
        IMG_DIR / "feature_importance.png",
    )

    # ── Final fit (deep models only — saved for SHAP and inference) ───────
    mean_iter_lgb = max(1, int(np.mean(cv_deep["best_iters_lgb"])))
    mean_iter_xgb = max(1, int(np.mean(cv_deep["best_iters_xgb"])))
    print(f"\n  Final fit — LGBM iter={mean_iter_lgb}  "
          f"XGB iter={mean_iter_xgb}")

    final_lgb = build_lgbm_deep(n_estimators=mean_iter_lgb)
    final_xgb = build_xgb_deep(n_estimators=mean_iter_xgb,
                                early_stopping=False)
    final_lgb.fit(X_deep, y_deep)
    final_xgb.fit(X_deep, y_deep)
    joblib.dump(final_lgb, OUT_DIR / "egap_lgbm.pkl")
    joblib.dump(final_xgb, OUT_DIR / "egap_xgb.pkl")
    print(f"  Saved → {OUT_DIR}/egap_lgbm.pkl")
    print(f"  Saved → {OUT_DIR}/egap_xgb.pkl")

    # ── Save metrics ──────────────────────────────────────────────────────
    save_metrics(results, cv_ref, cv_deep, args.egap_filter)

    print(f"\n{'═' * 60}")
    print(f"  Done.  Results in: {OUT_DIR}/")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()