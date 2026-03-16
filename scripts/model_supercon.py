
"""
model_supercon.py
─────────────────
Superconducting critical temperature (Tc) regression.
Composition-only features — StoichML featurizer.

Pipeline:
  Filter        Tc > 0 only (removes non-superconductors)
  Transform     log1p(Tc) — right-skewed distribution
  Models        LightGBM + XGBoost ensemble, 5-fold CV
  Analysis      Per-bin error, between/within variance decomposition,
                composition ceiling characterisation

Outputs (models/supercon/)
  supercon_lgbm.pkl
  supercon_xgb.pkl
  supercon_metrics.json
  images/
    parity.png
    residuals.png
    bin_mae.png
    residual_dist.png
    variance_decomposition.png
    feature_importance.png

Usage:
    python -m scripts.model_supercon
    python -m scripts.model_supercon --tc_filter 0 --no_log
"""

from __future__ import annotations

import argparse
import json
import os
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

matplotlib.use("Agg")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH     = "data/supercon_feat.pkl"
FEATURES_JSON = "data/selected_features.json"
OUT_DIR       = Path("models/supercon")
IMG_DIR       = OUT_DIR / "images"

RANDOM_STATE = 42
N_SPLITS     = 5

# Tc bins for per-bin analysis
BINS       = [0, 10, 20, 40, 77, 100, 200]
BIN_LABELS = ["0–10", "10–20", "20–40", "40–77", "77–100", "100+"]

# Literature baseline — Stanev et al. 2018 (RF on supercon2018)
LITERATURE_BASELINE = {
    "model": "Stanev et al. 2018 (RF)",
    "mae":   9.5,
    "r2":    0.88,
}


# ══════════════════════════════════════════════════════════════════════════════
# MODEL BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_lgbm(n_estimators: int = 5000) -> lgb.LGBMRegressor:
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


def build_xgb(
    n_estimators: int = 5000,
    early_stopping: bool = True,
) -> xgb.XGBRegressor:
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
    bin_idx = np.clip(np.digitize(y_true, BINS) - 1, 0, len(BIN_LABELS) - 1)
    stats   = []
    for i, label in enumerate(BIN_LABELS):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        n   = int(mask.sum())
        e   = errors[mask]
        r2b = float(r2_score(y_true[mask], y_pred[mask])) if n > 1 else float("nan")
        stats.append({
            "bin":      label,
            "n":        n,
            "mean_err": float(e.mean()),
            "mae":      float(np.abs(e).mean()),
            "rmse":     float(np.sqrt((e ** 2).mean())),
            "r2":       r2b,
        })
    return stats


def variance_decomposition(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Decompose R² into between-bin and within-bin components.

    Between-bin R²: how well the model predicts bin-level means.
    Within-bin R²:  how well the model predicts variation inside each bin.

    The gap between global R² and within-bin R² quantifies the
    composition-only ceiling for Tc prediction.
    """
    bin_idx    = np.clip(np.digitize(y_true, BINS) - 1, 0, len(BIN_LABELS) - 1)
    bin_means  = np.array([y_true[bin_idx == i].mean()
                           if (bin_idx == i).sum() > 0 else 0.0
                           for i in range(len(BIN_LABELS))])

    # Replace each true value with its bin mean — removes within-bin variance
    y_binmean  = bin_means[bin_idx]
    r2_between = float(r2_score(y_binmean, y_pred))

    # Per-bin R² — within-bin prediction quality
    within_r2s = []
    for i in range(len(BIN_LABELS)):
        mask = bin_idx == i
        if mask.sum() < 2:
            continue
        within_r2s.append(float(r2_score(y_true[mask], y_pred[mask])))

    return {
        "global_r2":          float(r2_score(y_true, y_pred)),
        "between_bin_r2":     r2_between,
        "within_bin_r2_mean": float(np.mean(within_r2s)),
        "within_bin_r2_std":  float(np.std(within_r2s)),
        "within_bin_r2_per_bin": within_r2s,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def run_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    use_log: bool,
) -> dict:
    cv    = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(cv.split(X))

    oof_lgb          = np.zeros(len(y))
    oof_xgb          = np.zeros(len(y))
    fold_metrics     = []
    best_iters_lgb   = []
    best_iters_xgb   = []
    feat_importances = []

    y_train = np.log1p(y) if use_log else y

    print(f"\n{'─' * 74}")
    print(f"  {'Fold':<6} {'LGBM iter':>9}  {'MAE':>7} {'RMSE':>7} {'R²':>7}  "
          f"{'XGB iter':>8}  {'MAE':>7} {'RMSE':>7} {'R²':>7}")
    print(f"{'─' * 74}")

    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        t0 = time.perf_counter()

        Xtr, Xva   = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva   = y_train[tr_idx], y_train[va_idx]
        yva_orig   = y[va_idx]

        lgbm = build_lgbm()
        xgbm = build_xgb(early_stopping=True)

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

        # Invert transform for metrics
        p_lgb = np.expm1(lgbm.predict(Xva)) if use_log else lgbm.predict(Xva)
        p_xgb = np.expm1(xgbm.predict(Xva)) if use_log else xgbm.predict(Xva)

        oof_lgb[va_idx] = p_lgb
        oof_xgb[va_idx] = p_xgb

        m_lgb = compute_metrics(yva_orig, p_lgb)
        m_xgb = compute_metrics(yva_orig, p_xgb)
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


# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(y: np.ndarray, cv: dict) -> dict:
    oof_metrics = {}

    print(f"\n{'─' * 52}")
    print(f"  {'Model':<14} {'MAE (K)':>8} {'RMSE (K)':>9} {'R²':>8}")
    print(f"  {'─' * 42}")

    for name, oof in [("LGBM",     cv["oof_lgb"]),
                      ("XGB",      cv["oof_xgb"]),
                      ("Ensemble", cv["oof_ens"])]:
        m = compute_metrics(y, oof)
        oof_metrics[name.lower()] = m
        print(f"  {name:<14} {m['mae']:8.4f} {m['rmse']:9.4f} {m['r2']:8.4f}")

    print(f"\n  Literature baseline: {LITERATURE_BASELINE['model']}")
    print(f"  MAE ≈ {LITERATURE_BASELINE['mae']} K   R² ≈ {LITERATURE_BASELINE['r2']}")

    print(f"\n  CV mean ± std  ({N_SPLITS} folds)")
    print(f"  {'─' * 52}")
    for model in ("lgbm", "xgb"):
        for metric, label in [("mae", "MAE"), ("rmse", "RMSE"), ("r2", "R²")]:
            vals = [f[model][metric] for f in cv["fold_metrics"]]
            print(f"  {model.upper()} {label:<5}  "
                  f"{np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print(f"\n  Per-bin OOF (Ensemble)")
    print(f"  {'Bin (K)':<10} {'N':>6} {'Mean err':>10} "
          f"{'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print(f"  {'─' * 58}")

    bin_metrics = {}
    for name, oof in [("lgbm",     cv["oof_lgb"]),
                      ("xgb",      cv["oof_xgb"]),
                      ("ensemble", cv["oof_ens"])]:
        stats = bin_analysis(y, oof)
        bin_metrics[name] = stats

    for s in bin_metrics["ensemble"]:
        print(f"  {s['bin']:<10} {s['n']:>6} {s['mean_err']:>+10.3f} "
              f"{s['mae']:>8.3f} {s['rmse']:>8.3f} {s['r2']:>8.4f}")

    print(f"\n  Variance Decomposition (Ensemble OOF)")
    print(f"  {'─' * 52}")
    vd = variance_decomposition(y, cv["oof_ens"])
    print(f"  Global R²:             {vd['global_r2']:8.4f}")
    print(f"  Between-bin R²:        {vd['between_bin_r2']:8.4f}")
    print(f"  Within-bin R² (mean):  {vd['within_bin_r2_mean']:8.4f} "
          f"± {vd['within_bin_r2_std']:.4f}")
    print(f"\n  Interpretation:")
    gap = vd['global_r2'] - vd['within_bin_r2_mean']
    print(f"  {gap*100:.1f}% of explained variance is between-family (composition signal)")
    print(f"  Within-family Tc prediction is beyond composition-only features")

    return {"oof": oof_metrics, "bins": bin_metrics, "variance": vd}


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def plot_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bin_stats: list[dict],
    vd: dict,
) -> None:
    errors     = y_pred - y_true
    abs_errors = np.abs(errors)
    m          = compute_metrics(y_true, y_pred)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"XGBoost+LGBM Ensemble — Tc Regression  (Tc > 0 K)\n"
        f"OOF  MAE={m['mae']:.3f} K   RMSE={m['rmse']:.3f} K   R²={m['r2']:.4f}",
        fontsize=13, fontweight="bold",
    )

    # A — parity
    ax = axes[0, 0]
    sc = ax.scatter(y_true, y_pred, c=abs_errors, cmap="RdYlGn_r",
                    norm=mcolors.Normalize(vmin=0, vmax=15),
                    alpha=0.3, s=6, rasterized=True)
    lim = [0, max(y_true.max(), y_pred.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1, label="perfect")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True Tc (K)")
    ax.set_ylabel("Predicted Tc (K)")
    ax.set_title("A — Parity  (colour = |error|)")
    plt.colorbar(sc, ax=ax, label="|error| (K)")
    ax.legend(fontsize=8)

    # B — signed residuals
    ax = axes[0, 1]
    ax.scatter(y_true, errors, alpha=0.2, s=6,
               color="steelblue", rasterized=True)
    ax.axhline(0, color="k", lw=1, ls="--")
    for bval in BINS[1:-1]:
        ax.axvline(bval, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("True Tc (K)")
    ax.set_ylabel("Residual: Predicted − True (K)")
    ax.set_title("B — Signed Residuals vs True Tc")

    # C — per-bin MAE
    ax = axes[0, 2]
    labels = [s["bin"]  for s in bin_stats]
    maes   = [s["mae"]  for s in bin_stats]
    counts = [s["n"]    for s in bin_stats]
    colors = ["#d73027" if l in ["77–100", "100+"] else "steelblue"
              for l in labels]
    bars = ax.bar(labels, maes, color=colors, edgecolor="white", width=0.6)
    ax.axhline(m["mae"], color="k", ls="--", lw=1,
               label=f"Overall MAE = {m['mae']:.3f} K")
    ax.axhline(LITERATURE_BASELINE["mae"], color="orange", ls="--", lw=1,
               label=f"Stanev 2018 MAE ≈ {LITERATURE_BASELINE['mae']} K")
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.1,
                f"n={n}", ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("Tc Bin (K)")
    ax.set_ylabel("MAE (K)")
    ax.set_title("C — MAE by Tc Bin")
    ax.legend(fontsize=8)

    # D — per-bin R²
    ax = axes[1, 0]
    r2s    = [s["r2"]  for s in bin_stats]
    colors = ["#d73027" if r < 0 else "steelblue" for r in r2s]
    ax.bar(labels, r2s, color=colors, edgecolor="white", width=0.6)
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.axhline(m["r2"], color="green", lw=1, ls="--",
               label=f"Global R² = {m['r2']:.4f}")
    ax.set_xlabel("Tc Bin (K)")
    ax.set_ylabel("R²")
    ax.set_title("D — R² by Tc Bin\n(red = negative R²)")
    ax.legend(fontsize=8)

    # E — variance decomposition bar
    ax = axes[1, 1]
    components = ["Global R²", "Between-bin R²", "Within-bin R²\n(mean)"]
    values     = [vd["global_r2"], vd["between_bin_r2"],
                  vd["within_bin_r2_mean"]]
    colors_vd  = ["steelblue", "seagreen", "tomato"]
    bars_vd    = ax.bar(components, values, color=colors_vd,
                        edgecolor="white", width=0.5)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    for bar, val in zip(bars_vd, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                max(val, 0) + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("R²")
    ax.set_title("E — Variance Decomposition\n"
                 "(between-bin vs within-bin)")
    ax.set_ylim(min(values) - 0.1, 1.05)

    # F — residual distribution
    ax = axes[1, 2]
    ax.hist(errors, bins=80, color="steelblue",
            edgecolor="white", alpha=0.8, density=True)
    ax.axvline(0,            color="k",   lw=1,   ls="--")
    ax.axvline(errors.mean(), color="red", lw=1.5, ls="-",
               label=f"mean = {errors.mean():+.3f} K")
    ax.set_xlabel("Residual (K)")
    ax.set_ylabel("Density")
    ax.set_title("F — Residual Distribution")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = IMG_DIR / "diagnostics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_feature_importance(
    feat_importances: list[np.ndarray],
    feature_names: list[str],
    top_n: int = 20,
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
    ax.set_yticklabels([feature_names[i] for i in idx[::-1]], fontsize=9)
    ax.set_xlabel("Mean Gain Importance (± std over folds)")
    ax.set_title(f"LGBM Feature Importance — Top {top_n}")
    ax.invert_yaxis()
    plt.tight_layout()
    path = IMG_DIR / "feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE METRICS
# ══════════════════════════════════════════════════════════════════════════════

def save_metrics(
    y: np.ndarray,
    cv: dict,
    metrics: dict,
    tc_filter: float,
    use_log: bool,
) -> None:
    payload = {
        "tc_filter":  tc_filter,
        "log_transform": use_log,
        "n_samples":  int(len(y)),
        "n_features": len(cv["feature_names"]),
        "y_stats": {
            "mean": float(y.mean()), "std":  float(y.std()),
            "min":  float(y.min()),  "max":  float(y.max()),
        },
        "oof":              metrics["oof"],
        "bins":             metrics["bins"],
        "variance":         metrics["variance"],
        "cv_fold_metrics":  cv["fold_metrics"],
        "best_iters": {
            "lgbm": cv["best_iters_lgb"],
            "xgb":  cv["best_iters_xgb"],
        },
        "literature_baseline": LITERATURE_BASELINE,
    }
    path = OUT_DIR / "supercon_metrics.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Superconductivity Tc regression — StoichML.",
    )
    p.add_argument("--tc_filter", type=float, default=0.1,
                   help="Min Tc threshold in K (default: 0 → Tc > 0)")
    p.add_argument("--no_log", action="store_true",
                   help="Disable log1p transform of Tc (default: use log1p)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args    = parse_args()
    use_log = not args.no_log

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * 60}")
    print(f"  Superconductivity Tc Regression")
    print(f"  Filter: Tc > {args.tc_filter} K")
    print(f"  Transform: {'log1p(Tc)' if use_log else 'none'}")
    print(f"{'═' * 60}")

    # Load
    df = pd.read_pickle(DATA_PATH)
    df = df[df["Tc"] > args.tc_filter].reset_index(drop=True)

    with open(FEATURES_JSON) as f:
        feats = json.load(f)["supercon"]

    X = df[feats]
    y = df["Tc"].values

    print(f"  Samples:  {len(y)}")
    print(f"  Features: {len(feats)}")
    print(f"  Tc  mean={y.mean():.2f}  std={y.std():.2f}  "
          f"min={y.min():.2f}  max={y.max():.2f} K")

    # CV
    cv      = run_cv(X, y, use_log)
    metrics = print_summary(y, cv)

    # Figures
    print(f"\n  Saving figures → {IMG_DIR}/")
    plot_diagnostics(y, cv["oof_ens"], metrics["bins"]["ensemble"],
                     metrics["variance"])
    plot_feature_importance(cv["feat_importances"], cv["feature_names"])

    # Final fit
    mean_iter_lgb = max(1, int(np.mean(cv["best_iters_lgb"])))
    mean_iter_xgb = max(1, int(np.mean(cv["best_iters_xgb"])))
    print(f"\n  Final fit — LGBM iter={mean_iter_lgb}  XGB iter={mean_iter_xgb}")

    y_train   = np.log1p(y) if use_log else y
    final_lgb = build_lgbm(n_estimators=mean_iter_lgb)
    final_xgb = build_xgb(n_estimators=mean_iter_xgb, early_stopping=False)
    final_lgb.fit(X, y_train)
    final_xgb.fit(X, y_train)

    joblib.dump(final_lgb, OUT_DIR / "supercon_lgbm.pkl")
    joblib.dump(final_xgb, OUT_DIR / "supercon_xgb.pkl")

    save_metrics(y, cv, metrics, args.tc_filter, use_log)

    print(f"\n{'═' * 60}")
    print(f"  Done.  Results in: {OUT_DIR}/")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()