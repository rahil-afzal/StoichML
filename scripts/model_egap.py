"""
egap_benchmark.py
─────────────────
Band gap regression benchmark — LGBM + XGBoost ensemble.
Trains on insulators (Egap > egap_filter) using composition-only features.

Outputs (all saved to models/egap/)
  metrics.json                  OOF + per-fold + per-bin metrics
  lgbm_error_diagnostics.png
  xgb_error_diagnostics.png
  ensemble_error_diagnostics.png
  feature_importance.png        Top-30 LGBM gain importance (mean over folds)

Usage
-----
  python egap_benchmark.py
  python egap_benchmark.py --egap_filter 0.1
  python egap_benchmark.py --feat_path data/data_feat.pkl --egap_filter 0.0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = Path("models/egap")
IMG_DIR = OUT_DIR / "images"
# ── Band gap bins ─────────────────────────────────────────────────────────────
BINS       = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 9.5]
BIN_LABELS = ["0–0.5", "0.5–1", "1–2", "2–4", "4–6", "6–9.5"]

RANDOM_STATE = 42
N_SPLITS     = 5


# ── Model builders ────────────────────────────────────────────────────────────

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


def build_xgb(n_estimators: int = 5000,
              early_stopping: bool = True) -> xgb.XGBRegressor:
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


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def bin_analysis(y_true: np.ndarray,
                 y_pred: np.ndarray) -> list[dict]:
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
            "bin": label, "n": n,
            "mean_err": float(e.mean()),
            "mae":      float(np.abs(e).mean()),
            "rmse":     float(np.sqrt((e ** 2).mean())),
            "r2":       r2b,
        })
    return stats


# ── Cross-validation ──────────────────────────────────────────────────────────

def run_cv(X: pd.DataFrame, y: np.ndarray) -> dict:
    cv    = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(cv.split(X))

    oof_lgb          = np.zeros(len(y))
    oof_xgb          = np.zeros(len(y))
    fold_metrics     = []
    best_iters_lgb   = []
    best_iters_xgb   = []
    feat_importances = []

    print(f"\n{'─' * 72}")
    print(f"  {'Fold':<6} {'LGBM iter':>9}  {'MAE':>7} {'RMSE':>7} {'R²':>7}  "
          f"{'XGB iter':>8}  {'MAE':>7} {'RMSE':>7} {'R²':>7}")
    print(f"{'─' * 72}")

    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        t0 = time.perf_counter()
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y[tr_idx],      y[va_idx]

        lgbm = build_lgbm()
        xgbm = build_xgb(early_stopping=True)

        lgbm.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            eval_metric="l2",
            callbacks=[
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


# ── Console summary ───────────────────────────────────────────────────────────

def print_summary(y: np.ndarray, cv: dict) -> dict:
    oof_metrics = {}
    print(f"\n{'─' * 52}")
    print(f"  {'Model':<14} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print(f"  {'─' * 42}")
    for name, oof in [("LGBM",     cv["oof_lgb"]),
                      ("XGB",      cv["oof_xgb"]),
                      ("Ensemble", cv["oof_ens"])]:
        m = compute_metrics(y, oof)
        oof_metrics[name.lower()] = m
        print(f"  {name:<14} {m['mae']:8.4f} {m['rmse']:8.4f} {m['r2']:8.4f}")

    print(f"\n  CV mean ± std  ({N_SPLITS} folds)")
    print(f"  {'─' * 52}")
    for model in ("lgbm", "xgb"):
        for metric, label in [("mae", "MAE"), ("rmse", "RMSE"), ("r2", "R²")]:
            vals = [f[model][metric] for f in cv["fold_metrics"]]
            print(f"  {model.upper()} {label:<5}  "
                  f"{np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print(f"\n  Per-bin OOF")
    print(f"  {'Bin (eV)':<12} {'N':>6} {'Mean err':>10} "
          f"{'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print(f"  {'─' * 60}")
    bin_metrics = {}
    for name, oof in [("LGBM",     cv["oof_lgb"]),
                      ("XGB",      cv["oof_xgb"]),
                      ("Ensemble", cv["oof_ens"])]:
        stats = bin_analysis(y, oof)
        bin_metrics[name.lower()] = stats
        print(f"\n  {name}")
        for s in stats:
            print(f"  {s['bin']:<12} {s['n']:>6} {s['mean_err']:>+10.4f} "
                  f"{s['mae']:>8.4f} {s['rmse']:>8.4f} {s['r2']:>8.4f}")

    return {"oof": oof_metrics, "bins": bin_metrics}


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_diagnostics(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     bin_stats: list[dict],
                     title_prefix: str,
                     out_path: Path,
                     egap_filter: float) -> None:
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
    sc = ax.scatter(y_true, y_pred, c=abs_errors, cmap="RdYlGn_r",
                    norm=mcolors.Normalize(vmin=0, vmax=2),
                    alpha=0.35, s=7, rasterized=True)
    lim = [0, max(y_true.max(), y_pred.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1, label="perfect")
    ax.axvspan(0, 0.5, alpha=0.08, color="red")
    ax.axvline(0.5, color="red", lw=0.8, ls=":", label="0.5 eV boundary")
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
    ax.axvline(0.5, color="red", lw=0.8, ls=":", label="0.5 eV boundary")
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
    colors = ["#d73027" if l == "0–0.5" else "steelblue" for l in labels]
    bars   = ax.bar(labels, maes, color=colors, edgecolor="white", width=0.6)
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
    ax.hist(near, bins=max(10, len(near) // 5), alpha=0.7, color="red",
            label=f"Egap < 0.5 eV  (n={len(near)})", density=True)
    ax.axvline(0,           color="k",        lw=1,   ls="--")
    ax.axvline(near.mean(), color="red",       lw=1.5, ls="-",
               label=f"Near-zero mean: {near.mean():+.3f} eV")
    ax.axvline(rest.mean(), color="steelblue", lw=1.5, ls="-",
               label=f"Rest mean: {rest.mean():+.3f} eV")
    ax.set_xlabel("Residual: Predicted − DFT (eV)")
    ax.set_ylabel("Density")
    ax.set_title("D — Residual Distribution: Near-zero vs Rest")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_feature_importance(feat_importances: list[np.ndarray],
                            feature_names: list[str],
                            out_path: Path,
                            top_n: int = 10) -> None:
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
    ax.set_title(f"LGBM Feature Importance — Top {top_n}\n"
                 f"(mean ± std, {len(feat_importances)} folds)")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_path, dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Save metrics.json ─────────────────────────────────────────────────────────

def save_metrics(y: np.ndarray,
                 cv: dict,
                 metrics: dict,
                 egap_filter: float) -> None:
    payload = {
        "egap_filter": egap_filter,
        "n_samples":   int(len(y)),
        "n_features":  len(cv["feature_names"]),
        "y_stats": {
            "mean": float(y.mean()), "std":  float(y.std()),
            "min":  float(y.min()),  "max":  float(y.max()),
        },
        "oof":  metrics["oof"],
        "bins": metrics["bins"],
        "cv_fold_metrics":  cv["fold_metrics"],
        "best_iters": {
            "lgbm": cv["best_iters_lgb"],
            "xgb":  cv["best_iters_xgb"],
        },
    }
    out_path = OUT_DIR / "benchmark_metrics.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Band gap regression benchmark — LGBM + XGBoost.",
    )
    p.add_argument("--feat_path",   default="data/data_feat.pkl",
                   help="Featurized data pickle (default: data/data_feat.pkl)")
    p.add_argument("--feats_json",  default="data/selected_features.json",
                   help="Selected features JSON (default: data/selected_features.json)")
    p.add_argument("--egap_filter", type=float, default=0.1,
                   help="Min Egap threshold in eV (default: 0.1)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * 55}")
    print(f"  Band Gap Benchmark  |  Egap > {args.egap_filter} eV")
    print(f"{'═' * 55}")

    # Load
    df = pd.read_pickle(args.feat_path)
    df = df[df["Egap"] > args.egap_filter].reset_index(drop=True)

    with open(args.feats_json) as f:
        feats = json.load(f)["egap"]

    X = df[feats]
    y = df["Egap"].values

    print(f"  Samples:   {len(y)}")
    print(f"  Features:  {len(feats)}")
    print(f"  Egap  mean={y.mean():.3f}  std={y.std():.3f}  "
          f"min={y.min():.3f}  max={y.max():.3f} eV")

    # CV
    cv      = run_cv(X, y)
    metrics = print_summary(y, cv)

    # Figures
    print(f"\n  Saving figures → {OUT_DIR}/")
    for label, oof, bstats, fname in [
        ("LGBM",     cv["oof_lgb"], metrics["bins"]["lgbm"],
         "lgbm_error_diagnostics.png"),
        ("XGB",      cv["oof_xgb"], metrics["bins"]["xgb"],
         "xgb_error_diagnostics.png"),
        ("Ensemble", cv["oof_ens"], metrics["bins"]["ensemble"],
         "ensemble_error_diagnostics.png"),
    ]:
        plot_diagnostics(y, oof, bstats, label,
                         IMG_DIR / fname, args.egap_filter)

    plot_feature_importance(
        cv["feat_importances"],
        cv["feature_names"],
        IMG_DIR / "feature_importance.png",
    )

    # Save metrics
    save_metrics(y, cv, metrics, args.egap_filter)

    print(f"\n{'═' * 55}")
    print(f"  Done.  Results in: {OUT_DIR}/")
    print(f"{'═' * 55}\n")


if __name__ == "__main__":
    main()