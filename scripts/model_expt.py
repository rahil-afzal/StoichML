"""
expt_gap_pipeline.py
────────────────────
Complete pipeline for the Zhuo et al. experimental band gap dataset.

Steps
─────
  1. Load expt_gap (Zhuo et al. 2018, N=6354) via matminer
  2. Parse compositions → elements / counts
  3. Featurize using StoichML featurizer
  4. Run band gap regression under two filters:
       filter_a : gap_expt > 0    eV  (all insulators)
       filter_b : gap_expt > 0.1  eV  (strict insulator, mirrors AFLOW run)
  5. Feature selection  τ = 0.85 (cumulative gain, LGBM)
  6. OOF training  LGBM + XGB, 5-fold CV, early stopping
  7. Metrics  MAE / RMSE / R²
  8. Per-bin signed residual analysis
  9. SHAP attribution — top-5 per task + top-3 highlighted
  10. Save all results to  models/egap/expt/results.json

Usage
─────
  python expt_gap_pipeline.py

Requirements
────────────
  pip install matminer pymatgen lightgbm xgboost shap scikit-learn mendeleev
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

import lightgbm as lgb
import xgboost  as xgb
import shap

from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)

# ── project paths ─────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent
OUT_DIR = ROOT / "models" / "egap" / "expt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from stoichml.featurizer import featurize

# ══════════════════════════════════════════════════════════════════════════════
# Hyperparameters  — identical to main model_egap deep configuration
# ══════════════════════════════════════════════════════════════════════════════

LGBM_PARAMS = dict(
    n_estimators      = 5000,
    learning_rate     = 0.03,
    num_leaves        = 127,
    max_depth         = 8,
    min_child_samples = 10,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.05,
    reg_lambda        = 0.1,
    min_split_gain    = 0.1,
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1,
)

XGB_PARAMS = dict(
    n_estimators      = 5000,
    learning_rate     = 0.03,
    max_depth         = 8,
    min_child_weight  = 3,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.05,
    reg_lambda        = 1.5,
    tree_method       = "hist",
    random_state      = 42,
    n_jobs            = -1,
    verbosity         = 0,
    EARLY_STOPPING_ROUNDS = 200,
)

CV_SPLITS    = 5
RANDOM_STATE = 42
TAU          = 0.85
EARLY_STOP   = 200

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load and parse dataset
# ══════════════════════════════════════════════════════════════════════════════

def load_and_parse() -> pd.DataFrame:
    print("[1] loading expt_gap (Zhuo et al. 2018, N=6354) …")
    from matminer.datasets import load_dataset
    from pymatgen.core import Composition

    df = load_dataset("expt_gap")
    df = df[df["formula"] != "GaAs0.1P0.9G1128"].reset_index(drop=True)

    print(f"    loaded {len(df)} records")

    def parse_formula(formula):
        comp         = Composition(formula)
        element_dict = comp.get_el_amt_dict()
        elements     = list(element_dict.keys())
        composition  = [
            int(v) if float(v).is_integer() else float(v)
            for v in element_dict.values()
        ]
        return pd.Series({"elements": elements, "composition": composition})

    parsed = df["formula"].apply(parse_formula)
    df     = pd.concat([df, parsed], axis=1)
    print(f"    parsed {len(df)} compositions")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. Featurize
# ══════════════════════════════════════════════════════════════════════════════

def run_featurize(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[2] featurizing …")
    df_feat = featurize(
        df,
        elements_col    = "elements",
        composition_col = "composition",
    )
    feat_cols = [
        c for c in df_feat.columns
        if c not in ("formula", "gap expt", "elements", "composition")
    ]
    print(f"    {len(feat_cols)} feature columns produced")
    return df_feat, feat_cols


# ══════════════════════════════════════════════════════════════════════════════
# 3. Feature selection  (cumulative gain τ = 0.85)
# ══════════════════════════════════════════════════════════════════════════════

def select_features(X: np.ndarray, y: np.ndarray,
                    feat_cols: list[str],
                    label: str) -> list[str]:
    print(f"\n[3] feature selection ({label}) τ={TAU} …")
    kf      = KFold(n_splits=CV_SPLITS, shuffle=True,
                    random_state=RANDOM_STATE)
    all_imp = np.zeros(len(feat_cols))

    for tr, va in kf.split(X):
        m = lgb.LGBMRegressor(
            n_estimators      = 500,
            learning_rate     = 0.05,
            num_leaves        = 64,
            max_depth         = 7,
            min_child_samples = 40,
            random_state      = RANDOM_STATE,
            n_jobs            = -1,
            verbose           = -1,
        )
        m.fit(X[tr], y[tr],
              eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])
        imp     = m.booster_.feature_importance(importance_type="gain")
        imp     = imp / (imp.sum() + 1e-12)
        all_imp += imp

    all_imp /= CV_SPLITS
    order    = np.argsort(all_imp)[::-1]
    cum      = np.cumsum(all_imp[order])
    n_keep   = int(np.searchsorted(cum, TAU)) + 1
    selected = [feat_cols[i] for i in order[:n_keep]]
    print(f"    {n_keep} / {len(feat_cols)} features retained")
    return selected


# ══════════════════════════════════════════════════════════════════════════════
# 4. OOF training
# ══════════════════════════════════════════════════════════════════════════════

def train_oof(X: np.ndarray, y: np.ndarray) -> dict:
    kf       = KFold(n_splits=CV_SPLITS, shuffle=True,
                     random_state=RANDOM_STATE)
    oof_lgbm = np.zeros(len(y))
    oof_xgb  = np.zeros(len(y))
    lgbm_models, xgb_models = [], []
    lgbm_maes, xgb_maes     = [], []

    for fold, (tr, va) in enumerate(kf.split(X)):
        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y[tr], y[va]

        m_lgb = lgb.LGBMRegressor(**LGBM_PARAMS)
        m_lgb.fit(X_tr, y_tr,
                  eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                             lgb.log_evaluation(-1)])
        oof_lgbm[va] = m_lgb.predict(X_va)
        lgbm_models.append(m_lgb)

        m_xgb = xgb.XGBRegressor(**XGB_PARAMS)
        m_xgb.fit(X_tr, y_tr,
                  eval_set=[(X_va, y_va)],
                  verbose=False,)
        oof_xgb[va] = m_xgb.predict(X_va)
        xgb_models.append(m_xgb)

        lgbm_maes.append(mean_absolute_error(y_va, oof_lgbm[va]))
        xgb_maes.append(mean_absolute_error(y_va, oof_xgb[va]))
        print(f"  fold {fold+1}  LGBM={lgbm_maes[-1]:.4f}  "
              f"XGB={xgb_maes[-1]:.4f} eV")

    oof_ens = (oof_lgbm + oof_xgb) / 2.0
    return dict(
        oof_lgbm    = oof_lgbm,
        oof_xgb     = oof_xgb,
        oof_ens     = oof_ens,
        lgbm_models = lgbm_models,
        xgb_models  = xgb_models,
        lgbm_cv_mae = float(np.mean(lgbm_maes)),
        lgbm_cv_std = float(np.std(lgbm_maes)),
        xgb_cv_mae  = float(np.mean(xgb_maes)),
        xgb_cv_std  = float(np.std(xgb_maes)),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. Metrics
# ══════════════════════════════════════════════════════════════════════════════

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE":  round(float(mean_absolute_error(y_true, y_pred)), 6),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 6),
        "R2":   round(float(r2_score(y_true, y_pred)), 6),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. Per-bin signed residual analysis
# ══════════════════════════════════════════════════════════════════════════════

def per_bin(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    edges  = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 25.0]
    labels = ["0–0.5", "0.5–1", "1–2", "2–4", "4–6", "6+"]
    rows   = []
    for lo, hi, lbl in zip(edges[:-1], edges[1:], labels):
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() < 2:
            continue
        res = y_pred[mask] - y_true[mask]
        rows.append({
            "bin":      lbl,
            "n":        int(mask.sum()),
            "mean_err": round(float(res.mean()), 4),
            "MAE":      round(float(np.abs(res).mean()), 4),
            "RMSE":     round(float(np.sqrt((res**2).mean())), 4),
            "R2":       round(float(r2_score(
                                y_true[mask], y_pred[mask])), 4),
        })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# 7. SHAP attribution
# ══════════════════════════════════════════════════════════════════════════════

def compute_shap(model, X: np.ndarray,
                 feat_cols: list[str],
                 top_k: int = 5) -> list[dict]:
    n_shap = min(5000, len(X))
    rng    = np.random.default_rng(42)
    idx    = rng.choice(len(X), n_shap, replace=False)
    X_sub  = X[idx]

    explainer = shap.TreeExplainer(model)
    sv        = explainer.shap_values(X_sub)
    mean_abs  = np.abs(sv).mean(axis=0)

    order  = np.argsort(mean_abs)[::-1]
    top    = [
        {
            "rank":          int(r + 1),
            "feature":       feat_cols[i],
            "mean_abs_shap": round(float(mean_abs[i]), 6),
            "top3":          r < 3,          # flag top-3 explicitly
        }
        for r, i in enumerate(order[:top_k])
    ]
    return top


# ══════════════════════════════════════════════════════════════════════════════
# 8. Run one complete experiment for a given filter
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(df_feat: pd.DataFrame,
                   feat_cols: list[str],
                   target_col: str,
                   min_gap: float,
                   label: str) -> dict:
    print(f"\n{'='*60}")
    print(f" Experiment: {label}  (gap > {min_gap} eV)")
    print(f"{'='*60}")

    # filter
    mask = df_feat[target_col] > min_gap
    df_f = df_feat[mask].reset_index(drop=True)
    y    = df_f[target_col].values.astype(float)
    X_all = df_f[feat_cols].values.astype(float)
    print(f"[filter] {mask.sum()} / {len(df_feat)} compounds retained")

    # feature selection
    selected  = select_features(X_all, y, feat_cols, label)
    sel_idx   = [feat_cols.index(f) for f in selected]
    X         = X_all[:, sel_idx]

    # training
    print(f"\n[4] OOF training ({CV_SPLITS}-fold) …")
    res = train_oof(X, y)

    # metrics
    m_lgbm = metrics(y, res["oof_lgbm"])
    m_xgb  = metrics(y, res["oof_xgb"])
    m_ens  = metrics(y, res["oof_ens"])

    print(f"\n[metrics]")
    print(f"  LGBM     MAE={m_lgbm['MAE']:.4f}  RMSE={m_lgbm['RMSE']:.4f}"
          f"  R²={m_lgbm['R2']:.4f}")
    print(f"  XGB      MAE={m_xgb['MAE']:.4f}  RMSE={m_xgb['RMSE']:.4f}"
          f"  R²={m_xgb['R2']:.4f}")
    print(f"  Ensemble MAE={m_ens['MAE']:.4f}  RMSE={m_ens['RMSE']:.4f}"
          f"  R²={m_ens['R2']:.4f}")

    # per-bin
    bins = per_bin(y, res["oof_ens"])

    # shap — use first-fold LGBM model
    print(f"\n[5] SHAP attribution …")
    shap_top = compute_shap(
        res["lgbm_models"][0], X, selected, top_k=5
    )
    print(f"    top-3 features:")
    for s in shap_top[:3]:
        print(f"      {s['rank']}. {s['feature']}  "
              f"|SHAP|={s['mean_abs_shap']:.4f}")

    return {
        "label":           label,
        "filter":          f"gap_expt > {min_gap} eV",
        "n":               int(len(y)),
        "n_features":      int(len(selected)),
        "selected_features": selected,
        "cv": {
            "lgbm_mae_mean": round(res["lgbm_cv_mae"], 6),
            "lgbm_mae_std":  round(res["lgbm_cv_std"], 6),
            "xgb_mae_mean":  round(res["xgb_cv_mae"],  6),
            "xgb_mae_std":   round(res["xgb_cv_std"],  6),
        },
        "oof_metrics": {
            "LGBM":     m_lgbm,
            "XGBoost":  m_xgb,
            "Ensemble": m_ens,
        },
        "per_bin_signed_residuals": bins,
        "shap_top5": shap_top,
        "shap_top3": [s["feature"] for s in shap_top[:3]],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(" StoichML — Zhuo et al. Experimental Band Gap Pipeline")
    print("=" * 60)

    # 1. load + parse
    df = load_and_parse()

    # 2. featurize (once — reused for both filters)
    df_feat, feat_cols = run_featurize(df)

    # expt_gap dataset uses column name "gap_expt"
    target_col = "gap expt"

    # 3. run both filter experiments
    exp_a = run_experiment(df_feat, feat_cols, target_col,
                           min_gap=0.0,  label="filter_a_gap_gt0")
    exp_b = run_experiment(df_feat, feat_cols, target_col,
                           min_gap=0.1,  label="filter_b_gap_gt0p1")

    # 4. assemble and save results.json
    output = {
        "dataset":     "expt_gap (Zhuo et al. 2018, N=6354)",
        "target":      target_col,
        "featurizer":  "StoichML 22-property × 8-stat + 18 physics = 194-dim",
        "cv_protocol": f"{CV_SPLITS}-fold KFold, OOF, random_state={RANDOM_STATE}",
        "feature_selection": f"LightGBM cumulative gain, τ={TAU}",
        "experiments": [exp_a, exp_b],
    }

    out_path = OUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # 5. print final comparison table
    print("\n" + "=" * 60)
    print(" FINAL SUMMARY")
    print("=" * 60)
    print(f"{'Filter':<25} {'N':>6} {'Feats':>6} "
          f"{'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("-" * 60)
    for exp in [exp_a, exp_b]:
        m = exp["oof_metrics"]["Ensemble"]
        print(f"{exp['filter']:<25} {exp['n']:>6} "
              f"{exp['n_features']:>6} "
              f"{m['MAE']:>8.4f} {m['RMSE']:>8.4f} "
              f"{m['R2']:>8.4f}")
    print("=" * 60)
    print(f"\nAll results saved → {out_path}")


if __name__ == "__main__":
    main()