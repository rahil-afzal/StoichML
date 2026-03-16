# feature_pruning_lgbm.py

import argparse
import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

INPUT_PATH  = "data/data_feat.pkl"
OUTPUT_JSON = "data/selected_features.json"

RANDOM_STATE      = 42
N_FOLDS           = 5
CUM_IMPORTANCE    = 0.85

# Must match seeds in train_egap_type_ensemble.py and train_hm_class_ensemble.py
UNDERSAMPLE_SEEDS = [0, 7, 21, 42, 99]

# Must match MAJORITY_CAP in train_hm_class_ensemble.py.
# hm_class minority (class 2) has only 661 samples — strict 1:1:1
# undersampling gives ~1,983 training rows which is too small for
# 151 features. Cap majority classes at 3,000 each instead, giving
# ~3,661 rows per model (ratio ≈ 4.5:4.5:1 vs original 63:12:1).
MAJORITY_CAP = 3000

TARGETS = {
    "enthalpy": {
        "input": "data/data_feat.pkl",
        "col":   "enthalpy_formation_atom",
        "task":  "regression",
    },
    "egap": {
        "input":  "data/data_feat.pkl",
        "col":    "Egap",
        "task":   "regression",
        "log":    True,
        "filter": ("Egap", ">", 0),
    },
    "egap_type": {
        "input":           "data/data_feat.pkl",
        "col":             "Egap_type_numeric",
        "task":            "binary",
        "use_undersample": True,
        "minority_class":  1,
    },
    "hm_class": {
        "input":           "data/data_feat.pkl",
        "col":             "hm_class",
        "task":            "multiclass",
        "use_undersample": True,
        "minority_class":  2,
        "majority_cap":    MAJORITY_CAP,
    },
    "supercon": {
        "input": "data/supercon_feat.pkl",   # ← different file
        "col":   "Tc",
        "task":  "regression",
    },
}
NON_FEATURE_COLS = [
    "compound",
    "compounds",
    "spacegroup_relax",
    "Egap",
    "Egap_type",
    "Egap_type_numeric",
    "enthalpy_formation_atom",
    "composition",
    "elements",
    "hm_class",
    "Tc",
]

PHYSICS_FEATURES = {
    # ── Stoichiometric structure ──────────────────────────────────────────
    "n_elements",
    "n_atoms",
    "max_weight",

    # ── Entropy features ──────────────────────────────────────────────────
    "conf_entropy",
    "S_mag",
    "S_orb",

    # ── Electronegativity mismatch ────────────────────────────────────────
    "chi_mad",
    "delta_chi",
    "pair_chi",

    # ── Structural mismatch ───────────────────────────────────────────────
    "r_mad",
    "mass_std",

    # ── Valence electron structure ────────────────────────────────────────
    "val_mean",
    "val_var",

    # ── d/f shell character ───────────────────────────────────────────────
    "dhalf_mean",
    "tm_frac",
    "f_frac",

    # ── Magnetic features ─────────────────────────────────────────────────
    "unpaired_mean",
    "unpaired_var",
}


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def sanitize(X: pd.DataFrame) -> pd.DataFrame:
    """
    Replace inf with NaN, impute column medians.
    For feature selection only — training pipeline handles imputation independently.
    """
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X


def apply_filter(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply row-level filter defined in task config before feature selection.

    Filter format: (column, operator, value)
    Supported operators: '>', '>=', '<', '<=', '==', '!='

    The filter is applied to the original df (which still has the raw
    target column) so that filtering on Egap > 0 works even after y
    has been log-transformed in prune_features().

    Returns filtered (X, y) with consistent index.
    """
    if "filter" not in cfg:
        return X, y

    col, op, val = cfg["filter"]

    ops = {
        ">":  lambda s: s >  val,
        ">=": lambda s: s >= val,
        "<":  lambda s: s <  val,
        "<=": lambda s: s <= val,
        "==": lambda s: s == val,
        "!=": lambda s: s != val,
    }

    if op not in ops:
        raise ValueError(f"Unsupported filter operator '{op}'. "
                         f"Choose from {list(ops)}")

    mask = ops[op](df[col])
    n_before = len(y)
    n_after  = mask.sum()

    print(f"  Filter: {col} {op} {val}  →  "
          f"{n_after} / {n_before} rows retained "
          f"({n_before - n_after} dropped)")

    return X.loc[mask], y.loc[mask]


def undersample_to_minority(
    X: pd.DataFrame,
    y: pd.Series,
    minority_class: int,
    seed: int,
    majority_cap: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Undersample all majority classes. Minority class is always kept whole.

    Two modes controlled by majority_cap:

      majority_cap=None  (egap_type, strict balancing)
        Each majority class is downsampled to the fold-local minority size.
        Produces a perfectly balanced dataset.
        n_minority computed from the fold, not the full dataset — avoids
        ValueError when the fold minority count is smaller than the full
        dataset majority count.

      majority_cap=N  (hm_class, capped undersampling)
        Each majority class is downsampled to min(N, available).
        Minority class size is irrelevant to the cap — keeps all minority.
        For hm_class with MAJORITY_CAP=3000:
          ~3,661 rows per model (4.5:4.5:1) vs 1,983 for strict 1:1:1.

    Must produce identical subsets to train_egap_type_ensemble.py and
    train_hm_class_ensemble.py for the same seed.
    """
    rng          = np.random.default_rng(seed)
    idx_minority = y[y == minority_class].index
    n_minority   = len(idx_minority)      # fold-local count

    balanced_idx = list(idx_minority)

    for cls in sorted(y.unique()):
        if cls == minority_class:
            continue
        idx_cls = y[y == cls].index

        if majority_cap is None:
            # Strict balancing — match minority size, guard against small folds
            n_sample = min(n_minority, len(idx_cls))
        else:
            # Capped — draw up to majority_cap regardless of minority size
            n_sample = min(majority_cap, len(idx_cls))

        sampled = rng.choice(idx_cls, size=n_sample, replace=False)
        balanced_idx.extend(sampled)

    balanced_idx = np.array(balanced_idx)
    rng.shuffle(balanced_idx)

    return X.loc[balanced_idx], y.loc[balanced_idx]


def lgbm_model(task: str, y: pd.Series, balanced: bool = False):
    """
    Build a LightGBM model appropriate for the task type.

    importance_type='gain' throughout — correctly weights sparse physics
    features that have few distinct split points.

    balanced=True  → data already balanced by undersampling, no class_weight
    balanced=False → standard class weighting for imbalanced data
    """
    if task == "regression":
        return lgb.LGBMRegressor(
            n_estimators=700,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            min_data_in_leaf=40,
            subsample=0.8,
            colsample_bytree=0.8,
            importance_type="gain",
            verbose=-1,
            random_state=RANDOM_STATE,
        )

    if task == "binary":
        return lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            min_data_in_leaf=40,
            class_weight=None if balanced else "balanced",
            importance_type="gain",
            verbose=-1,
            random_state=RANDOM_STATE,
        )

    if task == "multiclass":
        if balanced:
            class_weight = None
        else:
            classes = np.unique(y)
            weights = compute_class_weight(
                class_weight="balanced", classes=classes, y=y,
            )
            class_weight = dict(zip(classes.tolist(), weights.tolist()))
            if 2 in class_weight:
                class_weight[2] *= 2.0

        return lgb.LGBMClassifier(
            n_estimators=700,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            class_weight=class_weight,
            importance_type="gain",
            random_state=RANDOM_STATE,
        )


def average_importances(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
) -> pd.Series:
    """
    Standard N_FOLDS CV importance averaging.
    Used for enthalpy and egap (egap already filtered to insulators only).
    """
    task = cfg["task"]
    cv   = (
        KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        if task == "regression"
        else StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    )

    fold_imps = []

    for fold, (tr_idx, _) in enumerate(cv.split(X, y), 1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]

        model = lgbm_model(task, y_tr, balanced=False)
        model.fit(X_tr, y_tr)

        imp = pd.Series(model.feature_importances_, index=X.columns)
        imp = imp / imp.sum()
        fold_imps.append(imp)

        print(f"    fold {fold}/{N_FOLDS} done")

    return pd.concat(fold_imps, axis=1).mean(axis=1).sort_values(ascending=False)


def average_importances_undersampled(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
) -> pd.Series:
    """
    Importance averaging over N_FOLDS × N_SEEDS undersampled training runs.
    Used for egap_type and hm_class.

    For each fold × seed:
      - StratifiedKFold on the full dataset (val = true distribution)
      - Undersample only the training fold
      - Train without class weighting (data already balanced)
      - Record normalised gain importances

    Averaging across 25 runs gives stable estimates that reflect the
    balanced training regime rather than the skewed full distribution.
    """
    minority_class = cfg["minority_class"]
    task           = cfg["task"]

    cv       = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    all_imps = []
    n_total  = N_FOLDS * len(UNDERSAMPLE_SEEDS)
    run      = 0

    for fold, (tr_idx, _) in enumerate(cv.split(X, y), 1):
        X_tr_full = X.iloc[tr_idx]
        y_tr_full = y.iloc[tr_idx]

        for seed in UNDERSAMPLE_SEEDS:
            run += 1

            X_tr_bal, y_tr_bal = undersample_to_minority(
                X_tr_full, y_tr_full, minority_class, seed,
                majority_cap=cfg.get("majority_cap", None),
            )

            counts_str = "  ".join(
                f"cls{c}: {(y_tr_bal == c).sum()}"
                for c in sorted(y_tr_bal.unique())
            )

            model = lgbm_model(task, y_tr_bal, balanced=True)
            model.fit(X_tr_bal, y_tr_bal)

            imp = pd.Series(model.feature_importances_, index=X.columns)
            imp = imp / imp.sum()
            all_imps.append(imp)

            print(f"    fold {fold}/{N_FOLDS}  seed {seed:3d}  "
                  f"[{run:2d}/{n_total}]  {counts_str}")

    return pd.concat(all_imps, axis=1).mean(axis=1).sort_values(ascending=False)


def prune_features(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
    task_name: str,
    df_orig: pd.DataFrame,
) -> list[str]:
    """
    Select features for a single task.

    Order of operations:
      1. Apply row filter (e.g. egap > 0 for insulator-only regression)
      2. Apply log transform if configured
      3. Compute importances — undersampled path or standard CV path
      4. Apply cumsum cutoff
      5. Force-retain all PHYSICS_FEATURES

    df_orig is passed so apply_filter() can reference the raw target
    column even after y has been transformed.
    """
    # Step 1 — filter rows before any transformation
    X, y = apply_filter(df_orig, X, y, cfg)

    # Step 2 — log transform
    if cfg.get("log", False):
        y = np.log1p(y.clip(lower=0))

    # Step 3 — importances
    if cfg.get("use_undersample", False):
        n_runs = N_FOLDS * len(UNDERSAMPLE_SEEDS)
        print(f"  [{task_name}] Undersampled importance averaging  "
              f"minority_class={cfg['minority_class']}  "
              f"({N_FOLDS} folds × {len(UNDERSAMPLE_SEEDS)} seeds = {n_runs} runs)")
        imp = average_importances_undersampled(X, y, cfg)
    else:
        imp = average_importances(X, y, cfg)

    # Step 4 — cumsum cutoff (shifted: boundary feature included)
    cumsum    = imp.cumsum()
    keep_mask = cumsum.shift(1, fill_value=0.0) < CUM_IMPORTANCE
    selected  = set(imp[keep_mask].index)

    # Step 5 — force-retain physics features
    # available = {f for f in PHYSICS_FEATURES if f in imp.index}
    # missing   = {f for f in PHYSICS_FEATURES if f not in imp.index}
    # forced_in = available - selected

    # if forced_in:
    #     print(f"  [{task_name}] Force-retained (below cumsum cutoff): "
    #           f"{sorted(forced_in)}")
    # if missing:
    #     print(f"  [{task_name}] Not found in X (check featurizer): "
    #           f"{sorted(missing)}")

    #selected |= available

    return sorted(selected)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run_task(name: str, results: dict) -> dict:   # ← no df parameter
    cfg = TARGETS[name]

    print(f"\n{'═' * 55}")
    print(f"  Task: {name}")
    print(f"{'═' * 55}")

    df = pd.read_pickle(cfg["input"])             # ← loads here

    y = df[cfg["col"]]
    X = df.drop(
        columns=[c for c in NON_FEATURE_COLS if c in df.columns],
        errors="ignore",
    )
    X = sanitize(X)

    feats = prune_features(X, y, cfg, task_name=name, df_orig=df)
    results[name] = feats

    print(f"  → Selected: {len(feats)} features")
    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="StoichML feature pruning — run one task or all."
    )
    parser.add_argument(
        "--task",
        choices=list(TARGETS.keys()) + ["all"],
        default="all",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON) as f:
            results = json.load(f)
        if args.task != "all":
            print(f"Loaded existing results from {OUTPUT_JSON} "
                  f"— will update only '{args.task}'")
    else:
        results = {}

    tasks_to_run = list(TARGETS.keys()) if args.task == "all" else [args.task]

    for name in tasks_to_run:
        results = run_task(name, results)   # ← no df argument

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved → {OUTPUT_JSON}")