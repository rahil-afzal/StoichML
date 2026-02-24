# feature_pruning_lgbm.py

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

RANDOM_STATE   = 42
N_FOLDS        = 5       # CV folds for importance averaging
CUM_IMPORTANCE = 0.85    # cumulative importance threshold

# Must match the seeds used in train_egap_type_ensemble.py
# so that pruning and training see the same balanced distributions.
UNDERSAMPLE_SEEDS = [0, 7, 21, 42, 99]

TARGETS = {
    "enthalpy": {
        "col":  "enthalpy_formation_atom",
        "task": "regression",
    },
    "egap": {
        "col":  "Egap",
        "task": "regression",
        "log":  True,
    },
    "egap_type": {
        "col":             "Egap_type_numeric",
        "task":            "binary",
        "use_undersample": True,    # importance averaged over undersampled folds
    },
    "hm_class": {
        "col":  "hm_class",
        "task": "multiclass",
    },
}

NON_FEATURE_COLS = [
    "compound",
    "spacegroup_relax",
    "Egap",
    "Egap_type",
    "Egap_type_numeric",
    "enthalpy_formation_atom",
    "composition",
    "elements",
    "hm_class",
]

# ── Physics features — force-retained for ALL tasks ───────────────────────────
# Every feature in this set is kept regardless of its importance score.
# Domain knowledge takes precedence over data-driven pruning for features
# with explicit physical justification.
#
# Elemental stat features (e.g. magmom_mean, Ecoh_mean, chi_hmean) flow
# through the cumsum cutoff normally — only composition-level physics
# features computed in phys() are listed here.

PHYSICS_FEATURES = {
    # ── Stoichiometric structure ──────────────────────────────────────────
    "n_elements",       # number of distinct species
    "n_atoms",          # total atoms per formula unit (assumes reduced form)
    "max_weight",       # stoichiometric fraction of majority element

    # ── Entropy features ──────────────────────────────────────────────────
    "conf_entropy",     # configurational mixing entropy -Σ wᵢ·log(wᵢ)
    "S_mag",            # magnetic entropy Σ wᵢ·ln(2Sᵢ+1) — spin degeneracy
    "S_orb",            # orbital entropy over s/p/d/f fractions

    # ── Electronegativity mismatch ────────────────────────────────────────
    "chi_mad",          # weighted EN mismatch
    "delta_chi",        # EN span max−min — Phillips ionicity proxy
    "pair_chi",         # Miedema pairwise Σ wᵢwⱼ(χᵢ−χⱼ)²

    # ── Structural mismatch ───────────────────────────────────────────────
    "r_mad",            # atomic size mismatch — lattice strain proxy
    "mass_std",         # mass dispersion

    # ── Valence electron structure ────────────────────────────────────────
    "val_mean",         # weighted mean valence electron count
    "val_var",          # valence electron dispersion

    # ── d/f shell character ───────────────────────────────────────────────
    "dhalf_mean",       # mean d-shell half-filling distance |dcnt − 5|
    "tm_frac",          # d-block (transition metal) stoichiometric fraction
    "f_frac",           # f-block (lanthanide/actinide) fraction

    # ── Magnetic features ─────────────────────────────────────────────────
    "unpaired_mean",    # Hund's rule free-atom spin proxy
    "unpaired_var",     # spin dispersion across constituent elements
}


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def sanitize(X: pd.DataFrame) -> pd.DataFrame:
    """
    Replace inf values with NaN, then impute column medians.
    Median imputation is used here for feature selection only —
    the training pipeline handles imputation independently.
    """
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X


def undersample_majority(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Randomly undersample the majority class (class 0) to match the
    minority class (class 1) size. Minority class always kept whole.
    Must be identical to the implementation in train_egap_type_ensemble.py.
    """
    idx_minority = y[y == 1].index
    idx_majority = y[y == 0].index

    rng = np.random.default_rng(seed)
    idx_majority_sampled = rng.choice(
        idx_majority, size=len(idx_minority), replace=False
    )

    idx_balanced = np.concatenate([idx_minority, idx_majority_sampled])
    rng.shuffle(idx_balanced)

    return X.loc[idx_balanced], y.loc[idx_balanced]


def lgbm_model(task: str, y: pd.Series) -> lgb.LGBMClassifier | lgb.LGBMRegressor:
    """
    Build a LightGBM model for the given task type.
    importance_type='gain' — correctly weights sparse physics features.
    No class_weight for egap_type binary — training data is already
    balanced by undersampling, so weighting would double-correct.
    """
    if task == "regression":
        return lgb.LGBMRegressor(
            n_estimators=700,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            min_data_in_leaf=30,
            subsample=0.8,
            colsample_bytree=0.8,
            importance_type="gain",
            random_state=RANDOM_STATE,
        )

    if task == "binary":
        # No class_weight — used on already-balanced undersampled data
        return lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            importance_type="gain",
            random_state=RANDOM_STATE,
        )

    if task == "multiclass":
        classes = np.unique(y)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y,
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
    Compute CV-averaged gain importances for standard tasks (regression
    and multiclass). Uses N_FOLDS stratified folds.
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

        model = lgbm_model(task, y_tr)
        model.fit(X_tr, y_tr)

        imp = pd.Series(model.feature_importances_, index=X.columns)
        imp = imp / imp.sum()
        fold_imps.append(imp)

        print(f"    fold {fold}/{N_FOLDS} done")

    return pd.concat(fold_imps, axis=1).mean(axis=1).sort_values(ascending=False)


def average_importances_undersampled(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.Series:
    """
    Compute importance scores for egap_type using the same undersampled
    training distributions used in train_egap_type_ensemble.py.

    For each of N_FOLDS × N_SEEDS combinations:
      - Split into train/val using StratifiedKFold on the FULL dataset
        (so the val fold always reflects the true 5:1 distribution)
      - Undersample only the training fold with each seed
      - Train LightGBM on the balanced training fold
      - Record feature importances

    Average across all N_FOLDS × N_SEEDS runs.

    This ensures features are selected based on what's actually
    informative for the balanced training regime, not the imbalanced
    full dataset where the majority class dominates importance scores.
    """
    cv = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    all_imps = []
    n_total  = N_FOLDS * len(UNDERSAMPLE_SEEDS)
    run      = 0

    for fold, (tr_idx, _) in enumerate(cv.split(X, y), 1):
        X_tr_full = X.iloc[tr_idx]
        y_tr_full = y.iloc[tr_idx]

        for seed in UNDERSAMPLE_SEEDS:
            run += 1
            # Undersample only the training fold — val fold never touched
            X_tr_bal, y_tr_bal = undersample_majority(X_tr_full, y_tr_full, seed)

            model = lgbm_model("binary", y_tr_bal)
            model.fit(X_tr_bal, y_tr_bal)

            imp = pd.Series(model.feature_importances_, index=X.columns)
            imp = imp / imp.sum()
            all_imps.append(imp)

            print(f"    fold {fold}/{N_FOLDS}  seed {seed:3d}  "
                  f"[{run}/{n_total}]  "
                  f"train size: {len(y_tr_bal)} "
                  f"({(y_tr_bal==0).sum()} / {(y_tr_bal==1).sum()})")

    return pd.concat(all_imps, axis=1).mean(axis=1).sort_values(ascending=False)


def prune_features(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
    task_name: str,
) -> list[str]:
    """
    Select features for a single task.

    For egap_type: importances are averaged over N_FOLDS × N_SEEDS
    undersampled training runs, matching the actual training strategy.

    For all other tasks: standard N_FOLDS CV importance averaging.

    In both cases: all PHYSICS_FEATURES are force-retained regardless
    of importance score.
    """
    if cfg.get("log", False):
        y = np.log1p(y.clip(lower=0))

    if cfg.get("use_undersample", False):
        print(f"  [{task_name}] Using undersampled importance averaging "
              f"({N_FOLDS} folds × {len(UNDERSAMPLE_SEEDS)} seeds "
              f"= {N_FOLDS * len(UNDERSAMPLE_SEEDS)} runs)")
        imp = average_importances_undersampled(X, y)
    else:
        imp = average_importances(X, y, cfg)

    # Shifted cumsum — boundary feature is included, not excluded
    cumsum    = imp.cumsum()
    keep_mask = cumsum.shift(1, fill_value=0.0) < CUM_IMPORTANCE
    selected  = set(imp[keep_mask].index)

    # Force-retain all physics features
    available = {f for f in PHYSICS_FEATURES if f in imp.index}
    missing   = {f for f in PHYSICS_FEATURES if f not in imp.index}
    forced_in = available - selected

    if forced_in:
        print(f"  [{task_name}] Force-retained (below cumsum cutoff): "
              f"{sorted(forced_in)}")
    if missing:
        print(f"  [{task_name}] Not found in X (check featurizer): "
              f"{sorted(missing)}")

    selected |= available

    return sorted(selected)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    df = pd.read_pickle(INPUT_PATH)

    results = {}

    for name, cfg in TARGETS.items():
        print(f"\n{'═' * 55}")
        print(f"  Task: {name}")
        print(f"{'═' * 55}")

        y = df[cfg["col"]]
        X = df.drop(
            columns=[c for c in NON_FEATURE_COLS if c in df.columns],
            errors="ignore",
        )
        X = sanitize(X)

        feats = prune_features(X, y, cfg, task_name=name)
        results[name] = feats

        print(f"  → Selected: {len(feats)} features")

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved → {OUTPUT_JSON}")