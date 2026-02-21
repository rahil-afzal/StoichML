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
N_FOLDS        = 5       # folds for importance averaging — more stable than single split
CUM_IMPORTANCE = 0.85    # cumulative importance threshold

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
        "col":  "Egap_type_numeric",
        "task": "binary",
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

# ── Physics features grouped by task ─────────────────────────────────────────
# ALL physics features in a task's group are force-retained regardless of
# their importance score. The rationale: these features have explicit physical
# justification for the target property. A noisy importance estimate from a
# single model should not override domain knowledge.
#
# Global: stoichiometric structure and bond character — universal priors.
# Task-specific: only added to tasks where the physics directly connects
# to the target. This avoids forcing magnetic features into enthalpy models.

PHYSICS_FEATURES = {
    "global": {
        "n_elements",       # number of distinct species
        "max_weight",       # stoichiometric dominance of majority element
        "conf_entropy",     # mixing entropy
        "chi_mad",          # weighted EN mismatch
        "delta_chi",        # max EN span — Phillips ionicity proxy
        "r_mad",            # atomic size mismatch
        "val_mean",         # average valence electron count
        "val_var",          # valence electron dispersion
    },
    "enthalpy": {
        "mass_std",         # mass dispersion — zero-point energy proxy
        "dhalf_mean",       # d-shell half-filling — exchange stability
        "Ecoh_mean",        # weighted cohesive energy — Born-Haber reference
        "Ecoh_std",         # cohesive energy spread across elements
    },
    "egap": {
        "dhalf_mean",
        "tm_frac",          # transition metal fraction — gap suppression
        "f_frac",           # f-block fraction — correlated electron systems
    },
    "egap_type": {
        "dhalf_mean",
        "tm_frac",
        "f_frac",
    },
    "hm_class": {
        "unpaired_mean",    # free-atom spin proxy
        "unpaired_var",     # spin dispersion
        "magmom_mean",      # solid-state ordered moment — strongest half-metal signal
        "magmom_max",       # maximum constituent moment — detects Fe/Co/Ni presence
        "tm_frac",          # d-block fraction
        "f_frac",           # rare-earth fraction — high-moment magnets
        "dhalf_mean",       # half-filled d-shell proximity
        "mass_std",         # heavy elements → stronger spin-orbit coupling
    },
}


def physics_for_task(task_name: str) -> set:
    """Return global + task-specific physics features for a given task."""
    return PHYSICS_FEATURES["global"] | PHYSICS_FEATURES.get(task_name, set())


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def sanitize(X: pd.DataFrame) -> pd.DataFrame:
    """
    Replace inf values with NaN, then impute column medians.

    NOTE: median imputation is used here only for feature selection — it
    prevents rare missing values (e.g. a superheavy element in one row)
    from silently dropping an entire feature column for all rows.
    The actual training pipeline should handle imputation independently.
    """
    X = X.replace([np.inf, -np.inf], np.nan)
    # Impute median per column instead of dropping — preserves all features
    X = X.fillna(X.median(numeric_only=True))
    return X


def lgbm_model(task: str, y: pd.Series):
    """Build a LightGBM model appropriate for the task type."""

    # importance_type="gain" used throughout:
    # "gain" = average improvement in the loss function per split on that feature.
    # This is more informative than the default "split" (count of splits),
    # which systematically overranks high-cardinality continuous features
    # and underranks sparse physics features like n_elements or f_frac.

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
        return lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            class_weight="balanced",
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
            class_weight[2] *= 2.0      # amplify rare half-metal class

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
    Run N_FOLDS cross-validation and return mean normalised feature importances.

    Averaging over folds removes the variance from a single train/test split.
    A different random seed on a single split can change which features
    make the CUM_IMPORTANCE cutoff — CV makes the selection reproducible.
    """
    task = cfg["task"]
    cv   = (
        KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        if task == "regression"
        else StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    )

    fold_imps = []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]

        model = lgbm_model(task, y_tr)
        model.fit(X_tr, y_tr)

        imp = pd.Series(model.feature_importances_, index=X.columns)
        imp = imp / imp.sum()           # normalise per fold before averaging
        fold_imps.append(imp)

        print(f"    fold {fold}/{N_FOLDS} done")

    mean_imp = pd.concat(fold_imps, axis=1).mean(axis=1).sort_values(ascending=False)
    return mean_imp


def prune_features(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
    task_name: str,
) -> list[str]:
    """
    Select features for a single task using CV-averaged gain importances.

    Selection logic:
      1. Include all features whose cumulative importance < CUM_IMPORTANCE
         (FIX: uses < not <= so the boundary feature is included).
      2. Force-retain all physics features for this task regardless of
         their importance score — domain knowledge overrides the data-driven
         cutoff for features with explicit physical justification.
      3. Report which physics features were below the cumsum cutoff but
         retained by force, so you can inspect unexpected cases.
    """
    if cfg.get("log", False):
        y = np.log1p(y.clip(lower=0))

    imp = average_importances(X, y, cfg)

    # FIX: shift cumsum by 1 so the feature that first crosses the threshold
    # is included, not excluded. Original <= cutoff systematically
    # undershot the target retained importance.
    cumsum    = imp.cumsum()
    keep_mask = cumsum.shift(1, fill_value=0.0) < CUM_IMPORTANCE
    selected  = set(imp[keep_mask].index)

    # Force-retain task-specific physics features
    task_physics     = physics_for_task(task_name)
    available_physics = {f for f in task_physics if f in imp.index}
    missing_physics   = task_physics - set(imp.index)

    forced_in = available_physics - selected   # physics not caught by cumsum

    if forced_in:
        print(f"  [{task_name}] Physics features force-retained "
              f"(below cumsum cutoff): {sorted(forced_in)}")
    if missing_physics:
        print(f"  [{task_name}] Physics features not in X "
              f"(check featurizer): {sorted(missing_physics)}")

    selected |= available_physics

    return sorted(selected)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    df = pd.read_pickle(INPUT_PATH)

    results = {}

    for name, cfg in TARGETS.items():
        print(f"\n{'═'*55}")
        print(f"  Task: {name}")
        print(f"{'═'*55}")

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