# feature_pruning_lgbm.py

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

INPUT_PATH  = "data/data_feat.pkl"
OUTPUT_JSON = "data/selected_features.json"

RANDOM_STATE    = 42
TEST_SIZE       = 0.2
CUM_IMPORTANCE  = 0.80      # cumulative importance threshold for pruning

# Minimum normalised importance a physics feature must have to be force-retained.
# 1/n_features is the "random chance" baseline — features below this carry
# less signal than a random column and should not be force-retained.
# Set dynamically in prune_features() based on actual feature count.
PHYSICS_MIN_IMP_FACTOR = 0.5   # must be > 0.5× random-chance importance

# A feature is included in "core" if it is selected for at least this many tasks
CORE_MIN_TASKS = 2

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

# ── Physics features, grouped by task relevance ───────────────────────────────
# Splitting by task prevents task-irrelevant physics features from being
# force-retained where they are genuinely noise.
#
# GLOBAL: relevant for all tasks — stoichiometric structure and bond character
# are universal priors regardless of target property.
#
# ENTHALPY: thermochemical features — cohesive energy and atomic size/mass
# mismatch encode the energetic cost of forming the compound from elements.
#
# EGAP / EGAP_TYPE: electronic structure features — electronegativity spread
# (delta_chi) is the primary driver of band gap via Phillips ionicity theory.
# Valence electron count controls whether the gap is zero (metal) or nonzero.
#
# HM_CLASS: magnetic features — magmom, unpaired, f_frac are the primary
# signals for half-metal and spintronic classification. Forcing these into
# enthalpy or egap models adds noise.

PHYSICS_FEATURES = {
    # shared across all tasks
    "global": {
        "n_elements",       # number of distinct species — structural prior
        "max_weight",       # stoichiometric dominance of majority element
        "conf_entropy",     # mixing entropy — equiatomic vs. doped
        "chi_mad",          # weighted EN mismatch — bond ionicity
        "delta_chi",        # max EN span — Phillips ionicity proxy (renamed from chi_rng)
        "r_mad",            # atomic size mismatch — lattice strain
        "val_mean",         # average valence electron count
        "val_var",          # valence electron dispersion
    },
    "enthalpy": {
        "mass_std",         # mass dispersion — correlates with zero-point energy
        "dhalf_mean",       # d-shell half-filling — stability via exchange
    },
    "egap": {
        "dhalf_mean",       # d-shell character → metallic vs. insulating
        "tm_frac",          # transition metal fraction → band gap suppression
        "f_frac",           # f-block fraction → heavy-fermion / correlated electron
    },
    "egap_type": {
        "dhalf_mean",
        "tm_frac",
        "f_frac",
        "val_mean",         # already in global but explicit here for clarity
    },
    "hm_class": {
        "unpaired_mean",    # Hund's rule free-atom spin proxy
        "unpaired_var",     # spin dispersion — uniform vs. concentrated moment
        "tm_frac",          # d-block fraction — primary half-metal driver
        "f_frac",           # rare-earth fraction — high-moment magnets
        "dhalf_mean",       # half-filled d-shell → Hund's maximum
        "mass_std",         # heavy elements → stronger SOC
    },
}

def physics_for_task(task_name: str) -> set:
    """Return the union of global + task-specific physics features."""
    return PHYSICS_FEATURES["global"] | PHYSICS_FEATURES.get(task_name, set())


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def sanitize(X: pd.DataFrame) -> pd.DataFrame:
    """Drop inf values and columns with any NaN."""
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.dropna(axis=1)


def lgbm_model(task: str, y: pd.Series):
    if task == "regression":
        return lgb.LGBMRegressor(
            n_estimators=700,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            min_data_in_leaf=30,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
        )

    if task == "binary":
        return lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            class_weight="balanced",
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

        # Amplify rare class (half-metal, class 2) — minority oversampling
        # proxy at the loss level. Factor of 2× on top of balanced weights.
        if 2 in class_weight:
            class_weight[2] *= 2.0

        return lgb.LGBMClassifier(
            n_estimators=700,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
        )


def prune_features(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
    task_name: str,
) -> list[str]:
    """
    Train a LightGBM model, extract feature importances, and return the
    minimal feature set that explains CUM_IMPORTANCE of the total importance.

    Physics features are force-retained if their normalised importance
    exceeds a minimum threshold (0.5*random-chance baseline). This prevents
    junk retention (imp > 0 from a single split) while ensuring genuinely
    informative physics features are not pruned by the cumsum cutoff.

    FIX from original: cumsum <= threshold excludes the feature that first
    pushes cumsum over the line. Changed to < threshold so the boundary
    feature is included — otherwise the actual retained importance is
    systematically below the target.
    """
    if cfg.get("log", False):
        y = np.log1p(y.clip(lower=0))

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if cfg["task"] != "regression" else None,
    )

    model = lgbm_model(cfg["task"], y_tr)
    model.fit(X_tr, y_tr)

    imp = pd.Series(
        model.feature_importances_,
        index=X.columns,
    ).sort_values(ascending=False)

    imp_norm = imp / imp.sum()
    cumsum = imp_norm.cumsum()
    keep_mask = cumsum.shift(1, fill_value=0.0) < CUM_IMPORTANCE
    selected = set(imp_norm[keep_mask].index)

    # Physics feature retention with signal threshold.
    # Random-chance importance baseline = 1 / n_features.
    # Require physics features to clear 0.5× that baseline to be retained.
    # This excludes features that appear in the model from random splits only.
    random_baseline = 1.0 / len(X.columns)
    min_imp = PHYSICS_MIN_IMP_FACTOR * random_baseline

    task_physics = physics_for_task(task_name)
    retained_physics = {
        f for f in task_physics
        if f in imp_norm.index and imp_norm[f] >= min_imp
    }
    dropped_physics = {
        f for f in task_physics
        if f in imp_norm.index and imp_norm[f] < min_imp
    }

    if dropped_physics:
        print(f"  [{task_name}] Physics features below signal threshold "
              f"(dropped): {sorted(dropped_physics)}")

    selected |= retained_physics

    return sorted(selected)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    df = pd.read_pickle(INPUT_PATH)

    results   = {}
    all_sets  = []

    for name, cfg in TARGETS.items():
        print(f"\n── {name} ──────────────────────────────────────────────")

        y = df[cfg["col"]]
        X = df.drop(
            columns=[c for c in NON_FEATURE_COLS if c in df.columns],
            errors="ignore",
        )
        X = sanitize(X)

        feats = prune_features(X, y, cfg, task_name=name)
        results[name] = feats
        all_sets.append(set(feats))

        print(f"  Selected: {len(feats)} features")

    # ── Core feature set: selected for at least CORE_MIN_TASKS tasks ─────
    counts = Counter(f for s in all_sets for f in s)
    core   = sorted(f for f, c in counts.items() if c >= CORE_MIN_TASKS)
    results["core"] = core

    print(f"\n── Core (≥{CORE_MIN_TASKS} tasks): {len(core)} features ──────")

    # ── Feature overlap summary ───────────────────────────────────────────
    print("\n── Per-task feature counts ──────────────────────────────────")
    for name in TARGETS:
        exclusive = set(results[name]) - set(core)
        print(f"  {name:12s}: {len(results[name]):3d} total  "
              f"| {len(exclusive):3d} task-exclusive")

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved feature sets → {OUTPUT_JSON}")