# feature_pruning_lgbm.py

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ===============================
# Config
# ===============================
INPUT_PATH = "data/data_feat.pkl"
OUTPUT_JSON = "data/selected_features.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CUM_IMPORTANCE = 0.85

TARGETS = {
    "enthalpy": {
        "col": "enthalpy_formation_atom",
        "task": "regression"
    },
    "egap": {
        "col": "Egap",
        "task": "regression",
        "log": True
    },
    "egap_type": {
        "col": "Egap_type_numeric",
        "task": "binary"
    },
    "hm_class": {
        "col": "hm_class",
        "task": "multiclass"
    }
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

PHYSICS_FEATURES = {
    "conf_entropy",
    "chi_mad",
    "chi_rng",
    "r_mad",
    "mass_std",
    "val_mean",
    "val_var",
    "dhalf_mean",
    "tm_frac",
}

# ===============================
# Helpers
# ===============================
def sanitize(X):
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.dropna(axis=1)


def lgbm_model(task, y):
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
            y=y
        )
        class_weight = dict(zip(classes, weights))

        # amplify rare class (class 2)
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


def prune_features(X, y, cfg):
    if cfg.get("log", False):
        y = np.log1p(y.clip(lower=0))

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if cfg["task"] != "regression" else None
    )

    model = lgbm_model(cfg["task"], y_tr)
    model.fit(X_tr, y_tr)

    imp = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    imp = imp / imp.sum()
    keep = imp.cumsum() <= CUM_IMPORTANCE

    selected = set(imp[keep].index)

    # retain physics features if they carry signal
    selected |= {
        f for f in PHYSICS_FEATURES
        if f in imp.index and imp[f] > 0
    }

    return sorted(selected)


# ===============================
# Main
# ===============================
if __name__ == "__main__":

    df = pd.read_pickle(INPUT_PATH)

    results = {}
    all_sets = []

    for name, cfg in TARGETS.items():
        y = df[cfg["col"]]

        X = df.drop(
            columns=[c for c in NON_FEATURE_COLS if c in df.columns],
            errors="ignore"
        )

        X = sanitize(X)

        feats = prune_features(X, y, cfg)
        results[name] = feats
        all_sets.append(set(feats))

        print(f"{name}: {len(feats)} features")

    # shared core (≥2 targets)
    from collections import Counter
    counts = Counter(f for s in all_sets for f in s)
    core = sorted([f for f, c in counts.items() if c >= 2])

    results["core"] = core

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved feature sets to {OUTPUT_JSON}")
