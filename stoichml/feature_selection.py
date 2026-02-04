"""
feature_selection.py
--------------------
LightGBM-based feature pruning for StoichML.

Pipeline:
1. Drop non-feature columns
2. Sanitize data (inf/-inf)
3. Variance filtering
4. Drop constant columns
5. LightGBM importance pruning (train-only)
6. Save selected features
"""

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# -------------------------------
# User parameters
# -------------------------------
INPUT_PATH = "data/data_feat.pkl"
OUTPUT_FEATURES_JSON = "data/selected_features.json"

TARGET_COL = "enthalpy_formation_atom"   # change as needed
TEST_SIZE = 0.2
RANDOM_STATE = 42

VARIANCE_QUANTILE = 0.10
IMPORTANCE_CUTOFF = 0.80   # keep top 80% cumulative importance

NON_FEATURE_COLS = [
    "compound",
    "spacegroup_relax",
    "Egap",
    "Egap_type",
    "Egap_type_numeric",
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


# -------------------------------
# Helpers
# -------------------------------
def variance_filter(X, quantile=0.1, plot=True):
    var = X.var()
    thr = var.quantile(quantile)

    if plot:
        plt.hist(var, bins=40)
        plt.axvline(thr, color="r", linestyle="--")
        plt.title("Feature variance distribution")
        plt.show()

    return X.loc[:, var > thr]


def lgbm_prune(X, y):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    model = lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        min_data_in_leaf=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse"
    )

    imp = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    imp_norm = imp / imp.sum()
    keep = imp_norm.cumsum() <= IMPORTANCE_CUTOFF

    selected = set(imp_norm[keep].index)

    # Always keep physics features if they have nonzero importance
    phys_keep = {
        f for f in PHYSICS_FEATURES
        if f in imp_norm.index and imp_norm[f] > 0
    }

    return sorted(selected | phys_keep)


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":

    df = pd.read_pickle(INPUT_PATH)

    y = df[TARGET_COL]

    X = df.drop(
        columns=[c for c in NON_FEATURE_COLS + [TARGET_COL] if c in df.columns],
        errors="ignore",
    )

    print(f"Initial feature count: {X.shape[1]}")

    X = X.replace([np.inf, -np.inf], np.nan)
    nan_cols = X.columns[X.isna().any()]
    if len(nan_cols):
        print(f"Dropping {len(nan_cols)} NaN columns")
        X = X.drop(columns=nan_cols)

    X = variance_filter(X, VARIANCE_QUANTILE, plot=True)
    print(f"After variance filter: {X.shape[1]}")

    const_cols = X.columns[X.nunique() <= 1]
    if len(const_cols):
        X = X.drop(columns=const_cols)

    print(f"After constant removal: {X.shape[1]}")

    selected = lgbm_prune(X, y)
    print(f"Selected features: {len(selected)}")

    os.makedirs(os.path.dirname(OUTPUT_FEATURES_JSON), exist_ok=True)
    with open(OUTPUT_FEATURES_JSON, "w") as f:
        json.dump(selected, f, indent=2)

    print(f"Saved to {OUTPUT_FEATURES_JSON}")
