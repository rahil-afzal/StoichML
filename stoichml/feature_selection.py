"""
feature_selection.py
--------------------
Feature selection pipeline for StoichML.

Steps:
1. Drop non-feature columns (targets, identifiers, etc.)
2. Sanitize data (replace inf/-inf, drop columns with NaNs)
3. Variance filtering (drop low-variance features)
4. Drop constant columns (optional)
5. VIF filtering (drop multicollinear features)
6. Save selected features to JSON
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# -------------------------------
# User parameters
# -------------------------------
INPUT_PATH = "data/data_feat.pkl"
OUTPUT_FEATURES_JSON = "data/selected_features.json"
VARIANCE_QUANTILE = 0.10  # Drop bottom 10% low-variance features
VIF_THRESHOLD = 5.0       # Drop features with VIF > 5.0

# Columns to drop (targets, identifiers, etc.)
NON_FEATURE_COLS = [
    'compound', 'spacegroup_relax', 'Egap', 'Egap_type',
    'enthalpy_formation_atom', 'composition', 'elements',
    'Egap_type_numeric', 'hm_class'
]

# -------------------------------
# Helper functions
# -------------------------------
def variance_filter(X: pd.DataFrame, quantile: float = 0.10, plot: bool = True) -> pd.DataFrame:
    """Removes features with variance below given quantile threshold."""
    variances = X.var()
    threshold = variances.quantile(quantile)
    if plot:
        plt.figure(figsize=(10, 6))
        plt.hist(variances, bins=30, edgecolor='k', alpha=0.7)
        plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.4f}')
        plt.title('Histogram of Feature Variances')
        plt.xlabel('Variance')
        plt.ylabel('Number of Features')
        plt.legend()
        plt.grid(True)
        plt.show()
    return X.loc[:, variances > threshold]

def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Computes VIF for all features."""
    X_const = add_constant(X)
    vif_data = pd.DataFrame({
        "feature": X_const.columns,
        "VIF": [variance_inflation_factor(X_const.values, i)
                for i in range(X_const.shape[1])]
    })
    return vif_data[vif_data["feature"] != "const"]

def drop_high_vif(X: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
    """Drops features with VIF > threshold."""
    X = X.copy()
    while True:
        vif_df = calculate_vif(X).sort_values(by="VIF", ascending=False)
        if vif_df.empty or vif_df.iloc[0]['VIF'] <= threshold:
            break
        col = vif_df.iloc[0]['feature']
        print(f"Dropping {col} due to high VIF ({vif_df.iloc[0]['VIF']:.2f})")
        X = X.drop(columns=[col])
    return X

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Pickle file not found at {INPUT_PATH}")

    df = pd.read_pickle(INPUT_PATH)

    # Drop non-feature columns
    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns], errors='ignore')
    print(f"Initial feature count: {X.shape[1]}")

    # -------------------------------
    # Sanitize data
    # -------------------------------
    X = X.replace([float('inf'), -float('inf')], pd.NA)
    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        print(f"Dropping {len(nan_cols)} columns due to NaNs: {nan_cols}")
        X = X.drop(columns=nan_cols)
    print(f"Feature count after dropping NaNs/infs: {X.shape[1]}")

    # Step 1: Variance filtering
    X_var = variance_filter(X, quantile=VARIANCE_QUANTILE, plot=True)
    print(f"After variance filtering: {X_var.shape[1]} features")

    # Step 1a: Drop constant columns
    zero_var_cols = X_var.columns[X_var.nunique() <= 1].tolist()
    if zero_var_cols:
        print(f"Dropping {len(zero_var_cols)} constant columns: {zero_var_cols}")
        X_var = X_var.drop(columns=zero_var_cols)
    print(f"Feature count after dropping constant columns: {X_var.shape[1]}")

    # Step 2: VIF filtering
    X_vif = drop_high_vif(X_var, threshold=VIF_THRESHOLD)
    print(f"After VIF filtering: {X_vif.shape[1]} features")

    # Save selected features
    selected_features = list(X_vif.columns)
    os.makedirs(os.path.dirname(OUTPUT_FEATURES_JSON), exist_ok=True)
    with open(OUTPUT_FEATURES_JSON, "w") as f:
        json.dump(selected_features, f, indent=4)

    print(f"Selected features saved to: {OUTPUT_FEATURES_JSON}")
