import os
import json
import pandas as pd

from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split

from stoichml.utils import load_and_featurize


# -------------------- Paths --------------------
RAW_PATH = "data/data.pkl"
FEAT_PATH = "data/data_feat.pkl"
SELECTED_FEATURES_PATH = "data/selected_features.json"

TARGET_COL = "enthalpy_formation_atom"
#hm_class, Egap, Egap_type_numeric

# -------------------- I/O --------------------
def load_selected_features(path: str) -> list[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Selected feature file not found: {path}")

    with open(path, "r") as f:
        feats = json.load(f)

    if not isinstance(feats, list):
        raise ValueError("selected_features.json must be a list of column names")

    if len(feats) == 0:
        raise ValueError("selected_features.json is empty")

    return feats


def load_or_featurize() -> pd.DataFrame:
    if os.path.exists(FEAT_PATH):
        print(f"Loading featurized data: {FEAT_PATH}")
        return pd.read_pickle(FEAT_PATH)

    print("Featurized data not found. Running featurization...")

    df_feat = load_and_featurize(
        path=RAW_PATH,
        elements_col="elements",
        composition_col="composition",
        verbose=True
    )

    df_feat.to_pickle(FEAT_PATH)
    print(f"Saved featurized data: {FEAT_PATH}")

    return df_feat


# -------------------- Main --------------------
def main():
    df = load_or_featurize()

    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column missing: {TARGET_COL}")

    selected_features = load_selected_features()

    missing = [c for c in selected_features if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{len(missing)} selected features not found in dataframe:\n{missing}"
        )

    X = df[selected_features]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    reg = LazyRegressor(
        verbose=1,
        ignore_warnings=True,
        predictions=False
    )

    models, _ = reg.fit(X_train, X_test, y_train, y_test)

    print("\n=== LazyRegressor Benchmark Results ===\n")
    print(models.sort_values("R-Squared", ascending=False))


# -------------------- Entry --------------------
if __name__ == "__main__":
    main()
