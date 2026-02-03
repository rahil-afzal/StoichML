import os
import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from stoichml.utils import load_and_featurize


FEAT_PATH = "data/data_feat.pkl"
RAW_PATH  = "data/data.pkl"


def load_or_featurize():
    if os.path.exists(FEAT_PATH):
        print(f"Loading featurized data from: {FEAT_PATH}")
        return pd.read_pickle(FEAT_PATH)

    print("Featurized file not found. Running featurization...")
    df_feat = load_and_featurize(
        path=RAW_PATH,
        elements_col="elements",
        composition_col="composition",
        verbose=True
    )

    df_feat.to_pickle(FEAT_PATH)
    print(f"Saved featurized data to: {FEAT_PATH}")
    return df_feat


def main():
    df = load_or_featurize()

    # Target
    y = df["enthalpy_formation_atom"]

    # Feature columns (drop non-features)
    drop_cols = [
        "compound", "spacegroup_relax", "Egap", "Egap_type",
        "Egap_type_numeric", "enthalpy_formation_atom",
        "elements", "composition"
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Benchmark
    reg = LazyRegressor(verbose=1, ignore_warnings=True, predictions=False)
    models, _ = reg.fit(X_train, X_test, y_train, y_test)

    print("\n=== LazyRegressor Benchmark Results ===\n")
    print(models.sort_values(by="R-Squared", ascending=False))


if __name__ == "__main__":
    main()
