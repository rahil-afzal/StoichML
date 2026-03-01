import os
import json
import argparse
import numpy as np
import pandas as pd
from lazypredict.Supervised import LazyRegressor, LazyClassifier
from sklearn.model_selection import train_test_split
from stoichml.utils import load_and_featurize

# -------------------- Paths --------------------
RAW_PATH = "data/data.pkl"
FEAT_PATH = "data/data_feat.pkl"
SELECTED_FEATURES_PATH = "data/selected_features.json"

# -------------------- TASK registry --------------------
TASKS = {
    "enthalpy": {
        "target": "enthalpy_formation_atom",
        "type": "regression",
        "transform": None,
    },
    "egap": {
        "target": "Egap",
        "type": "regression",
        "transform": None,          # No log1p — consistent with train.py
        "filter": ("Egap", ">", 0), # Hurdle model: insulators only
    },
    "egap_type": {
        "target": "Egap_type_numeric",
        "type": "binary",
    },
    "hm_class": {
        "target": "hm_class",
        "type": "multiclass",
    },
}

# -------------------- I/O --------------------
def load_selected_features(task: str) -> list[str]:
    if not os.path.exists(SELECTED_FEATURES_PATH):
        raise FileNotFoundError("selected_features.json not found")
    with open(SELECTED_FEATURES_PATH, "r") as f:
        feats = json.load(f)
    if not isinstance(feats, dict):
        raise ValueError("selected_features.json must be a dict with task keys")
    core = feats.get("core", [])
    task_feats = feats.get(task, [])
    selected = list(dict.fromkeys(core + task_feats))
    if not selected:
        raise ValueError(f"No features found for task: {task}")
    return selected


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


def apply_filter(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply optional row-level filter from task config before target extraction."""
    if "filter" not in cfg:
        return df
    col, op, val = cfg["filter"]
    ops = {
        ">":  lambda s: s >  val,
        ">=": lambda s: s >= val,
        "<":  lambda s: s <  val,
        "<=": lambda s: s <= val,
        "==": lambda s: s == val,
        "!=": lambda s: s != val,
    }
    mask = ops[op](df[col])
    print(f"  Filter: {col} {op} {val}  →  {mask.sum()} / {len(df)} rows retained "
          f"({(~mask).sum()} dropped)")
    return df.loc[mask].reset_index(drop=True)


def transform_target(y, mode):
    if mode == "log1p":
        return np.log1p(y.clip(lower=0))
    return y


# -------------------- Main --------------------
def main(task_name: str):
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}")
    cfg = TASKS[task_name]

    print(f"\n=== Running Lazy Benchmark for: {task_name} ===")

    df = load_or_featurize()

    # Apply row filter (e.g. Egap > 0 for the egap hurdle model)
    df = apply_filter(df, cfg)

    if cfg["target"] not in df.columns:
        raise KeyError(f"Target column missing: {cfg['target']}")

    selected_features = load_selected_features(task_name)
    missing = [c for c in selected_features if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{len(missing)} selected features missing:\n{missing}"
        )

    print(f"  Samples after filter: {len(df)}")

    X = df[selected_features]
    y = transform_target(df[cfg["target"]], cfg.get("transform"))

    stratify = y if cfg["type"] != "regression" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )

    # -------------------- Model selection --------------------
    if cfg["type"] == "regression":
        model = LazyRegressor(
            verbose=1,
            ignore_warnings=True,
            predictions=False
        )
        models, _ = model.fit(X_train, X_test, y_train, y_test)
        print("\n=== LazyRegressor Results ===\n")
        print(models.sort_values("R-Squared", ascending=False))
    else:
        model = LazyClassifier(
            verbose=1,
            ignore_warnings=True,
            predictions=False
        )
        models, _ = model.fit(X_train, X_test, y_train, y_test)
        print("\n=== LazyClassifier Results ===\n")
        print(models.sort_values("Accuracy", ascending=False))


# -------------------- Entry --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASKS.keys())
    args = parser.parse_args()
    main(args.task)