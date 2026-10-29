import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold

from .config import N_SPLITS, RANDOM_STATE
from .featurizer import featurize

def load_and_featurize(
    path="data/data.pkl",
    elements_col="elements",
    composition_col="composition",
    verbose=True
):
    """
    Loads a pickled DataFrame and featurizes it.

    Returns the featurized DataFrame.
    """
    if verbose:
        print(f"Loading dataset from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pickle file not found at {path}")

    df = pd.read_pickle(path)

    if verbose:
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print("Featurizing dataset...")

    df_feat = featurize(df, elements_col=elements_col, composition_col=composition_col)

    if verbose:
        print(f"Featurization complete: {df_feat.shape[0]} rows, {df_feat.shape[1]} columns")


    # Assign half-metal classification
    def hm_class(egap_type):
        if egap_type == 'metal':
            return 0
        elif egap_type == 'half-metal':
            return 2
        else:
            return 1

    df_feat['hm_class'] = df_feat['Egap_type'].apply(hm_class)

    return df_feat


def save_featurized(
    input_path="data/data.pkl",
    output_path="data/data_feat.pkl",
    elements_col="elements",
    composition_col="composition",
    verbose=True
):
    """
    Loads a dataset, featurizes it, assigns numeric labels, and saves as a pickle.

    Returns
    -------
    pd.DataFrame
        The featurized DataFrame with numeric labels.
    """
    df_feat = load_and_featurize(
        path=input_path,
        elements_col=elements_col,
        composition_col=composition_col,
        verbose=verbose
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if verbose:
        print(f"Saving featurized dataset to: {output_path}")

    df_feat.to_pickle(output_path)

    if verbose:
        print("Save complete.")

    return df_feat


# ══════════════════════════════════════════════════════════════════════════════
# Shared training/evaluation utilities
# ══════════════════════════════════════════════════════════════════════════════
# Consolidated Stage-1 duplication. Each function below previously existed,
# verbatim or near-verbatim, in multiple scripts. See the refactor report for
# the exact provenance of each one. Behavior is unchanged: every consolidation
# here was verified against the original per-script implementation on
# synthetic data before scripts were switched over to import from here.

def undersample_to_minority(
    X: pd.DataFrame,
    y: pd.Series,
    minority_class,
    seed: int,
    majority_cap: "int | None" = None,
):
    """
    Undersample all majority classes; the minority class is always kept whole.

    majority_cap=None  → each majority class is downsampled to the
                          fold-local minority size (strict 1:1 balancing).
    majority_cap=N     → each majority class is downsampled to
                          min(N, available), independent of minority size.

    This is the canonical version of what previously existed as:
      - stoichml/feature_selection.py: undersample_to_minority (identical)
      - scripts/model_train.py:        undersample_strict (majority_cap=None case)
      - scripts/topk.py:                undersample_strict (majority_cap=None case)
      - scripts/model_hm.py:            capped (majority_cap=N case)

    NOT consolidated: scripts/model_hm.py's `strict_11`, which loops over
    ALL classes (including the minority) rather than prepending the minority
    index before the loop. For its current 2-class use (y in {0,1}), testing
    (tests/test_shared_utils.py) shows it samples the same SET of rows as
    this function for the same seed, but in a DIFFERENT ROW ORDER (the
    pre-shuffle concatenation order differs, so identical shuffle draws
    produce a different permutation). Sample content is unaffected; row
    order fed to the model is not. Left untouched — see refactor report.
    """
    rng          = np.random.default_rng(seed)
    idx_minority = y[y == minority_class].index
    n_minority   = len(idx_minority)

    balanced_idx = list(idx_minority)

    for cls in sorted(y.unique()):
        if cls == minority_class:
            continue
        idx_cls = y[y == cls].index

        if majority_cap is None:
            n_sample = min(n_minority, len(idx_cls))
        else:
            n_sample = min(majority_cap, len(idx_cls))

        sampled = rng.choice(idx_cls, size=n_sample, replace=False)
        balanced_idx.extend(sampled)

    balanced_idx = np.array(balanced_idx)
    rng.shuffle(balanced_idx)

    return X.loc[balanced_idx], y.loc[balanced_idx]


def regression_metrics(y_true, y_pred) -> dict:
    """
    MAE / RMSE / R2, as a dict.

    Consolidates identical implementations previously named:
      - scripts/model_train.py:  regression_metrics
      - scripts/model_egap.py:   compute_metrics
      - scripts/model_supercon.py: compute_metrics
      - scripts/topk.py:         reg_metrics
    """
    return {
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def bin_analysis(y_true, y_pred, bins, bin_labels) -> list[dict]:
    """
    Per-bin MAE/RMSE/R2 and mean signed error, for a set of bin edges
    (`bins`, as used by np.digitize) and their `bin_labels`.

    Consolidates identical implementations previously named `bin_analysis`
    in scripts/model_egap.py (BINS in eV) and scripts/model_supercon.py
    (BINS in K). The bin edges/labels remain task-specific and are passed
    in by the caller rather than centralized here, since they encode
    scientific choices (band-gap eV bins vs. Tc K bins), not shared config.
    """
    y_true  = np.asarray(y_true)
    y_pred  = np.asarray(y_pred)
    errors  = y_pred - y_true
    bin_idx = np.clip(np.digitize(y_true, bins) - 1, 0, len(bin_labels) - 1)

    stats = []
    for i, label in enumerate(bin_labels):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        n   = int(mask.sum())
        e   = errors[mask]
        r2b = float(r2_score(y_true[mask], y_pred[mask])) if n > 1 else float("nan")
        stats.append({
            "bin":      label,
            "n":        n,
            "mean_err": float(e.mean()),
            "mae":      float(np.abs(e).mean()),
            "rmse":     float(np.sqrt((e ** 2).mean())),
            "r2":       r2b,
        })
    return stats


def make_kfold(n_splits: int = N_SPLITS, random_state: int = RANDOM_STATE) -> KFold:
    """KFold(n_splits, shuffle=True, random_state) — the construction used
    identically in every regression CV loop across the training scripts."""
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def make_stratified_kfold(
    n_splits: int = N_SPLITS, random_state: int = RANDOM_STATE
) -> StratifiedKFold:
    """StratifiedKFold(n_splits, shuffle=True, random_state) — the
    construction used identically in every classification CV loop."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def save_json(obj: dict, path: str, indent: int = 4) -> None:
    """
    Ensure the parent directory exists, then write `obj` as JSON.

    `indent` defaults to 4 (matching scripts/model_train.py's prior
    default) but can be overridden per call site to preserve a script's
    existing output formatting exactly (e.g. feature_selection.py used
    indent=2).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)
