"""
Stage-1 refactor tests, part 2: shared utilities behave exactly like the
per-script implementations they replaced.

Each `_orig_*` function below is a verbatim copy of the implementation as
it existed in the named script BEFORE the Stage-1 refactor (see the
baseline git commit). Each test compares that original against the new
shared implementation in stoichml.utils across multiple synthetic
scenarios, seeds, and class distributions. Numeric equality is exact
(not approximate) for the sampling functions, since both draw from the
same numpy Generator API in the same call order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, StratifiedKFold

from stoichml.utils import (
    bin_analysis,
    make_kfold,
    make_stratified_kfold,
    regression_metrics,
    save_json,
    undersample_to_minority,
)

RANDOM_STATE = 42
N_SPLITS = 5


# ══════════════════════════════════════════════════════════════════════════════
# Reference copies of the pre-refactor implementations
# ══════════════════════════════════════════════════════════════════════════════

def _orig_undersample_to_minority_feature_selection(X, y, minority_class, seed, majority_cap=None):
    """Verbatim from stoichml/feature_selection.py (pre-refactor)."""
    rng = np.random.default_rng(seed)
    idx_minority = y[y == minority_class].index
    n_minority = len(idx_minority)
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


def _orig_undersample_strict_model_train(X, y, minority_class, seed):
    """Verbatim from scripts/model_train.py (pre-refactor)."""
    rng = np.random.default_rng(seed)
    idx_minority = y[y == minority_class].index
    n_minority = len(idx_minority)
    balanced_idx = list(idx_minority)
    for cls in sorted(y.unique()):
        if cls == minority_class:
            continue
        idx_cls = y[y == cls].index
        n_sample = min(n_minority, len(idx_cls))
        balanced_idx.extend(rng.choice(idx_cls, size=n_sample, replace=False))
    balanced_idx = np.array(balanced_idx)
    rng.shuffle(balanced_idx)
    return X.loc[balanced_idx], y.loc[balanced_idx]


def _orig_undersample_strict_topk(X, y, minority_class, seed):
    """Verbatim from scripts/topk.py (pre-refactor)."""
    rng = np.random.default_rng(seed)
    idx_min = list(y[y == minority_class].index)
    n_min = len(idx_min)
    idx_all = list(idx_min)
    for cls in sorted(y.unique()):
        if cls == minority_class:
            continue
        idx_cls = y[y == cls].index
        chosen = rng.choice(idx_cls, size=min(n_min, len(idx_cls)), replace=False)
        idx_all.extend(chosen)
    idx_all = np.array(idx_all)
    rng.shuffle(idx_all)
    return X.loc[idx_all], y.loc[idx_all]


def _orig_capped_model_hm(X, y, minority_class, cap, seed):
    """Verbatim from scripts/model_hm.py (pre-refactor)."""
    rng = np.random.default_rng(seed)
    idx = list(y[y == minority_class].index)
    for cls in sorted(y.unique()):
        if cls == minority_class:
            continue
        idx_cls = y[y == cls].index
        idx.extend(rng.choice(idx_cls, size=min(cap, len(idx_cls)), replace=False))
    rng.shuffle(idx)
    return X.loc[idx], y.loc[idx]


def _orig_strict_11_model_hm(X, y, minority_class, seed):
    """
    Verbatim from scripts/model_hm.py (pre-refactor). Deliberately NOT
    consolidated — kept here only to document/verify the 2-class
    equivalence claimed in scripts/model_hm.py's docstring.
    """
    rng = np.random.default_rng(seed)
    n_minority = (y == minority_class).sum()
    idx = []
    for cls in sorted(y.unique()):
        idx_cls = y[y == cls].index
        n_sample = n_minority if cls != minority_class else len(idx_cls)
        chosen = (rng.choice(idx_cls, size=n_sample, replace=False)
                  if n_sample < len(idx_cls) else idx_cls)
        idx.extend(chosen)
    rng.shuffle(idx)
    return X.loc[idx], y.loc[idx]


def _orig_compute_metrics(y_true, y_pred):
    """Verbatim from scripts/model_egap.py / model_supercon.py (pre-refactor)."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def _orig_regression_metrics_model_train(y_true, y_pred):
    """Verbatim from scripts/model_train.py (pre-refactor, different key order)."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def _orig_bin_analysis(y_true, y_pred, bins, bin_labels):
    """Verbatim from scripts/model_egap.py / model_supercon.py (pre-refactor)."""
    from sklearn.metrics import r2_score
    errors = y_pred - y_true
    bin_idx = np.clip(np.digitize(y_true, bins) - 1, 0, len(bin_labels) - 1)
    stats = []
    for i, label in enumerate(bin_labels):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        n = int(mask.sum())
        e = errors[mask]
        r2b = float(r2_score(y_true[mask], y_pred[mask])) if n > 1 else float("nan")
        stats.append({
            "bin": label, "n": n, "mean_err": float(e.mean()),
            "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "r2": r2b,
        })
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic scenarios
# ══════════════════════════════════════════════════════════════════════════════

def _make_binary_frame(n=400, minority_frac=0.1, n_features=6, seed=0):
    rng = np.random.default_rng(seed)
    n_minority = max(2, int(n * minority_frac))
    n_majority = n - n_minority
    y = pd.Series([1] * n_minority + [0] * n_majority)
    X = pd.DataFrame(rng.normal(size=(n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    return X, y


def _make_ternary_frame(n=900, class_fracs=(0.85, 0.13, 0.02), n_features=6, seed=0):
    rng = np.random.default_rng(seed)
    counts = [max(2, int(n * f)) for f in class_fracs]
    y = pd.Series([0] * counts[0] + [1] * counts[1] + [2] * counts[2])
    X = pd.DataFrame(rng.normal(size=(len(y), n_features)), columns=[f"f{i}" for i in range(n_features)])
    return X, y


SEEDS = [0, 7, 21, 42, 99, 123]


# ══════════════════════════════════════════════════════════════════════════════
# Undersampling equivalence
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("seed", SEEDS)
def test_undersample_matches_feature_selection_strict(seed):
    X, y = _make_binary_frame(seed=1)
    Xa, ya = undersample_to_minority(X, y, minority_class=1, seed=seed, majority_cap=None)
    Xb, yb = _orig_undersample_to_minority_feature_selection(X, y, minority_class=1, seed=seed, majority_cap=None)
    pd.testing.assert_frame_equal(Xa, Xb)
    pd.testing.assert_series_equal(ya, yb)


@pytest.mark.parametrize("seed", SEEDS)
def test_undersample_matches_feature_selection_capped(seed):
    X, y = _make_ternary_frame(seed=2)
    cap = 200
    Xa, ya = undersample_to_minority(X, y, minority_class=2, seed=seed, majority_cap=cap)
    Xb, yb = _orig_undersample_to_minority_feature_selection(X, y, minority_class=2, seed=seed, majority_cap=cap)
    pd.testing.assert_frame_equal(Xa, Xb)
    pd.testing.assert_series_equal(ya, yb)


@pytest.mark.parametrize("seed", SEEDS)
def test_undersample_matches_model_train_undersample_strict(seed):
    X, y = _make_binary_frame(seed=3)
    Xa, ya = undersample_to_minority(X, y, minority_class=1, seed=seed, majority_cap=None)
    Xb, yb = _orig_undersample_strict_model_train(X, y, minority_class=1, seed=seed)
    pd.testing.assert_frame_equal(Xa, Xb)
    pd.testing.assert_series_equal(ya, yb)


@pytest.mark.parametrize("seed", SEEDS)
def test_undersample_matches_topk_undersample_strict(seed):
    X, y = _make_binary_frame(seed=4)
    Xa, ya = undersample_to_minority(X, y, minority_class=1, seed=seed, majority_cap=None)
    Xb, yb = _orig_undersample_strict_topk(X, y, minority_class=1, seed=seed)
    pd.testing.assert_frame_equal(Xa, Xb)
    pd.testing.assert_series_equal(ya, yb)


@pytest.mark.parametrize("seed", SEEDS)
def test_undersample_matches_model_hm_capped(seed):
    X, y = _make_ternary_frame(seed=5)
    cap = 3000
    Xa, ya = undersample_to_minority(X, y, minority_class=2, seed=seed, majority_cap=cap)
    Xb, yb = _orig_capped_model_hm(X, y, minority_class=2, cap=cap, seed=seed)
    pd.testing.assert_frame_equal(Xa, Xb)
    pd.testing.assert_series_equal(ya, yb)


@pytest.mark.parametrize("seed", SEEDS)
def test_model_hm_strict_11_selects_same_rows_as_shared_util_but_different_order(seed):
    """
    scripts/model_hm.py's `strict_11` was left untouched rather than merged
    into the shared undersample_to_minority. This test documents *why*:
    for its actual use case (binary y in {0, 1}), `strict_11` samples the
    same SET of rows as undersample_to_minority(majority_cap=None) for the
    same seed, but returns them in a different ROW ORDER. This is because
    `strict_11` builds its pre-shuffle list as [majority_sample, minority]
    while undersample_to_minority builds [minority, majority_sample] —
    same rng.choice() draw, but shuffling a differently-ordered list with
    the same random draws yields a different final permutation.

    Net effect: the balanced training set has identical content but is
    fed to LightGBM/XGBoost in a different row order. This is a
    pre-existing property of the original code, not something introduced
    by this refactor. If `strict_11` is ever merged into the shared
    utility, note that doing so silently changes row order (not sample
    content) for scripts/model_hm.py's binary detector and one-shot
    baseline, which could shift training results at the floating-point
    level and break bit-for-bit reproducibility of previously saved runs.
    """
    X, y = _make_binary_frame(seed=6)
    Xa, ya = undersample_to_minority(X, y, minority_class=1, seed=seed, majority_cap=None)
    Xb, yb = _orig_strict_11_model_hm(X, y, minority_class=1, seed=seed)

    assert set(Xa.index) == set(Xb.index), "row SET must match (same sample content)"
    assert list(Xa.index) != list(Xb.index), (
        "row ORDER is expected to differ (see docstring) — if this ever "
        "starts passing, the two implementations' shuffle behavior has "
        "changed and the non-consolidation rationale should be re-checked"
    )
    # Same rows, sorted the same way, contain identical data either way:
    pd.testing.assert_frame_equal(Xa.sort_index(), Xb.sort_index())
    pd.testing.assert_series_equal(ya.sort_index(), yb.sort_index())


# ══════════════════════════════════════════════════════════════════════════════
# Metrics equivalence
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("seed", SEEDS)
def test_regression_metrics_matches_compute_metrics(seed):
    rng = np.random.default_rng(seed)
    y_true = rng.normal(size=300)
    y_pred = y_true + rng.normal(scale=0.3, size=300)
    assert regression_metrics(y_true, y_pred) == _orig_compute_metrics(y_true, y_pred)


@pytest.mark.parametrize("seed", SEEDS)
def test_regression_metrics_matches_model_train_regression_metrics(seed):
    rng = np.random.default_rng(seed)
    y_true = rng.normal(size=300)
    y_pred = y_true + rng.normal(scale=0.3, size=300)
    new = regression_metrics(y_true, y_pred)
    old = _orig_regression_metrics_model_train(y_true, y_pred)
    # model_train.py's dict had a different key order; compare as sets of items.
    assert new == old


# ══════════════════════════════════════════════════════════════════════════════
# Bin analysis equivalence
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("seed", SEEDS)
def test_bin_analysis_matches_original_egap_bins(seed):
    rng = np.random.default_rng(seed)
    y_true = rng.uniform(0, 9.5, size=2000)
    y_pred = y_true + rng.normal(scale=0.4, size=2000)
    bins = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 9.5]
    bin_labels = ["0–0.5", "0.5–1", "1–2", "2–4", "4–6", "6–9.5"]
    assert bin_analysis(y_true, y_pred, bins, bin_labels) == _orig_bin_analysis(y_true, y_pred, bins, bin_labels)


@pytest.mark.parametrize("seed", SEEDS)
def test_bin_analysis_matches_original_supercon_bins(seed):
    rng = np.random.default_rng(seed)
    y_true = rng.uniform(0, 200, size=2000)
    y_pred = y_true + rng.normal(scale=5, size=2000)
    bins = [0, 10, 20, 40, 77, 100, 200]
    bin_labels = ["0–10", "10–20", "20–40", "40–77", "77–100", "100+"]
    assert bin_analysis(y_true, y_pred, bins, bin_labels) == _orig_bin_analysis(y_true, y_pred, bins, bin_labels)


# ══════════════════════════════════════════════════════════════════════════════
# CV construction equivalence
# ══════════════════════════════════════════════════════════════════════════════

def test_make_kfold_matches_inline_construction():
    X = np.arange(100).reshape(-1, 1)
    a = list(make_kfold().split(X))
    b = list(KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE).split(X))
    assert len(a) == len(b)
    for (tr_a, va_a), (tr_b, va_b) in zip(a, b):
        assert np.array_equal(tr_a, tr_b)
        assert np.array_equal(va_a, va_b)


def test_make_stratified_kfold_matches_inline_construction():
    X, y = _make_binary_frame(n=200, seed=9)
    a = list(make_stratified_kfold().split(X, y))
    b = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE).split(X, y))
    assert len(a) == len(b)
    for (tr_a, va_a), (tr_b, va_b) in zip(a, b):
        assert np.array_equal(tr_a, tr_b)
        assert np.array_equal(va_a, va_b)


# ══════════════════════════════════════════════════════════════════════════════
# save_json: preserves each script's prior indent width
# ══════════════════════════════════════════════════════════════════════════════

def test_save_json_default_indent_matches_model_train_prior_default(tmp_path):
    payload = {"a": 1, "b": {"c": 2}}
    path = tmp_path / "out.json"
    save_json(payload, str(path))  # default indent=4, matches model_train.py's prior default
    text = path.read_text()
    assert text == __import__("json").dumps(payload, indent=4)


def test_save_json_indent2_matches_feature_selection_prior_output(tmp_path):
    payload = {"enthalpy": ["a", "b"], "egap": ["c"]}
    path = tmp_path / "out.json"
    save_json(payload, str(path), indent=2)  # feature_selection.py used indent=2
    text = path.read_text()
    assert text == __import__("json").dumps(payload, indent=2)
