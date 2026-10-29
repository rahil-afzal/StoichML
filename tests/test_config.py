"""
Stage-1 refactor tests, part 1: shared constants.

Asserts that every value now centralized in stoichml/config.py matches
the literal value that was hardcoded in each script before the refactor.
These are the values as they appeared in the pre-refactor source (see
the refactor report / git history), not independently chosen.
"""

from stoichml import config


def test_random_state_matches_prior_literal():
    # Was hardcoded as 42 in every script (feature_selection.py,
    # model_train.py, model_egap.py, model_supercon.py, model_hm.py,
    # topk.py, shap.py, shap_supercon.py, shap_spearman.py, model_expt.py).
    assert config.RANDOM_STATE == 42


def test_n_splits_matches_prior_literal():
    # Was N_FOLDS = 5 in feature_selection.py, N_SPLITS = 5 everywhere else.
    assert config.N_SPLITS == 5


def test_undersample_seeds_matches_prior_literal():
    # Was [0, 7, 21, 42, 99] in feature_selection.py, model_train.py,
    # shap.py, topk.py, model_hm.py.
    assert config.UNDERSAMPLE_SEEDS == [0, 7, 21, 42, 99]


def test_cum_importance_matches_prior_literal():
    # Was CUM_IMPORTANCE = 0.85 in feature_selection.py, TAU = 0.85 in
    # model_expt.py.
    assert config.CUM_IMPORTANCE == 0.85


def test_majority_cap_matches_prior_literal():
    # Was MAJORITY_CAP = 3000 in feature_selection.py and model_hm.py.
    assert config.MAJORITY_CAP == 3000
