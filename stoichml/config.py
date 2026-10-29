"""
config.py
─────────
Shared constants for StoichML training, feature-selection, and SHAP scripts.

Scope (intentionally narrow): only constants that are (a) genuinely
duplicated across multiple scripts and (b) verified to hold the same
value everywhere they currently appear. See the audit note at the
bottom of this file for constants that were considered and deliberately
NOT centralized here.

Changing any value in this file changes training/feature-selection
behavior across every script that imports it. Do not edit these values
as part of a routine refactor — they encode methodology decisions
(paper Table 5, feature_selection.py docstring, etc.).
"""

# Global reproducibility seed. Used for CV shuffling, LGBM/XGB
# random_state, and any other stochastic step across every task.
RANDOM_STATE: int = 42

# Number of cross-validation folds. Referred to as N_FOLDS in
# stoichml/feature_selection.py and N_SPLITS elsewhere; the value has
# always been 5 in every script.
N_SPLITS: int = 5

# Seeds used for the 5-seed undersampling ensembles (egap_type binary
# classifier, hm_class one-shot baseline and binary detector, and the
# corresponding feature-selection importance averaging).
UNDERSAMPLE_SEEDS: list[int] = [0, 7, 21, 42, 99]

# Cumulative-gain-importance cutoff (tau) used by LightGBM-based feature
# selection (stoichml/feature_selection.py) and by the standalone
# expt_gap pipeline (scripts/model_expt.py, previously local `TAU`).
CUM_IMPORTANCE: float = 0.85

# Cap on majority-class sample count for hm_class's capped undersampling
# (feature_selection.py and scripts/model_hm.py's one-shot baseline).
MAJORITY_CAP: int = 3000
