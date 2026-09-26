# Analyses performed manually / outside this repository

This repository does not contain code for every analysis reported in the
paper. The following were performed manually, outside version control, and
**cannot currently be regenerated from `scripts/`**:

- **`|r| > 0.8` correlated-feature grouping** — pairwise Spearman correlation
  among each task's selected features, thresholded and grouped to identify
  redundant feature clusters.
- **Composite-feature ablations** — retraining with correlated-feature
  groups collapsed into composite descriptors, to test whether SHAP
  attribution to an individual feature reflects a group-level signal.
- **Top-k SHAP ablations** *(beyond what `scripts/topk.py` covers — see
  below)* — retraining on progressively larger top-k SHAP feature subsets
  as a robustness/sufficiency check.

## What is reproducible from this repository

- **`scripts/shap_spearman.py`** computes the pairwise Spearman correlation
  matrix among each task's selected features and reports mean |SHAP| plus
  correlated-partner groupings for a chosen top-N feature list, using a
  configurable `|r|` threshold (`--threshold`, default 0.8). This is the
  reproducible code behind the correlation analysis described above — it
  computes the correlation structure and grouping; it does **not** perform
  the composite-feature ablation (retraining with collapsed groups).
- **`scripts/topk.py`** runs cross-validated evaluation for models trained
  on the top-`k` SHAP features (`k` in a fixed list) per task, using SHAP
  importance CSVs from `shap_outputs/`. This is a top-k ablation, but its
  `k` range and exact configuration should be checked against what is
  reported in the paper before treating it as the source of any specific
  top-k ablation number in the manuscript — it may or may not be the exact
  study referenced there.

## Why this matters for reproducibility

A researcher cloning this repository and running every script in
`README.md`'s "Reproducing the paper results" section will regenerate the
five trained models, their metrics, and the standard SHAP outputs — but
**not** the composite-descriptor ablation results or the correlated-feature
grouping used to motivate them. Those numbers in the paper should be
treated as manually derived until (if ever) scripted and added here.

This file should be updated if any of the above is later implemented as a
script, or if the exact manual procedure (e.g. which composite groups were
formed, exact k values used) is written down elsewhere and can be linked
from here.
