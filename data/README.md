# Data provenance

This documents where each dataset in `data/` comes from and how it was
processed, distinguishing what is **established** (stated in the paper
and/or implemented in this repository) from what is **not recoverable
from this repository** (steps that were performed but whose code is not
checked in). Nothing below is inferred or guessed beyond what the paper
and the code actually state.

## AFLOW pipeline (enthalpy, egap, egap_type, hm_class)

```
Raw AFLOWLIB query (JSON)
        │
        │   [NOT IN THIS REPO — see gap below]
        ▼
Filtered to natoms > 1  →  N = 32,990 compounds
        │
        ▼
data/dataset.pkl              (elements, composition, + target columns)
        │   scripts/run_featurize.py  →  stoichml.featurizer.featurize()
        ▼
data/data_feat.pkl            (dataset.pkl columns + 189 feature columns)
        │   stoichml/feature_selection.py
        ▼
data/selected_features.json   (per-task feature subsets, Table 1)
```

- **Filter criterion (established, from the paper, §2.1):** AFLOWLIB compounds are filtered to `natoms > 1`, excluding elemental solids (zero formation enthalpy by definition). This gives N = 32,990 at the GGA-PBE level.
- **Gap:** the script/notebook that (a) queries or loads the raw AFLOWLIB data, (b) applies the `natoms > 1` filter, and (c) writes the result as `data/dataset.pkl` in the `elements` / `composition` schema `stoichml/utils.py` expects, **is not present in this repository**. `data/dataset.pkl` is the earliest artifact currently checked in for this pipeline. If this step exists (e.g. as a Colab notebook), it should be added here, or this file should be updated to link to it.
- Per-task row counts downstream of `data_feat.pkl` differ by task-specific target filters applied in the training scripts themselves (fully reproducible from code):
  - `egap`: `Egap > 0.1 eV` in `scripts/model_egap.py` / `stoichml/feature_selection.py` → N = 6,916.
  - `egap_type`, `hm_class`, `enthalpy`: no additional row filter; N = 32,990.

### Expected schema (`data/dataset.pkl` → input to `featurize()`)

| Column | Type | Description |
|---|---|---|
| `elements` | `list[str]` | Element symbols, e.g. `['Fe', 'O']` |
| `composition` | `list[float]` | Stoichiometric counts in **reduced form**, aligned with `elements`, e.g. `[2, 3]` for Fe₂O₃ |
| target columns | — | `enthalpy_formation_atom`, `Egap`, `Egap_type` (→ `Egap_type_numeric`, `hm_class` derived in `stoichml/utils.py`) |
| `compound`, `spacegroup_relax` | — | Carried through as metadata; explicitly excluded from the feature matrix (`NON_FEATURE_COLS` in `stoichml/feature_selection.py` / `scripts/model_train.py`) |

`n_atoms` (a composition-level feature) requires `composition` to already be in reduced form — see `README.md`'s "Known Limitations".

## SuperCon pipeline (supercon)

```
Raw SuperCon 2018 data
        │
        │   [NOT IN THIS REPO — see gap below]
        ▼
data/superconductivity.pkl    (elements, composition, Tc)
        │   scripts/run_featurize.py --task supercon
        ▼
data/supercon_feat.pkl        (+ 189 feature columns; no target filter applied here)
        │   scripts/model_supercon.py  (--tc_filter, default 0.1)
        ▼
Training set: Tc > 0.1 K  →  N = 12,288
```

- **Filter criterion (established, from the paper, §2.1):** SuperCon 2018, filtered to `Tc ≥ 0.1 K` to exclude non-superconductor negatives, giving N = 12,288.
- **Note on operator:** the paper states `Tc ≥ 0.1 K`; `scripts/model_supercon.py`'s `--tc_filter` default applies `Tc > 0.1` (strict). This is a wording/boundary discrepancy worth confirming against the actual value counts — with continuous-valued Tc it is unlikely to change N, but it has not been verified against the raw data.
- **Gap, same shape as AFLOW:** the step producing `data/superconductivity.pkl` from the original SuperCon 2018 release (source parsing, any dedup or unit normalization) is not present in this repository. Unlike the AFLOW `natoms > 1` filter, the `Tc` filter itself **is** implemented in code (`scripts/model_supercon.py`), applied at training time rather than baked into the pickle — `data/supercon_feat.pkl` itself is expected to contain rows with `Tc` below the filter as well (not verified here, per this repository's policy of not loading full data files for analysis).

## Experimental band-gap dataset (`scripts/model_expt.py`)

A third, independent dataset — the Zhuo et al. (2018) experimental band-gap set (N=6,354, via `matminer`'s `load_dataset("expt_gap")`) — is used only for the replication study described in the paper. Its full pipeline (raw fetch → `pymatgen`-based formula parsing → featurization → feature selection → training) **is** implemented end-to-end in `scripts/model_expt.py`, unlike the two pipelines above. It is a separate dataset from the AFLOW `egap` task and its numbers should never be conflated with AFLOW's.

## What this file does not do

It does not state a filtering criterion, row count, or preprocessing step that isn't traceable to either the paper text or the code in this repository. Where a step is known only from the paper (both AFLOW and SuperCon raw-to-pickle steps), that is marked as a gap above rather than reconstructed.
