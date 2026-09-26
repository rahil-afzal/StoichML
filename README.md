# StoichML

> **Stoichiometry driven machine learning for inorganic materials property prediction; no crystal structure required.**

StoichML predicts five materials properties directly from chemical composition using physics-informed elemental descriptors and gradient-boosted tree ensembles. It is designed for researchers who want interpretable, fast, and scalable screening of large composition spaces without DFT geometry as a prerequisite.

**Built on:** [AFLOW](http://aflow.org) + [supercon2018](https://github.com/vstanev1/Supercon) | **Models:** LightGBM + XGBoost | **Interpretability:** SHAP

---

## Motivation

Most ML models for materials properties require a relaxed crystal structure, limiting their use to compounds already computed by DFT. StoichML takes the opposite approach: given only a chemical formula, it simultaneously predicts formation enthalpy, band gap, metal/insulator character, half-metallic behaviour, and superconducting critical temperature.

The primary scientific contribution of StoichML is not performance benchmarking but **quantification of the composition-only prediction ceiling**  a systematic characterisation of what stoichiometric features can and cannot predict, and why. SHAP attribution analysis across all five tasks recovers known physical mechanisms from composition alone without supervision, providing an interpretable map of composition-property relationships in inorganic materials space.

---

## Tasks

| Task | Target | Type | Dataset | N | Selected features |
|---|---|---|---|---|---|
| `enthalpy` | Formation enthalpy (eV/atom) | Regression | AFLOW | 32,990 | 20 |
| `egap` | Band gap (eV), filtered to `Egap > 0.1 eV` | Regression | AFLOW | 6,916 | 64 |
| `egap_type` | Metal vs. insulator | Binary classification | AFLOW | 32,990 | 18 |
| `hm_class` | Conductor / Insulator / Half-metal | Ternary classification | AFLOW | 32,990 | 52 |
| `supercon` | Critical temperature Tc (K), filtered to `Tc > 0.1 K` | Regression | supercon2018 | 12,288 | 52 |

AFLOW N = 32,990 reflects AFLOWLIB compounds filtered to `natoms > 1` (excludes elemental solids, which have zero formation enthalpy by definition). Values verified against `data/selected_features.json`, the paper's Table 1, and the per-task metrics JSONs.

---

## Key Results

### Regression

| Task | Model | MAE | RMSE | R² |
|---|---|---|---|---|
| Enthalpy | Ensemble | 0.107 eV/atom | 0.175 eV/atom | 0.957 |
| Band gap | Ensemble (deep, `Egap > 0.1 eV`) | 0.474 eV | 0.695 eV | 0.831 |
| Tc | Ensemble | 3.796 K | 7.588 K | 0.930 |

Band gap R² ≈ 0.83 is a **composition-only plateau** — six architecturally different models (SVR, LGBM, XGBoost, standard and deep hyperparameter sets, ensemble) converge to the same range (0.827–0.835) with identical signed-residual bias patterns, and an independently refit experimental band-gap dataset lands in the same range, indicating the missing information is structural (crystal symmetry, coordination geometry) rather than elemental. For Tc, global R² = 0.930 but within-bin R² = −2.26 (mean, across the six Tc bins) meaning the model discriminates between compound families but does not predict Tc variation within a family.

### Classification

| Task | ROC-AUC | Macro F1 | Balanced Acc |
|---|---|---|---|
| Metal/insulator | 0.994 | 0.957 | 0.971 |
| Half-metal (one-shot, 3-class) | — | 0.787 | 0.825 |
| Half-metal (binary detector, end-to-end 3-class, Youden-J) | — | 0.647 | 0.837 |

Half-metal binary-detector standalone performance: ROC-AUC = 0.929, PR-AUC = 0.336.

### Half-metal precision-recall comparison

One-shot ternary softmax vs. the binary half-metal detector at three operating points (end-to-end three-class reconstruction; P/R/F1 are for the half-metal class, MF1 is macro-F1 over all three classes):

| Model | Threshold | Precision | Recall | F1 (HM) | Macro F1 |
|---|---|---|---|---|---|
| One-shot (softmax, argmax) | — | 0.392 | 0.548 | 0.457 | **0.787** |
| Binary detector — Youden-J | 0.481 | 0.087 | **0.873** | 0.158 | 0.647 |
| Binary detector — F1-optimal | 0.939 | 0.377 | 0.400 | 0.388 | 0.781 |
| Binary detector — precision-matched | 0.945 | 0.392 | 0.363 | 0.377 | 0.777 |

**The binary detector does not outperform the one-shot baseline on macro-F1 or half-metal F1 at any threshold** — the one-shot classifier's fixed operating point (P=0.392, R=0.548) already has the best macro-F1 (0.787) of the four. The detector's value is that it exposes a *tunable* precision–recall frontier the one-shot softmax cannot reach: at Youden-J, recall rises to 0.873 (a 59% relative gain over the one-shot's 0.548) at the cost of precision falling to 0.087, which suits high-throughput screening where missing a half-metal is costlier than chasing a false positive. This tradeoff, not an accuracy improvement, is the reported contribution.

### SHAP cross-task attribution

Each task is governed by a different physical descriptor family, recovered without supervision:

| Task | Dominant descriptor | Physical mechanism |
|---|---|---|
| Enthalpy | `chi_mad` (electronegativity MAD) | Ionic bonding stability |
| Band gap | `period_hmean` (period number) | Orbital size and bandwidth |
| Metal/insulator | `tm_frac` (TM fraction) | d-band occupation |
| Half-metal | `Z_gmean` (atomic number) | Heavy TM identity |
| Superconductivity | `volume_mad` (atomic volume MAD) | Lattice strain, phonon softening |

---

## Feature Engineering

### Elemental property database

22 elemental properties retrieved from `mendeleev`, supplemented with a literature patch table for 74 elements. Properties include:

| Group | Properties |
|---|---|
| Identity | Z, atomic mass |
| Electronic | χ (Pauling), I₁, EA, η = (I₁−EA)/2, valence count, vacancies, d-count (valence shell only), p-count, \|d−5\|, unpaired electrons |
| Structural | covalent radius, atomic volume, dipole polarisability |
| Thermodynamic | Tm, Tb, κ, Ecoh |
| Magnetic | solid-state magnetic moment |
| New in this work | work function φ, period number |

### Statistical aggregation

For each property, 8 weighted statistics are computed using stoichiometric fractions as weights: `mean`, `std`, `min`, `max`, `mad`, `pos` = (mean−min)/(max−min), `hmean` (harmonic), `gmean` (geometric). This yields **22 × 8 = 176** elemental statistics.

### Composition-level physics features

13 additional features encode composition-level structure not reducible to single-element statistics: stoichiometric complexity (n_elements, n_atoms, max_weight, conf_entropy), electronegativity mismatch (delta_chi, pair_chi), orbital character (S_orb), magnetic character (S_mag, tm_frac, f_frac, unpaired_var), and structural mismatch (r_mad, val_var).

*(`stoichml/featurizer.py`'s `phys()` function internally computes 18 composition-level quantities, but 5 of them — `chi_mad`, `mass_std`, `val_mean`, `dhalf_mean`, `unpaired_mean` — share their name with, and are numerically identical to, an elemental statistic already produced above, so they overwrite rather than add a column. Net unique contribution is 13, not 18; see the corrected docstring in `featurize()` for detail.)*

**Total: 176 + 13 = 189 features.**

### Feature selection

Task-specific feature subsets are selected by LightGBM cumulative gain importance (threshold τ = 0.85, 5-fold CV). For imbalanced tasks (egap_type, hm_class), importances are averaged across 5 folds × 5 seeds = 25 undersampled training runs. Physics feature force-retention is **disabled** — all features survive on predictive merit alone.

---

## Architecture

### Regression tasks (enthalpy, egap, supercon)

- LightGBM + XGBoost, 5-fold KFold CV with early stopping (patience = 200)
- Final model n_estimators set to mean best iteration across CV folds
- Ensemble = (LGBM prediction + XGB prediction) / 2

### Binary classification (egap_type)

- 5-seed LGBM ensemble, strict 1:1 undersampling per seed
- Youden-J threshold tuned on OOF ensemble probabilities

### Ternary classification (hm_class) — two-stage

**Stage 1 — half-metal detector (binary):**
- 5-seed LGBM + XGBoost ensemble, strict 1:1 undersampling (482 half-metals)
- Three thresholds evaluated from OOF probabilities:
  - Youden-J (balanced sensitivity/specificity)
  - F1-optimal (maximises half-metal F1)
  - Precision-matched (matches one-shot baseline precision)

**Stage 2 — conductor/insulator (binary):**
- Standard 5-fold CV LGBM + XGBoost on non-half-metal samples
- No undersampling needed (ratio ≈ 6:1, manageable)

**Inference:** if stage 1 probability ≥ threshold → class 2 (half-metal); else → stage 2 prediction (class 0 or 1).

---

## Project Structure

```
StoichML/
│
├── stoichml/
│   ├── __init__.py
│   ├── featurizer.py              # vec(), stats(), phys() — 189 features (see docstring)
│   ├── feature_selection.py       # cumulative gain importance pruning
│   ├── utils.py                   # load and featurize the dataset
│   └── predict.py                 # (currently empty — inference helpers live in
│                                   #  scripts/model_train.py; see reproducibility gaps)
│
├── scripts/
│   ├── run_featurize.py           # featurize AFLOW and supercon datasets --task {aflow,supercon}
│   ├── run_benchmark.py           # exploratory LazyPredict screen — --task required, no "all"
│   ├── model_train.py             # enthalpy + egap_type training
│   ├── model_egap.py              # band gap full benchmark (6 models)
│   ├── model_supercon.py          # Tc regression + variance decomposition
│   ├── model_hm.py                # half-metal: one-shot vs two-stage binary detector
│   ├── model_expt.py              # independent replication on the Zhuo et al. experimental
│   │                               #  band-gap dataset (separate from the AFLOW egap task)
│   ├── shap.py                    # unified SHAP for enthalpy/egap/egap_type/hm_class
│   ├── shap_supercon.py           # SHAP for superconductivity (incl. per-Tc-bin analysis)
│   ├── shap_spearman.py           # |r|>0.8 correlated-feature grouping (see
│   │                               #  docs/MANUAL_ANALYSES.md for what this does and does not cover)
│   └── topk.py                    # top-k SHAP-feature ablation sweep
│
├── data/
│   ├── README.md                  # data provenance (added — see below)
│   ├── dataset.pkl                # raw/filtered AFLOW data, elements+composition schema
│   ├── superconductivity.pkl      # raw supercon2018 data
│   ├── data_feat.pkl              # featurized AFLOW (generated, gitignored)
│   ├── supercon_feat.pkl          # featurized supercon (generated, gitignored)
│   └── selected_features.json     # pruned feature sets per task (tracked, not generated
│                                   #  on every run — regenerate via feature_selection.py)
│
├── models/            (generated, gitignored)
│   └── {task}/
│       ├── {task}_lgbm.pkl
│       ├── {task}_xgb.pkl
│       ├── {task}_metrics.json
│       └── thresholds.json        # classification tasks
│
├── shap_outputs/      (generated, gitignored)
│   └── {task}/
│       ├── shap_summary_{task}.png
│       ├── shap_bar_{task}.png
│       ├── shap_dependence_*_{task}.png
│       └── feature_importance_{task}.csv
│
└── docs/
    └── MANUAL_ANALYSES.md         # analyses performed outside this repo (added — see below)
```

---

## Installation

```bash
git clone https://github.com/rahil-afzal/StoichML
cd StoichML

conda create -n stoichml python
conda activate stoichml

pip install -r requirements.txt
```

**Core dependencies:**

```
mendeleev        # elemental property database
numpy
pandas
scikit-learn
lightgbm
xgboost
shap
joblib
matplotlib
seaborn
pymatgen         
```

---

## Quickstart

### 1. Prepare your data

Your input DataFrame needs two columns:
- `elements` — list of element symbols, e.g. `["Fe", "O"]`
- `composition` — list of stoichiometric counts in reduced form, e.g. `[2, 3]` for Fe₂O₃

For the other dataset, parse from formula strings using pymatgen:

```python
from pymatgen.core import Composition
import pandas as pd

df = pd.read_csv("your_file.csv")
df["composition_obj"] = df["formula"].apply(Composition)
df["elements"] = df["composition_obj"].apply(lambda c: [str(el) for el in c.elements])
df["composition"] = df["composition_obj"].apply(lambda c: [c[el] for el in c.elements])
```

### 2. Featurize

```python
from stoichml.featurizer import featurize

df_feat = featurize(df, elements_col="elements", composition_col="composition")
# → original columns + 189 feature columns
```

Or via script (handles both datasets):

```bash
python -m scripts.run_featurize                   # both datasets
python -m scripts.run_featurize --task aflow
python -m scripts.run_featurize --task supercon
```

### 3. Feature selection

```bash
python -m scripts.feature_selection               # all 5 tasks
python -m scripts.feature_selection --task supercon
```

Writes `data/selected_features.json` with per-task feature lists.

### 4. Train

```bash
python -m scripts.model_train --task enthalpy
python -m scripts.model_train --task egap_type
python -m scripts.model_egap                      # 6-model band gap benchmark
python -m scripts.model_supercon                  # Tc + variance decomposition
python -m scripts.train_hm_class                  # one-shot vs two-stage
```

### 5. SHAP analysis

```bash
python -m scripts.shap --task all
python -m scripts.shap --task supercon   # includes per-bin analysis
```

---

## Reproducing the paper results

Run scripts in this order:

```bash
# Step 1 — featurize
python -m scripts.run_featurize

# Step 2 — feature selection
python -m scripts.feature_selection

# Step 3 — train all tasks
python -m scripts.model_train --task enthalpy
python -m scripts.model_train --task egap_type
python -m scripts.model_egap
python -m scripts.model_supercon --no_log
python -m scripts.train_hm

# Step 4 — SHAP
python -m scripts.shap --task all
```

All metrics are saved as JSON files in `models/{task}/`. Figures are saved in `models/{task}/images/` and `shap_outputs/{task}/`.

---

## Known Limitations

**Composition-only ceiling.** Crystal symmetry and local coordination geometry are absent. Band gap prediction plateaus at R² = 0.828 regardless of model architecture. Tc prediction within compound families has R² < 0 across all Tc ranges — the model learns inter-family discrimination but not intra-family variation.

**Half-metal class imbalance.** Half-metals are rare (482 / 32,990 ≈ 1.5%, a 67.4:1 minority imbalance). The two-stage architecture substantially improves achievable recall but precision remains limited by the rarity of the class, and does not improve macro-F1 over the one-shot baseline (see comparison table above).

**Formula unit convention.** `n_atoms` requires reduced compositions. Non-reduced inputs give inconsistent values. AFLOW data satisfies this by default; other sources should be normalised before featurisation.

**DFT target bias.** GGA-PBE systematically underestimates band gaps by ~30–50%. Models learn to reproduce DFT values; comparisons with experiment should account for this offset.

---

## Citation

If you use StoichML in your research, please cite:

```bibtex
@article{stoichml2025,
  author  = {TODO},
  title   = {What Stoichiometry Can and Cannot Predict: Composition-Only
             Machine Learning for Inorganic Materials},
  journal = {TODO},
  year    = {2026},
}
```

---

## License

MIT License. See `LICENSE` for full terms.