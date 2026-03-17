# StoichML

> **Stoichiometry-driven machine learning for inorganic materials property prediction — no crystal structure required.**

StoichML predicts five materials properties directly from chemical composition using physics-informed elemental descriptors and gradient-boosted tree ensembles. It is designed for researchers who want interpretable, fast, and scalable screening of large composition spaces without DFT geometry as a prerequisite.

**Built on:** [AFLOW](http://aflow.org) + [supercon2018](https://github.com/vstanev1/Supercon) | **Models:** LightGBM + XGBoost | **Interpretability:** SHAP

---

## Motivation

Most ML models for materials properties require a relaxed crystal structure, limiting their use to compounds already computed by DFT. StoichML takes the opposite approach: given only a chemical formula, it simultaneously predicts formation enthalpy, band gap, metal/insulator character, half-metallic behaviour, and superconducting critical temperature.

The primary scientific contribution of StoichML is not performance benchmarking but **quantification of the composition-only prediction ceiling** — a systematic characterisation of what stoichiometric features can and cannot predict, and why. SHAP attribution analysis across all five tasks recovers known physical mechanisms from composition alone without supervision, providing an interpretable map of composition-property relationships in inorganic materials space.

---

## Tasks

| Task | Target | Type | Dataset | N | Selected features |
|---|---|---|---|---|---|
| `enthalpy` | Formation enthalpy (eV/atom) | Regression | AFLOW | 50,493 | 34 |
| `egap` | Band gap (eV) | Regression | AFLOW | 7,795 | 73 |
| `egap_type` | Metal vs. insulator | Binary classification | AFLOW | 47,685 | 76 |
| `hm_class` | Conductor / Insulator / Half-metal | Ternary classification | AFLOW | 50,493 | 30 |
| `supercon` | Critical temperature Tc (K) | Regression | supercon2018 | 12,288 | 52 |

---

## Key Results

### Regression

| Task | Model | MAE | RMSE | R² |
|---|---|---|---|---|
| Enthalpy | Ensemble | 0.117 eV/atom | 0.186 eV/atom | 0.943 |
| Band gap | Ensemble | 0.481 eV | 0.712 eV | 0.828 |
| Tc | Ensemble | 3.796 K | 7.588 K | 0.930 |

Band gap R² = 0.828 is a **hard composition-only ceiling** — six architecturally different models all converge to the same plateau with identical signed-residual bias patterns, confirming the missing information is structural (crystal symmetry, coordination geometry) rather than elemental. For Tc, global R² = 0.930 but within-bin R² = −2.26 across all Tc ranges, meaning the model has learned inter-family discrimination but cannot predict Tc variation within any compound family.

### Classification

| Task | ROC-AUC | Macro F1 | Balanced Acc |
|---|---|---|---|
| Metal/insulator | 0.994 | 0.928 | 0.970 |
| Half-metal (one-shot) | — | 0.749 | — |
| Half-metal (two-stage, F1-optimal) | — | **0.781** | 0.973 |

### Half-metal precision-recall comparison

| Model | Precision | Recall | F1 |
|---|---|---|---|
| One-shot softmax (baseline) | 0.263 | 0.587 | 0.363 |
| Two-stage (F1-optimal) | 0.288 | **1.000** | **0.447** |
| Two-stage (precision-matched) | 0.263 | **1.000** | 0.416 |

At matched precision (P = 0.263), the two-stage architecture achieves recall = 1.000 vs the one-shot recall of 0.587 — a +70.6% absolute improvement. The two-stage PR curve strictly dominates the one-shot fixed operating point at every precision level.

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

18 additional features encode composition-level structure not reducible to single-element statistics: stoichiometric complexity (n_elements, n_atoms, max_weight, conf_entropy), electronegativity mismatch (chi_mad, delta_chi, pair_chi), orbital character (S_orb), magnetic character (S_mag, tm_frac, f_frac, unpaired_mean, unpaired_var), and structural mismatch (r_mad, mass_std, val_mean, val_var).

**Total: 176 + 18 = 194 features.**

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
- 5-seed LGBM + XGBoost ensemble, strict 1:1 undersampling (661 half-metals)
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
│   ├── feature_selection.py       # cumulative gain importance pruning
│   ├── utils.py 
│   └── featurizer.py              # vec(), stats(), phys() — 194 features
│
├── scripts/
│   ├── run_featurize.py           # featurize AFLOW and supercon datasets

│   ├── model_train.py             # enthalpy + egap_type training
│   ├── model_egap.py              # band gap full benchmark (6 models)
│   ├── model_supercon.py          # Tc regression + variance decomposition
│   ├── train_hm.py                # one-shot vs two-stage comparison
│   ├── shap_supercon.py           # SHAP for Superconductivity
│   └── shap.py                    # unified SHAP for all 5 tasks
│
├── data/
│   ├── dataset.pkl                # raw AFLOW data
│   ├── superconductivity.pkl      # raw supercon2018 data
│   ├── data_feat.pkl              # featurized AFLOW (generated)
│   ├── supercon_feat.pkl          # featurized supercon (generated)
│   └── selected_features.json     # pruned feature sets per task (generated)
│
├── models/
│   └── {task}/
│       ├── {task}_lgbm.pkl
│       ├── {task}_xgb.pkl
│       ├── {task}_metrics.json
│       └── thresholds.json        # classification tasks
│
└── shap_outputs/
    └── {task}/
        ├── shap_summary_{task}.png
        ├── shap_bar_{task}.png
        ├── shap_dependence_*_{task}.png
        └── feature_importance_{task}.csv
```

---

## Installation

```bash
git clone https://github.com/your-username/StoichML.git
cd StoichML

conda create -n stoichml python=3.11
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
# → original columns + 194 feature columns
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
python -m scripts.shap_analysis --task all
python -m scripts.shap_analysis --task supercon   # includes per-bin analysis
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
python -m scripts.train_hm_class

# Step 4 — SHAP
python -m scripts.shap --task all
```

All metrics are saved as JSON files in `models/{task}/`. Figures are saved in `models/{task}/images/` and `shap_outputs/{task}/`.

---

## Known Limitations

**Composition-only ceiling.** Crystal symmetry and local coordination geometry are absent. Band gap prediction plateaus at R² = 0.828 regardless of model architecture. Tc prediction within compound families has R² < 0 across all Tc ranges — the model learns inter-family discrimination but not intra-family variation.

**Half-metal class imbalance.** Half-metals are rare (661 / 50,493 = 1.3%). The two-stage architecture substantially improves recall but precision remains limited by the rarity of the class.

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
  year    = {2025},
}
```

---

## License

MIT License. See `LICENSE` for full terms.