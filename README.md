# StoichML

> **Stoichiometry-driven machine learning for materials property prediction — no crystal structure required.**

StoichML predicts materials properties directly from chemical composition using physics-informed elemental descriptors and gradient-boosted tree ensembles. It is designed for researchers who want interpretable, fast, and scalable screening of large composition spaces without DFT geometry as a prerequisite.

**Built on:** [AFLOW](http://aflow.org) | **Models:** LightGBM + XGBoost | **Interpretability:** SHAP

---

## Motivation

Most ML models for materials properties require a relaxed crystal structure — limiting their use to compounds already computed by DFT. StoichML takes the opposite approach: given only a chemical formula, it predicts whether a compound is a metal or insulator, estimates its band gap and formation enthalpy, and — its primary scientific target — identifies candidates for **half-metallic behaviour** relevant to spintronics and magnetoelectronics.

This makes StoichML practical for high-throughput screening of hypothetical compositions where no structural data exists.

---

## Tasks

| Task | Target | Type | Primary Metric |
|---|---|---|---|
| `enthalpy` | Formation enthalpy (eV/atom) | Regression | RMSE, MAE, R² |
| `egap` | Band gap (eV) | Regression | RMSE, MAE, R² |
| `egap_type` | Metal vs. insulator | Binary classification | ROC-AUC, PR-AUC |
| `hm_class` | Conductor / Insulator / Half-metal | Multiclass classification | Macro F1, per-class F1 |

The `hm_class` task is the core scientific contribution. Half-metals (class 2) are severely underrepresented in the AFLOW database — StoichML addresses this through class-weighted training and explicit monitoring of per-class recall across all CV folds.

---

## Dataset

StoichML is trained and evaluated on data from the **AFLOW** database — one of the largest open repositories of high-throughput DFT calculations for inorganic compounds. AFLOW provides formation enthalpies, electronic band gaps, and space group symmetry data computed at a consistent DFT level (LDA/GGA).

Electronic class labels (`hm_class`) are derived from the AFLOW band structure data:
- **Class 0** — Conductor (metallic, zero gap)
- **Class 1** — Insulator / semiconductor (finite gap, both spin channels)
- **Class 2** — Half-metal (finite gap in one spin channel only)

Raw AFLOW data is not redistributed here. See [aflow.org](http://aflow.org) for access and the `data/` preparation scripts for the preprocessing pipeline applied before featurization.

---

## Design philosophy

### Composition-only representation
No CIF file. No DFT geometry. No structure relaxation. Features are derived entirely from the element list and stoichiometric fractions. For a compound A_x B_y, the pipeline computes:

1. Per-element property vectors (19 properties each)
2. Weighted statistics over elements using stoichiometric fractions as weights
3. Composition-level physics features encoding mixing, mismatch, and electronic character

This representation is invariant to unit cell choice and scales to millions of hypothetical compounds in seconds.

### Physics-informed, not just descriptor-heavy
The feature set goes beyond standard MAGPIE descriptors. Key additions include:

- **Valence d-electron count** restricted to the valence shell only (fixing overcounting for 4d/5d elements with filled 3d cores)
- **Hund's rule unpaired electrons** — free-atom spin estimate, distinct from the solid-state magnetic moment
- **Solid-state magnetic moment** (μB/atom from NIST/CRC) — reflects crystal-field quenching, relevant to half-metal classification
- **Cohesive energy** — directly related to formation enthalpy via the Born-Haber cycle
- **Chemical hardness** η = (I₁ − EA) / 2 in consistent eV units throughout
- **f-block fraction** — distinguishes rare-earth magnetic compounds from d-block transition metals
- **Configurational entropy** and **Δχ (electronegativity span)** — thermodynamic mixing and bond ionicity priors

### Transparent by design
All models use `importance_type="gain"` (average loss improvement per split) rather than the default split-count, which systematically underranks sparse physics features. SHAP values are computed for every task and model. The feature selection pipeline reports which physics features fall below the signal threshold, making the selection auditable.

---

## Project structure

```
StoichML/
│
├── stoichml/
│   └── featurizer.py             # core featurizer — vec(), stats(), phys()
│
├── scripts/
│   ├── model_train.py            # training pipeline (--task flag)
│   ├── feature_pruning_lgbm.py   # CV-based feature selection per task
│   └── property_audit.py         # elemental property coverage audit
│
├── data/
│   ├── data_feat.pkl             # featurized dataset (generated)
│   └── selected_features.json   # pruned feature sets per task (generated)
│
├── models/
│   └── {task}/
│       ├── {task}_lgbm.pkl
│       ├── {task}_xgb.pkl
│       ├── {task}_metrics.json
│       └── {task}_thresholds.json    # binary tasks only
│
└── shap_outputs/
    └── {task}/
        └── *.png
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
tabulate         # property audit script
```

---

## Quickstart

### 1. Check elemental property coverage

Before featurizing, audit which elements in your dataset have missing properties and which are resolved by the patch table:

```bash
python -m scripts.property_audit
# → prints per-property missingness summary
# → exports property_audit_full.csv and property_audit_missing_only.csv
```

### 2. Featurize your dataset

Your input DataFrame needs two columns:
- `elements` — list of element symbols, e.g. `["Fe", "O"]`
- `composition` — list of stoichiometric counts in **reduced form**, e.g. `[2, 3]` for Fe₂O₃

```python
from stoichml.featurizer import featurize
import pandas as pd

df = pd.DataFrame({
    "elements":    [["Fe", "O"],  ["Co", "Mn", "O"]],
    "composition": [[2, 3],       [1, 1, 2]],
})

df_feat = featurize(df)
# → original columns + 110 feature columns per row
```

**Feature count breakdown:**
- 19 elemental properties × 5 weighted statistics = **95 elemental features**
- 15 composition-level physics features
- **110 total**

> ⚠️ `n_atoms` is convention-dependent. Provide compositions in lowest integer ratios (Fe₂O₃ not Fe₄O₆). AFLOW data is already reduced.

### 3. Run feature pruning

Selects the minimal feature set per task using 5-fold CV-averaged gain importances. Physics features are **force-retained** regardless of importance score — domain knowledge takes precedence over data-driven pruning for features with explicit physical justification.

```bash
python -m scripts.feature_pruning_lgbm
# → writes data/selected_features.json
```

The pruning script reports which physics features fall below the signal threshold per task, making the selection auditable.

### 4. Train

```bash
python -m scripts.model_train --task enthalpy
python -m scripts.model_train --task egap
python -m scripts.model_train --task egap_type
python -m scripts.model_train --task hm_class
```

Each run:
- Trains LightGBM and XGBoost in parallel
- Runs 5-fold stratified cross-validation with early stopping (patience = 200)
- Reports per-fold metrics including **per-class F1 for `hm_class`** — macro F1 alone does not reveal whether half-metals are ever predicted
- Determines optimal classification threshold from OOF probabilities (binary tasks) — unbiased, not per-fold
- Sets final model `n_estimators` to the mean best iteration across CV folds
- Saves models, metrics, and thresholds to `models/{task}/`

---

## Feature reference

### Elemental properties

For each property, five statistics are computed using stoichiometric fractions as weights: `mean`, `std`, `min`, `max`, `mad` (weighted mean absolute deviation), and `pos` = (mean − min) / (max − min).

> `rng` (= max − min) is intentionally excluded — it is fully determined by `min` and `max` and carries no additional information.

| Property | Symbol | Unit | Notes |
|---|---|---|---|
| Atomic number | `Z` | — | |
| Atomic mass | `mass` | u | |
| Pauling electronegativity | `chi` | — | Noble gases set to 0 |
| Covalent / atomic radius | `radius` | pm | |
| Atomic volume | `volume` | cm³/mol | |
| Dipole polarizability | `polar` | Bohr³ | |
| Chemical hardness | `hard` | eV | η = (I₁ − EA)/2 |
| Valence electron count | `val` | — | |
| Valence shell vacancies | `vac` | — | |
| Valence d-electron count | `dcount` | — | Valence shell only (n = nmax−1) |
| d-shell half-filling distance | `dhalf` | — | \|dcnt − 5\| |
| Unpaired electrons | `unpaired` | — | Hund's rule free-atom estimate |
| Electron affinity | `EA` | eV | Unstable anions set to 0 |
| First ionisation energy | `I1` | eV | |
| Melting point | `Tm` | K | Stable allotrope at STP |
| Boiling point | `Tb` | K | |
| Thermal conductivity | `kappa` | W/(m·K) | |
| Cohesive energy | `Ecoh` | kJ/mol | Born-Haber elemental reference |
| Solid-state magnetic moment | `magmom` | μB/atom | Non-magnetic elements set to 0 |

### Composition-level physics features

| Feature | Physical meaning | Primary task |
|---|---|---|
| `n_elements` | Number of distinct species | All |
| `n_atoms` | Total atoms per formula unit | `enthalpy` |
| `max_weight` | Stoichiometric fraction of majority element | All |
| `conf_entropy` | Configurational mixing entropy | All |
| `chi_mad` | Weighted electronegativity mismatch | `egap`, `egap_type` |
| `delta_chi` | Electronegativity span max−min (Phillips ionicity) | `egap`, `egap_type` |
| `r_mad` | Atomic size mismatch (lattice strain proxy) | `enthalpy` |
| `mass_std` | Mass dispersion | `enthalpy`, `hm_class` |
| `val_mean` | Weighted mean valence electron count | All |
| `val_var` | Valence electron dispersion | All |
| `dhalf_mean` | Mean d-shell half-filling distance | `hm_class` |
| `tm_frac` | d-block (transition metal) fraction | `egap`, `hm_class` |
| `f_frac` | f-block (lanthanide / actinide) fraction | `hm_class` |
| `unpaired_mean` | Mean unpaired electrons | `hm_class` |
| `unpaired_var` | Unpaired electron dispersion | `hm_class` |

---

## Training details

| Setting | Value |
|---|---|
| CV strategy | 5-fold StratifiedKFold (classification) / KFold (regression) |
| Early stopping patience | 200 rounds |
| Final `n_estimators` | Mean best iteration across CV folds |
| Importance type | `gain` (LightGBM + XGBoost) |
| Binary threshold | Youden-J optimised on OOF probabilities |
| Imbalance — binary | Balanced class weights |
| Imbalance — multiclass | Manual weights: conductor ×1, insulator ×1, half-metal ×8 |
| XGBoost multiclass weighting | `sample_weight` per fold and final fit |
| Band gap transform | log1p (clipped at 0 before transform) |

---

## Known limitations

**No structural awareness.** Topology-dependent phenomena — local bonding geometry, magnetic ordering, and Fermi surface topology — cannot be captured from composition alone. This sets an inherent precision ceiling, particularly for formation energy and spin-dependent properties.

**Half-metal class imbalance.** Half-metals (class 2) are rare in the AFLOW database. Class weighting partially compensates but high-recall detection of class 2 remains challenging. Future work may explore SMOTE, physics-guided synthetic augmentation, or retrieval-augmented approaches.

**Formula unit convention.** `n_atoms` requires reduced compositions. Non-reduced inputs give inconsistent values. AFLOW data satisfies this by default.

**Exotic elements.** Superheavy elements (Z ≥ 104) and heavy transuranics are missing 5–7 properties in mendeleev. These elements should not appear in real AFLOW materials datasets; if they do, the missingness signal is informative rather than noise.

---

## How StoichML compares

| Framework | Representation | Model | Interpretability |
|---|---|---|---|
| MAGPIE | Composition descriptors | Various | Partial |
| CrabNet | Composition + attention | Transformer | Low |
| ElemNet | Composition | Deep NN | Low |
| **StoichML** | **Extended physics descriptors** | **LGBM + XGB** | **SHAP, full** |

StoichML's distinguishing features are extended physics-informed descriptors beyond MAGPIE, explicit half-metal classification as a primary task, and a fully integrated SHAP interpretability pipeline across all four tasks.

---

## Contributing

Contributions, issues, and feature requests are welcome. If you use StoichML on a dataset other than AFLOW, please open an issue reporting which elements caused missing-property failures — this helps improve the patch table.

---

## License

MIT License. See `LICENSE` for full terms.
