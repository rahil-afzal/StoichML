
"""
run_featurize.py
────────────────
Loads the raw dataset from data/data.pkl, runs the StoichML featurizer,
and saves the featurized DataFrame to data/data_feat.pkl.

Expected columns in data.pkl:
  elements     list of element symbols, e.g. ['Fe', 'O']
  composition  list of stoichiometric counts in reduced form, e.g. [2, 3]

Output:
  data/data_feat.pkl  — original columns + 151 feature columns

Usage:
    python -m scripts.run_featurize
    python scripts/run_featurize.py
"""

import os
import sys
import time
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
# Allows running from the project root as either:
#   python -m scripts.run_featurize
#   python scripts/run_featurize.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stoichml.featurizer import featurize

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

INPUT_PATH  = "data/dataset.pkl"
OUTPUT_PATH = "data/data_feat.pkl"

ELEMENTS_COL    = "elements"
COMPOSITION_COL = "composition"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():

    # ── Load ──────────────────────────────────────────────────────────────
    print(f"Loading  →  {INPUT_PATH}")
    df = pd.read_pickle(INPUT_PATH)
    print(f"  {len(df)} rows  |  columns: {list(df.columns)}")

    # ── Validate required columns ─────────────────────────────────────────
    missing = [c for c in [ELEMENTS_COL, COMPOSITION_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required columns missing from {INPUT_PATH}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # ── Featurize ─────────────────────────────────────────────────────────
    print(f"\nFeaturizing  ...")
    t0 = time.time()

    df_feat = featurize(
        df,
        elements_col=ELEMENTS_COL,
        composition_col=COMPOSITION_COL,
    )

    elapsed = time.time() - t0
    n_feat  = len(df_feat.columns) - len(df.columns)

    print(f"  Done in {elapsed:.1f}s")
    print(f"  {n_feat} feature columns added")
    print(f"  Output shape: {df_feat.shape}")

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_feat.to_pickle(OUTPUT_PATH)
    print(f"\nSaved  →  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()