"""
run_featurize.py
────────────────
Loads raw datasets, runs the StoichML featurizer, and saves featurized
DataFrames to disk.

Tasks:
  aflow     data/dataset.pkl          →  data/data_feat.pkl
  supercon  data/superconductivity.pkl →  data/supercon_feat.pkl

Usage:
    python -m scripts.run_featurize                   # run all tasks
    python -m scripts.run_featurize --task aflow
    python -m scripts.run_featurize --task supercon
"""

import argparse
import os
import sys
import time

import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stoichml.featurizer import featurize

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

ELEMENTS_COL    = "elements"
COMPOSITION_COL = "composition"

TASKS = {
    "aflow": {
        "input":  "data/dataset.pkl",
        "output": "data/data_feat.pkl",
    },
    "supercon": {
        "input":  "data/superconductivity.pkl",
        "output": "data/supercon_feat.pkl",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# CORE
# ══════════════════════════════════════════════════════════════════════════════

def run_task(task_name: str) -> None:
    cfg = TASKS[task_name]

    print(f"\n{'═' * 55}")
    print(f"  Task: {task_name}")
    print(f"{'═' * 55}")

    # ── Load ──────────────────────────────────────────────────
    print(f"  Loading  →  {cfg['input']}")
    df = pd.read_pickle(cfg["input"])
    print(f"  {len(df)} rows  |  columns: {list(df.columns)}")

    # ── Validate ──────────────────────────────────────────────
    missing = [c for c in [ELEMENTS_COL, COMPOSITION_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required columns missing from {cfg['input']}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # ── Featurize ─────────────────────────────────────────────
    print(f"\n  Featurizing ...")
    t0      = time.time()
    df_feat = featurize(
        df,
        elements_col    = ELEMENTS_COL,
        composition_col = COMPOSITION_COL,
    )
    elapsed = time.time() - t0
    n_feat  = len(df_feat.columns) - len(df.columns)

    print(f"  Done in {elapsed:.1f}s")
    print(f"  {n_feat} feature columns added")
    print(f"  Output shape: {df_feat.shape}")

    # ── Save ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(cfg["output"]), exist_ok=True)
    df_feat.to_pickle(cfg["output"])
    print(f"\n  Saved  →  {cfg['output']}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="StoichML featurizer — run one task or all.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m scripts.run_featurize\n"
            "  python -m scripts.run_featurize --task aflow\n"
            "  python -m scripts.run_featurize --task supercon"
        ),
    )
    parser.add_argument(
        "--task",
        choices=list(TASKS.keys()),
        default=None,
        help="Task to featurize. If omitted, all tasks are run.",
    )
    args = parser.parse_args()

    tasks_to_run = [args.task] if args.task else list(TASKS.keys())

    for task in tasks_to_run:
        run_task(task)

    print(f"\n{'═' * 55}")
    print(f"  Done — featurized: {', '.join(tasks_to_run)}")
    print(f"{'═' * 55}")


if __name__ == "__main__":
    main()