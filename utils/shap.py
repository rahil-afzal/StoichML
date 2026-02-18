import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import logging

# =====================
# Logging & Random Seed
# =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =====================
# Paths & Constants
# =====================
DATA_PATH = "data/data_feat.pkl"
FEATURES_JSON = "data/selected_features.json"
MODEL_BASE_DIR = "models"
OUT_DIR = "shap_outputs"

MAX_SAMPLES = 10000
TOP_K = 8
DPI = 1200

# =====================
# Task Registry
# =====================
TASKS = {
    "enthalpy": {"target": "enthalpy_formation_atom", "type": "regression"},
    "egap": {"target": "Egap", "type": "regression"},
    "egap_type": {"target": "Egap_type_numeric", "type": "binary"},
    "hm_class": {
        "target": "hm_class",
        "type": "multiclass",
        "num_class": 3,
        "shap_class": 2,
    },
}

# =====================
# Helper Functions
# =====================
def load_features(task: str) -> list:
    """Load combined core + task-specific features from JSON."""
    with open(FEATURES_JSON) as f:
        feats = json.load(f)
    core = feats.get("core", [])
    task_feats = feats.get(task, [])
    return list(dict.fromkeys(core + task_feats))


def select_shap_values(raw_sv: np.ndarray, cfg: dict) -> np.ndarray:
    """Select SHAP values according to task type."""
    if cfg["type"] == "regression":
        return raw_sv
    if cfg["type"] == "binary":
        return raw_sv[1]
    if cfg["type"] == "multiclass":
        return raw_sv[cfg.get("shap_class", 0)]
    raise RuntimeError(f"Unknown task type: {cfg['type']}")


def get_model_path(task: str, model_name: str) -> str:
    """Return the path to the model stored in models/{task}/{task}_{model}.pkl."""
    return os.path.join(MODEL_BASE_DIR, task, f"{task}_{model_name}.pkl")


# =====================
# SHAP Analysis Routine
# =====================
def run_shap(task_name: str, max_samples: int = MAX_SAMPLES):
    """Run SHAP analysis for one task."""
    cfg = TASKS[task_name]
    logging.info(f"Starting SHAP analysis for task: {task_name}")

    # Load data
    try:
        df = pd.read_pickle(DATA_PATH)
    except FileNotFoundError:
        logging.error(f"Data file not found: {DATA_PATH}")
        return

    features = load_features(task_name)
    X = df[features]

    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=RANDOM_STATE)
        logging.info(f"Sampled {max_samples} rows.")

    # Load models from task-specific folder
    models = {}
    for name in ["lgbm", "xgb"]:
        path = get_model_path(task_name, name)
        try:
            models[name] = joblib.load(path)
        except FileNotFoundError:
            logging.warning(f"Model not found: {path}, skipping.")

    if not models:
        logging.error("No models loaded. Skipping task.")
        return

    for name, model in models.items():
        model_tag = "LGBM" if name == "lgbm" else "XGBoost"
        out_path = os.path.join(OUT_DIR, task_name, name)
        os.makedirs(out_path, exist_ok=True)

        # SHAP Explainer
        try:
            explainer = shap.TreeExplainer(model)
        except Exception as e:
            logging.warning(f"Failed to create SHAP explainer for {model_tag}: {e}")
            continue

        raw_sv = explainer.shap_values(X)
        shap_vals = select_shap_values(raw_sv, cfg)

        # Top features
        mean_abs = np.abs(shap_vals).mean(axis=0)
        imp = pd.Series(mean_abs, index=X.columns).sort_values(ascending=False)
        top_feats = imp.index[:TOP_K]
        top_idx = [X.columns.get_loc(f) for f in top_feats]

        X_top = X[top_feats]
        shap_top = shap_vals[:, top_idx]

        # Save SHAP values
        np.save(os.path.join(out_path, f"shap_values_{model_tag}_{task_name}.npy"), shap_vals)

        # ---- Summary Plot ----
        plt.figure(figsize=(7, 6))
        shap.summary_plot(shap_top, X_top, show=False, max_display=TOP_K)
        plt.suptitle(f"SHAP Summary – {model_tag} ({task_name.capitalize()})", fontsize=11)
        plt.figtext(0.5, -0.08,
                    f"Top {TOP_K} features for {task_name} using {model_tag}.",
                    ha="center", fontsize=9, wrap=True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, f"shap_summary_{model_tag}_{task_name}.png"), dpi=DPI, bbox_inches="tight")
        plt.close()

        # ---- Bar Plot ----
        plt.figure(figsize=(7, 6))
        shap.summary_plot(shap_top, X_top, plot_type="bar", show=False)
        plt.suptitle(f"Mean |SHAP| Importance – {model_tag} ({task_name.capitalize()})", fontsize=11)
        plt.figtext(0.5, -0.08,
                    f"Mean absolute SHAP values of top {TOP_K} features for {task_name} using {model_tag}.",
                    ha="center", fontsize=9, wrap=True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, f"shap_bar_{model_tag}_{task_name}.png"), dpi=DPI, bbox_inches="tight")
        plt.close()

        # ---- Dependence Plots ----
        for feat in top_feats:
            plt.figure(figsize=(6, 5))
            shap.dependence_plot(feat, shap_vals, X, show=False)
            plt.suptitle(f"{feat} vs SHAP – {model_tag} ({task_name})", fontsize=11)
            plt.figtext(0.5, -0.15,
                        f"Feature '{feat}' influence on {task_name} prediction in {model_tag}.",
                        ha="center", fontsize=9, wrap=True)
            plt.tight_layout()
            plt.savefig(os.path.join(out_path, f"shap_dependence_{feat}_{model_tag}_{task_name}.png"),
                        dpi=DPI, bbox_inches="tight")
            plt.close()

        logging.info(f"Completed SHAP for {model_tag}")

    logging.info(f"Finished task: {task_name}")


# =====================
# CLI
# =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP analysis for materials ML tasks")
    parser.add_argument("--task", required=True, choices=list(TASKS.keys()) + ["all"], help="Task name or 'all'")
    parser.add_argument("--max_samples", type=int, default=MAX_SAMPLES, help="Max number of samples to use")
    args = parser.parse_args()

    if args.task == "all":
        for t in TASKS.keys():
            run_shap(t, max_samples=args.max_samples)
    else:
        run_shap(args.task, max_samples=args.max_samples)
