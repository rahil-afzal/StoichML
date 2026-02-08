import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# =====================
# Paths & constants
# =====================

DATA_PATH = "data/data_feat.pkl"
FEATURES_JSON = "data/selected_features.json"
MODEL_DIR = "models"
OUT_DIR = "shap_outputs"

MAX_SAMPLES = 10000
RANDOM_STATE = 42
TOP_K = 8
DPI = 1200

# =====================
# TASK registry
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
# Helpers
# =====================

def load_features(task):
    with open(FEATURES_JSON) as f:
        feats = json.load(f)
    core = feats.get("core", [])
    task_feats = feats.get(task, [])
    return list(dict.fromkeys(core + task_feats))


def select_shap_values(raw_sv, cfg):
    if cfg["type"] == "regression":
        return raw_sv
    if cfg["type"] == "binary":
        return raw_sv[1]
    if cfg["type"] == "multiclass":
        return raw_sv[cfg.get("shap_class", 0)]
    raise RuntimeError("Unknown task type")


# =====================
# SHAP routine
# =====================

def run_shap(task_name):

    cfg = TASKS[task_name]
    print(f"\n=== SHAP analysis: {task_name} ===")

    df = pd.read_pickle(DATA_PATH)
    features = load_features(task_name)
    X = df[features]

    if len(X) > MAX_SAMPLES:
        X = X.sample(MAX_SAMPLES, random_state=RANDOM_STATE)

    models = {
        "lgbm": joblib.load(f"{MODEL_DIR}/{task_name}_lgbm.pkl"),
        "xgb": joblib.load(f"{MODEL_DIR}/{task_name}_xgb.pkl"),
    }

    for name, model in models.items():
        model_tag = "LGBM" if name == "lgbm" else "XGBoost"
        out_path = os.path.join(OUT_DIR, task_name, name)
        os.makedirs(out_path, exist_ok=True)

        explainer = shap.TreeExplainer(model)
        raw_sv = explainer.shap_values(X)
        shap_vals = select_shap_values(raw_sv, cfg)

        mean_abs = np.abs(shap_vals).mean(axis=0)
        imp = pd.Series(mean_abs, index=X.columns).sort_values(ascending=False)
        top_feats = imp.index[:TOP_K]

        X_top = X[top_feats]
        shap_top = shap_vals[:, [X.columns.get_loc(f) for f in top_feats]]

        # -------- Summary Plot --------
        plt.figure(figsize=(7, 6))
        shap.summary_plot(shap_top, X_top, show=False, max_display=TOP_K)
        plt.suptitle(f"SHAP Summary – {model_tag} ({task_name.capitalize()})", fontsize=11)
        plt.figtext(
            0.5, -0.08,
            f"Figure: SHAP summary plot showing the top {TOP_K} most influential stoichiometric features "
            f"for {task_name} prediction using {model_tag}. Each point represents a material, colored by "
            f"feature value (low → high).",
            ha="center", fontsize=9, wrap=True
        )
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, f"shap_summary_{model_tag}_{task_name}.png"), dpi=DPI, bbox_inches="tight")
        plt.close()

        # -------- Bar Plot --------
        plt.figure(figsize=(7, 6))
        shap.summary_plot(shap_top, X_top, plot_type="bar", show=False)
        plt.suptitle(f"Mean |SHAP| Importance – {model_tag} ({task_name.capitalize()})", fontsize=11)
        plt.figtext(
            0.5, -0.08,
            f"Figure: Mean absolute SHAP values of the top {TOP_K} features for {task_name} prediction using "
            f"{model_tag}, indicating their global contribution to model output.",
            ha="center", fontsize=9, wrap=True
        )
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, f"shap_bar_{model_tag}_{task_name}.png"), dpi=DPI, bbox_inches="tight")
        plt.close()

        # -------- Dependence Plots --------
        for feat in top_feats:
            plt.figure(figsize=(6, 5))
            shap.dependence_plot(feat, shap_vals, X, show=False)
            plt.suptitle(f"{feat} vs SHAP – {model_tag} ({task_name})", fontsize=11)
            plt.figtext(
                0.5, -0.15,
                f"Figure: SHAP dependence plot showing how the feature '{feat}' influences the {task_name} "
                f"prediction in the {model_tag} model. Interaction effects are captured via color.",
                ha="center", fontsize=9, wrap=True
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(out_path, f"shap_dependence_{feat}_{model_tag}_{task_name}.png"),
                dpi=DPI, bbox_inches="tight"
            )
            plt.close()

        print(f"Completed SHAP for {model_tag}")

    print("\nSHAP analysis finished.")


# =====================
# CLI
# =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASKS.keys())
    args = parser.parse_args()
    run_shap(args.task)
