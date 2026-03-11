"""
Phase 1 Configuration: XAI Evaluation on Tabular Data (Adult Income Dataset)
Thesis: Bridging Technical and Regulatory Transparency - Biying Feng, KTH 2026
"""
import os
from pathlib import Path

# === Paths ===
CODE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CODE_DIR / "data"
SAVED_MODEL_DIR = CODE_DIR / "models"
RESULTS_DIR = CODE_DIR / "results" / "phase1"
FIGURES_DIR = RESULTS_DIR / "figures"

for d in [DATA_DIR, SAVED_MODEL_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === Reproducibility ===
RANDOM_STATE = 42

# === Dataset ===
TEST_SIZE = 0.2
# Number of test instances used for evaluation (balance speed vs statistical power)
EVAL_SAMPLE_SIZE = 300

# Adult Income feature specification
NUMERICAL_FEATURES = [
    "age", "education-num", "capital-gain", "capital-loss", "hours-per-week"
]
CATEGORICAL_FEATURES = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]
# Final feature order after ColumnTransformer: numerical first, then categorical
FEATURE_NAMES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET = "class"

# === XGBoost Model ===
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
}

# === LIME ===
LIME_N_SAMPLES = 1000       # neighborhood samples for local surrogate
LIME_N_FEATURES = 10        # max features in explanation

# === SHAP ===
# TreeExplainer is exact for XGBoost; no background samples needed
# background used for interventional SHAP variant
SHAP_BACKGROUND_SAMPLES = 100

# === DiCE (Counterfactuals) ===
CF_NUM_CFS = 5              # number of counterfactuals per instance
CF_DESIRED_CLASS = 1        # target class for counterfactuals (>50K)
CF_METHOD = "random"        # 'random' or 'genetic' or 'kdtree'

# === Evaluation Metrics ===
# Faithfulness: AOPC, Comprehensiveness, Sufficiency
FAITHFULNESS_N_STEPS = 10   # steps in perturbation curve (AOPC)
FAITHFULNESS_TOP_K = 5      # top-k features for comprehensiveness/sufficiency

# Stability: rank correlation and average sensitivity
STABILITY_N_PERTURBATIONS = 15     # perturbations per instance
STABILITY_NOISE_STD = 0.05         # Gaussian noise std (relative to feature std)
STABILITY_SAMPLE_SIZE = 100        # instances used for stability evaluation
