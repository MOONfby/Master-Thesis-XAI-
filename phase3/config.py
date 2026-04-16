"""
Phase 3 Dashboard Configuration.

All paths resolve from the `code/` working directory.
"""
from pathlib import Path

# ------------------------------------------------------------------
# Model and data paths
# ------------------------------------------------------------------
MODELS_DIR   = Path("models")
DATA_DIR     = Path("data")
RESULTS_DIR  = Path("results")

TS_MODEL_PATH      = MODELS_DIR / "inceptiontime_ecg5000.pth"
IMAGE_MODEL_PATH   = MODELS_DIR / "resnet50_cub.pth"
TABULAR_MODEL_PATH = MODELS_DIR / "xgboost_adult.pkl"

TS_DATA_PATH      = DATA_DIR / "ecg5000" / "ecg5000_processed.pkl"
IMAGE_DATA_PATH   = DATA_DIR / "cub200" / "cub200_processed.pkl"
TABULAR_DATA_PATH = DATA_DIR / "adult_processed.pkl"

TS_METRICS_CSV      = RESULTS_DIR / "phase2_ts"      / "metrics_summary.csv"
IMAGE_METRICS_CSV   = RESULTS_DIR / "phase1_image"   / "metrics_summary.csv"
TABULAR_METRICS_CSV = RESULTS_DIR / "phase1"         / "metrics_summary.csv"

# ------------------------------------------------------------------
# Modality keys (used in session state and UI)
# ------------------------------------------------------------------
MODALITY_TS       = "Time Series (ECG5000)"
MODALITY_IMAGE    = "Image (CUB-200-2011)"
MODALITY_TABULAR  = "Tabular (Adult Income)"

MODALITIES = [MODALITY_TS, MODALITY_IMAGE, MODALITY_TABULAR]

# ------------------------------------------------------------------
# Methods available per modality
# ------------------------------------------------------------------
TS_METHODS      = ["LIME-TS", "TimeSHAP", "Integrated Gradients"]
IMAGE_METHODS   = ["LIME", "GradientSHAP"]
TABULAR_METHODS = ["LIME", "SHAP"]

# ------------------------------------------------------------------
# Personas
# ------------------------------------------------------------------
PERSONA_ML        = "ML Engineer"
PERSONA_DOMAIN    = "Domain Expert"
PERSONA_REGULATOR = "Regulator / Auditor"

PERSONAS = [PERSONA_ML, PERSONA_DOMAIN, PERSONA_REGULATOR]

# ------------------------------------------------------------------
# ECG class names (ECG5000, 5-class)
# ------------------------------------------------------------------
ECG_CLASS_NAMES = [
    "Normal",
    "R-on-T PVC",
    "PVC",
    "Supra-ventricular Ectopic Beat",
    "Unclassifiable",
]

# ------------------------------------------------------------------
# Adult Income feature names (must match data_loader column order)
# ------------------------------------------------------------------
ADULT_NUMERICAL_FEATURES = [
    "age", "education-num", "capital-gain", "capital-loss", "hours-per-week",
]
ADULT_CATEGORICAL_FEATURES = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]
ADULT_FEATURE_NAMES = ADULT_NUMERICAL_FEATURES + ADULT_CATEGORICAL_FEATURES
ADULT_CLASS_NAMES   = ["<=50K", ">50K"]

# ------------------------------------------------------------------
# LLM
# ------------------------------------------------------------------
LLM_MODEL  = "deepseek-chat"   # DeepSeek model, accessible from mainland China
MAX_TOKENS = 1024
