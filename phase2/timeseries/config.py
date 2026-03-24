"""
Configuration for Phase 2: Time Series XAI Evaluation.

Dataset : ECG5000 (UCR archive) — 5 classes, 500 train, 4500 test, length 140
Model   : InceptionTime (1D-CNN)
Methods : LIMESegment, TimeSHAP, Integrated Gradients
"""
from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
TS_DATA_DIR    = Path("data/ecg5000")
TS_MODEL_PATH  = Path("models/inceptiontime_ecg5000.pth")
TS_RESULTS_DIR = Path("results/phase2_ts")
TS_FIGURES_DIR = Path("results/phase2_ts/figures")

# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
SERIES_LENGTH  = 140
NUM_CLASSES    = 5
RANDOM_STATE   = 42

# ------------------------------------------------------------------
# Superpixel analogue: temporal segments
# 140 timesteps / 20 segments = 7 timesteps per segment
# Mirrors N_SEGMENTS=50 in phase1/image/config.py
# ------------------------------------------------------------------
N_SEGMENTS_TS  = 20    # temporal segments
SEGMENT_LENGTH = SERIES_LENGTH // N_SEGMENTS_TS   # 7 timesteps

# ------------------------------------------------------------------
# Evaluation subsets
# ------------------------------------------------------------------
EVAL_SAMPLE_SIZE      = 200
STABILITY_SAMPLE_SIZE = 50

# ------------------------------------------------------------------
# Faithfulness
# ------------------------------------------------------------------
FAITHFULNESS_N_STEPS = 10
FAITHFULNESS_TOP_K   = 4    # 20% of N_SEGMENTS_TS=20

# ------------------------------------------------------------------
# Stability
# ------------------------------------------------------------------
STABILITY_N_PERTURBATIONS = 10
STABILITY_NOISE_STD       = 0.02   # Gaussian σ in normalised series space

# ------------------------------------------------------------------
# Explainer hyperparameters
# ------------------------------------------------------------------
LIME_N_SAMPLES  = 1000
IG_N_STEPS      = 50    # Integrated Gradients interpolation steps

# ------------------------------------------------------------------
# InceptionTime training
# ------------------------------------------------------------------
BATCH_SIZE   = 64
LR           = 1e-3
MAX_EPOCHS   = 100
PATIENCE     = 15      # early stopping
