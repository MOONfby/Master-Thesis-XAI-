"""
Configuration for Phase 1 Image XAI Evaluation.
Dataset: CUB-200-2011 | Model: ResNet-50 | Methods: LIME-image, GradientSHAP
"""
from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
IMAGE_DATA_DIR    = Path("data/cub200")
IMAGE_MODEL_PATH  = Path("models/resnet50_cub.pth")
IMAGE_RESULTS_DIR = Path("results/phase1_image")
IMAGE_FIGURES_DIR = Path("results/phase1_image/figures")

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
BACKBONE    = "resnet50"
IMAGE_SIZE  = 224
NUM_CLASSES = 200

# ------------------------------------------------------------------
# Superpixels (SLIC) — same parameters for ALL methods
# ------------------------------------------------------------------
N_SEGMENTS       = 50
SLIC_COMPACTNESS = 10
SLIC_SIGMA       = 1

# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
RANDOM_STATE             = 42
EVAL_SAMPLE_SIZE         = 200   # test images for main evaluation
STABILITY_SAMPLE_SIZE    = 50    # subset for stability (multiple passes per image)
FAITHFULNESS_N_STEPS     = 10    # progressive masking steps for AOPC
FAITHFULNESS_TOP_K       = 10    # top-k superpixels (20% of N_SEGMENTS=50)

# ------------------------------------------------------------------
# Stability
# ------------------------------------------------------------------
STABILITY_N_PERTURBATIONS = 10
STABILITY_NOISE_STD       = 0.02   # Gaussian noise std in normalised [0,1] pixel space

# ------------------------------------------------------------------
# LIME-image
# ------------------------------------------------------------------
LIME_NUM_SAMPLES  = 1000   # perturbation samples per explanation
LIME_NUM_FEATURES = N_SEGMENTS   # return all segments (= N_SEGMENTS)

# ------------------------------------------------------------------
# GradientSHAP
# ------------------------------------------------------------------
GRADSHAP_N_SAMPLES   = 50    # noise samples for gradient averaging
GRADSHAP_STDEV_NOISE = 0.1   # noise std for SmoothGrad-style averaging
GRADSHAP_BACKGROUND  = 20    # background reference images from training set
