# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

KTH Master's thesis: *"Bridging Technical and Regulatory Transparency: A Multi-Method Evaluation and Lifecycle Integration for AI Systems"* — Biying Feng, 2026.

Three phases:
1. **Phase 1** (implemented): XAI evaluation on tabular data (Adult Income + XGBoost) and image data (CUB-200-2011 + ResNet-50)
2. **Phase 2** (planned): Explainer network via knowledge distillation (FastSHAP-style), using GradientSHAP outputs from Phase 1 as training targets
3. **Phase 3** (planned): Interactive LLM dashboard for human-readable explanations

## Running the Code

All scripts must be run from `code/` as the working directory.

```bash
pip install -r requirements.txt

# Phase 1 — Tabular
python run_phase1.py                  # full evaluation
python run_phase1.py --skip-cf        # skip slow DiCE counterfactual generation
python run_phase1.py --force-reload   # re-download data and retrain model

# Phase 1 — Image
python run_phase1_image.py                      # full evaluation (downloads CUB-200, fine-tunes ResNet-50)
python run_phase1_image.py --skip-localization  # skip Pointing Game / Seg IoU
python run_phase1_image.py --force-reload       # re-download and retrain
```

## Architecture

### Package structure

```
phase1/
├── tabular/    ← Adult Income + XGBoost evaluation (LIME, SHAP, DiCE)
└── image/      ← CUB-200-2011 + ResNet-50 evaluation (LIME-image, GradientSHAP)
```

Each subpackage follows the same layout: `config.py`, `data_loader.py`, `model_trainer.py`, `explainers/`, `metrics/`, `evaluation/evaluator.py`, `visualization/plots.py`.

### Tabular data flow

`phase1/tabular/data_loader.py` downloads Adult Income via `sklearn.fetch_openml` (UCI ID 1590), drops `fnlwgt`, applies `ColumnTransformer` (numerical passthrough + `OrdinalEncoder` for categoricals), and caches to `data/adult_processed.pkl`. Column order is always `NUMERICAL_FEATURES` then `CATEGORICAL_FEATURES` (= `FEATURE_NAMES` in `config.py`) — all downstream code depends on this ordering. The returned dict also carries `baseline_values` (training mean), `feature_std`, `num_indices`, `cat_indices`, and `X_train_df`/`X_test_df` DataFrames for LIME and DiCE.

### Image data flow

`phase1/image/data_loader.py` downloads CUB-200-2011 (tar.gz from Caltech), extracts bounding boxes and segmentation masks, applies ImageNet normalisation, and caches to `data/cub200/cub200_processed.pkl`. Returns `baseline_image` (per-channel training mean, used for masking) and `background_images` (20 random training images, used as GradientSHAP references).

### Shared design principle: superpixel as feature unit

Image explainers output `(S,)` attribution vectors over SLIC superpixels (`N_SEGMENTS=50`), mirroring tabular's `(n_features,)=14`. This lets faithfulness and stability metric logic be shared across modalities. SLIC is pre-computed once per image in `_precompute_superpixels()` and shared across both methods.

### Explainer interface contract (both modalities)

Attribution-based explainers implement:
- `explain(instance, [segment_map]) -> np.ndarray (n_features,)` or `(S,)`
- `explain_batch(instances, [segment_maps]) -> np.ndarray (N, ...)`
- `measure_runtime(instances, ...) -> dict`

Image explainers additionally implement `explain_pixel_level()` returning `(H, W)` heatmaps used for localization metrics.

### Metrics

**Tabular** (`phase1/tabular/metrics/`): faithfulness (AOPC, Comprehensiveness, Sufficiency, Local Fidelity R²), stability (Rank Correlation, Average Sensitivity), counterfactual quality (Validity, Proximity, Sparsity, Diversity).

**Image** (`phase1/image/metrics/`): same faithfulness/stability metrics adapted for superpixel masking via `apply_superpixel_mask()`, plus Insertion/Deletion AUC (image-specific), and localization (Pointing Game, Segmentation IoU) using CUB-200-2011 GT bounding boxes and segmentation masks.

Perturbation baseline: tabular uses training-set column means; image uses per-channel training-set mean image (`baseline_image` in data dict).

### Configuration

Hyperparameters are in `phase1/tabular/config.py` and `phase1/image/config.py`. Key image values:
- `EVAL_SAMPLE_SIZE=200`, `STABILITY_SAMPLE_SIZE=50`
- `N_SEGMENTS=50`, `FAITHFULNESS_TOP_K=10` (20% of segments)
- `LIME_NUM_SAMPLES=1000`, `GRADSHAP_BACKGROUND=20`

### Output locations

| Run | Data cache | Model | Results CSV | Figures |
|---|---|---|---|---|
| Tabular | `data/adult_processed.pkl` | `models/xgboost_adult.pkl` | `results/phase1/metrics_summary.csv` | `results/phase1/figures/` |
| Image | `data/cub200/cub200_processed.pkl` | `models/resnet50_cub.pth` | `results/phase1_image/metrics_summary.csv` | `results/phase1_image/figures/` |

## Key Design Decisions

- **OrdinalEncoder (not one-hot)** for tabular categoricals: keeps features 1-to-1 mapped, required for clean per-feature attribution in faithfulness/stability metrics. DiCE receives the ordinal-encoded DataFrame directly with all features declared `continuous_features`.
- **SHAP TreeExplainer** uses `tree_path_dependent` (default). Switch to `interventional` only if conditional independence semantics are needed.
- **ResNet-50** fine-tuned on CUB-200-2011: differential learning rates (backbone 1e-5, FC head 1e-4), AdamW, CosineAnnealingLR, 30 epochs. GradientSHAP uses `shap.GradientExplainer` against 20 random training images as background references.
- **Stability perturbation**: tabular perturbs only `num_indices` (avoids invalid categoricals); image applies Gaussian noise to all pixels and clips to `[0, 1]`.
- **Figures** use `matplotlib.use("Agg")` — do not change when running headlessly.

## Adding Phase 2

Place in `phase2/`. Training targets are GradientSHAP attributions from `phase1/image/explainers/gradshap_explainer.py` (`explain_batch` output). The ExplainerNetwork (MLP) maps raw image features → SHAP-approximating `(S,)` vectors. Evaluate with MSE, rank correlation vs. GradientSHAP, and inference time speedup.
