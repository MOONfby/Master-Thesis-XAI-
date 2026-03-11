# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

KTH Master's thesis: *"Bridging Technical and Regulatory Transparency: A Multi-Method Evaluation and Lifecycle Integration for AI Systems"* — Biying Feng, 2026.

The thesis is structured in three phases:
1. **Phase 1** (implemented): Evaluate LIME, SHAP, and DiCE counterfactuals on tabular data (Adult Income dataset + XGBoost)
2. **Phase 2** (planned): High-performance explainer network using knowledge distillation (FastSHAP-style)
3. **Phase 3** (planned): Interactive LLM dashboard for human-readable explanations

## Running the Code

```bash
# Install dependencies
pip install -r requirements.txt

# Full Phase 1 experiment
python run_phase1.py

# Skip slow DiCE counterfactual generation (faster iteration)
python run_phase1.py --skip-cf

# Re-download data and retrain model from scratch
python run_phase1.py --force-reload
```

All scripts must be run from `code/` as the working directory (so `phase1` is importable as a package).

## Architecture

### Data flow

`data_loader.py` downloads the Adult Income dataset via `sklearn.fetch_openml`, applies `ColumnTransformer` (passthrough numerical + `OrdinalEncoder` for categorical), and caches the result to `data/adult_processed.pkl`. The output column order is **always**: `NUMERICAL_FEATURES` first, then `CATEGORICAL_FEATURES` (= `FEATURE_NAMES` in `config.py`). All downstream code depends on this ordering.

### Explainer interface contract

All attribution-based explainers (`LIMEExplainer`, `SHAPExplainer`) implement:
- `explain(instance: np.ndarray) -> np.ndarray` — single instance, returns shape `(n_features,)`
- `explain_batch(instances: np.ndarray) -> np.ndarray` — returns shape `(n, n_features)`
- `measure_runtime(instances) -> dict`

`DiCEExplainer` has a different interface (`generate`, `generate_batch`) because counterfactuals return full alternative instances, not scalar attributions.

### Metrics

Faithfulness and stability metrics take a **callable `predict_fn`** (not the model directly) so they remain model-agnostic. All perturbation-based metrics use `baseline_values` (training set mean) stored in the data dict for masking features. Stability metrics perturb **only numerical features** (indices in `data["num_indices"]`) to avoid invalid categorical values.

### Configuration

All experiment hyperparameters live in `phase1/config.py`. Key values:
- `EVAL_SAMPLE_SIZE = 300` — test instances used for evaluation
- `STABILITY_SAMPLE_SIZE = 100` — subset for stability (slow per-instance explainer calls)
- `FAITHFULNESS_N_STEPS = 10`, `FAITHFULNESS_TOP_K = 5`
- `CF_NUM_CFS = 5` — counterfactuals per instance

### Output locations
- Cached data: `data/adult_processed.pkl`
- Saved model: `models/xgboost_adult.pkl`
- Results CSV: `results/phase1/metrics_summary.csv`
- Figures: `results/phase1/figures/*.png`

## Key Design Decisions

- **OrdinalEncoder (not one-hot)** for categorical features: tree models handle ordinal integers natively, and it keeps features 1-to-1 mapped, which is required for clean per-feature attribution analysis in faithfulness/stability metrics.
- **SHAP TreeExplainer** uses `tree_path_dependent` (default) — fast and theoretically correct for tree models. Switch to `interventional` only if you need conditional independence semantics.
- **DiCE** wraps the raw sklearn-compatible XGBoost model (not a pipeline) and receives the ordinal-encoded DataFrame directly. All features are declared `continuous_features` in the DiCE Data object, which is valid for ordinal integers with tree-based models.
- **Figures** use `matplotlib.use("Agg")` (non-interactive) and are always saved to disk. Do not change this when running headlessly.

## Adding Phase 2

The explainer network (knowledge distillation) should be placed in `phase2/`. It will need:
- SHAP attributions from Phase 1 as training targets (already computed in `shap_explainer.explain_batch`)
- A new `ExplainerNetwork` class (MLP) that takes raw encoded features and outputs SHAP-approximating attributions
- Evaluation comparing attribution accuracy (MSE, rank correlation vs. TreeSHAP) and inference time speedup
