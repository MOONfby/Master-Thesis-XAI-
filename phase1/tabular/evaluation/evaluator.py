"""
Phase 1 Evaluation Pipeline: Comprehensive XAI Method Comparison on Tabular Data.

This module orchestrates the full evaluation of LIME, SHAP, and DiCE
(Counterfactual Explanations) across all metrics:

  Faithfulness:
    - AOPC (feature attribution methods: LIME, SHAP)
    - Comprehensiveness (LIME, SHAP)
    - Sufficiency (LIME, SHAP)
    - LIME Local Fidelity R² (LIME only)

  Stability:
    - Rank Correlation Stability / Spearman ρ (LIME, SHAP)
    - Average Sensitivity / Lipschitz constant (LIME, SHAP)

  Counterfactual-specific (DiCE):
    - Validity
    - Proximity L1, L2
    - Sparsity
    - Diversity

  Runtime:
    - Mean explanation time per instance (all three methods)

Results are saved to:
  results/phase1/metrics_summary.csv     — aggregated scores
  results/phase1/metrics_detailed.csv    — per-instance scores
"""
import time
import numpy as np
import pandas as pd
from pathlib import Path

from phase1.tabular.config import (
    EVAL_SAMPLE_SIZE, FAITHFULNESS_N_STEPS, FAITHFULNESS_TOP_K,
    STABILITY_SAMPLE_SIZE, STABILITY_N_PERTURBATIONS, STABILITY_NOISE_STD,
    CF_NUM_CFS, LIME_N_SAMPLES, LIME_N_FEATURES, RESULTS_DIR, RANDOM_STATE
)
from phase1.tabular.metrics.faithfulness import (
    aopc_score, comprehensiveness, sufficiency, lime_local_fidelity_batch
)
from phase1.tabular.metrics.stability import rank_correlation_stability, average_sensitivity
from phase1.tabular.metrics.cf_metrics import evaluate_counterfactuals


class Phase1Evaluator:
    """
    Runs the complete Phase 1 evaluation for all three XAI methods.

    Usage
    -----
    evaluator = Phase1Evaluator(data, model, lime_exp, shap_exp, dice_exp)
    results = evaluator.run()
    evaluator.save_results(results)
    """

    def __init__(self, data: dict, model,
                 lime_explainer, shap_explainer, dice_explainer):
        self.data = data
        self.model = model
        self.lime_exp = lime_explainer
        self.shap_exp = shap_explainer
        self.dice_exp = dice_explainer

        # Sample a fixed subset of test instances for evaluation
        rng = np.random.default_rng(RANDOM_STATE)
        n_test = len(data["X_test"])
        n_eval = min(EVAL_SAMPLE_SIZE, n_test)
        self.eval_idx = rng.choice(n_test, size=n_eval, replace=False)

        self.X_eval = data["X_test"][self.eval_idx]
        self.y_eval = data["y_test"][self.eval_idx]
        self.X_eval_df = data["X_test_df"].iloc[self.eval_idx].reset_index(drop=True)
        self.baseline = data["baseline_values"]
        self.feature_std = data["feature_std"]
        self.num_indices = data["num_indices"]
        self.feature_names = data["feature_names"]

        # Feature ranges for normalized CF metrics
        self.feature_ranges = (
            data["X_train"].max(axis=0) - data["X_train"].min(axis=0) + 1e-8
        )

        # Stability uses a smaller subset to keep runtime manageable
        n_stab = min(STABILITY_SAMPLE_SIZE, n_eval)
        stab_idx = rng.choice(n_eval, size=n_stab, replace=False)
        self.X_stab = self.X_eval[stab_idx]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_fn(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    # ------------------------------------------------------------------
    # Step 1: Compute attributions (batch, for efficiency)
    # ------------------------------------------------------------------

    def _compute_attributions(self) -> tuple[np.ndarray, np.ndarray]:
        print("  Computing LIME attributions...")
        lime_attrs = self.lime_exp.explain_batch(
            self.X_eval, n_samples=LIME_N_SAMPLES, n_features=LIME_N_FEATURES
        )
        print("  Computing SHAP attributions (batch)...")
        shap_attrs = self.shap_exp.explain_batch(self.X_eval)
        return lime_attrs, shap_attrs

    # ------------------------------------------------------------------
    # Step 2: Faithfulness metrics
    # ------------------------------------------------------------------

    def _eval_faithfulness(self, lime_attrs: np.ndarray,
                           shap_attrs: np.ndarray) -> dict:
        print("\n--- Faithfulness Metrics ---")
        results = {}

        print("  [LIME] AOPC...")
        _, results["LIME_AOPC"] = aopc_score(
            self._predict_fn, self.X_eval, lime_attrs,
            self.baseline, FAITHFULNESS_N_STEPS
        )
        print(f"    LIME AOPC = {results['LIME_AOPC']:.4f}")

        print("  [SHAP] AOPC...")
        _, results["SHAP_AOPC"] = aopc_score(
            self._predict_fn, self.X_eval, shap_attrs,
            self.baseline, FAITHFULNESS_N_STEPS
        )
        print(f"    SHAP AOPC = {results['SHAP_AOPC']:.4f}")

        print("  [LIME] Comprehensiveness...")
        _, results["LIME_Comprehensiveness"] = comprehensiveness(
            self._predict_fn, self.X_eval, lime_attrs,
            self.baseline, FAITHFULNESS_TOP_K
        )
        print(f"    LIME Comprehensiveness = {results['LIME_Comprehensiveness']:.4f}")

        print("  [SHAP] Comprehensiveness...")
        _, results["SHAP_Comprehensiveness"] = comprehensiveness(
            self._predict_fn, self.X_eval, shap_attrs,
            self.baseline, FAITHFULNESS_TOP_K
        )
        print(f"    SHAP Comprehensiveness = {results['SHAP_Comprehensiveness']:.4f}")

        print("  [LIME] Sufficiency...")
        _, results["LIME_Sufficiency"] = sufficiency(
            self._predict_fn, self.X_eval, lime_attrs,
            self.baseline, FAITHFULNESS_TOP_K
        )
        print(f"    LIME Sufficiency = {results['LIME_Sufficiency']:.4f}")

        print("  [SHAP] Sufficiency...")
        _, results["SHAP_Sufficiency"] = sufficiency(
            self._predict_fn, self.X_eval, shap_attrs,
            self.baseline, FAITHFULNESS_TOP_K
        )
        print(f"    SHAP Sufficiency = {results['SHAP_Sufficiency']:.4f}")

        print("  [LIME] Local Fidelity R²...")
        _, results["LIME_LocalFidelity_R2"] = lime_local_fidelity_batch(
            self.lime_exp, self.X_eval[:50],  # subset for speed
            LIME_N_SAMPLES, LIME_N_FEATURES
        )
        print(f"    LIME Local Fidelity R² = {results['LIME_LocalFidelity_R2']:.4f}")

        return results

    # ------------------------------------------------------------------
    # Step 3: Stability metrics
    # ------------------------------------------------------------------

    def _eval_stability(self) -> dict:
        print("\n--- Stability Metrics ---")
        results = {}

        print("  [LIME] Rank Correlation Stability...")
        _, results["LIME_RankCorrelation"] = rank_correlation_stability(
            self.lime_exp, self.X_stab, self.num_indices, self.feature_std,
            STABILITY_N_PERTURBATIONS, STABILITY_NOISE_STD
        )
        print(f"    LIME Rank Correlation = {results['LIME_RankCorrelation']:.4f}")

        print("  [SHAP] Rank Correlation Stability...")
        _, results["SHAP_RankCorrelation"] = rank_correlation_stability(
            self.shap_exp, self.X_stab, self.num_indices, self.feature_std,
            STABILITY_N_PERTURBATIONS, STABILITY_NOISE_STD
        )
        print(f"    SHAP Rank Correlation = {results['SHAP_RankCorrelation']:.4f}")

        print("  [LIME] Average Sensitivity...")
        _, results["LIME_AvgSensitivity"] = average_sensitivity(
            self.lime_exp, self.X_stab, self.num_indices, self.feature_std,
            STABILITY_N_PERTURBATIONS, STABILITY_NOISE_STD
        )
        print(f"    LIME Avg Sensitivity = {results['LIME_AvgSensitivity']:.4f}")

        print("  [SHAP] Average Sensitivity...")
        _, results["SHAP_AvgSensitivity"] = average_sensitivity(
            self.shap_exp, self.X_stab, self.num_indices, self.feature_std,
            STABILITY_N_PERTURBATIONS, STABILITY_NOISE_STD
        )
        print(f"    SHAP Avg Sensitivity = {results['SHAP_AvgSensitivity']:.4f}")

        return results

    # ------------------------------------------------------------------
    # Step 4: Counterfactual metrics
    # ------------------------------------------------------------------

    def _eval_counterfactuals(self, n_cf_instances: int = 50) -> dict:
        print("\n--- Counterfactual Metrics ---")

        # Use a small subset for CF evaluation (can be slow)
        cf_eval_df = self.X_eval_df.iloc[:n_cf_instances]
        cf_eval_X = self.X_eval[:n_cf_instances]

        print(f"  [DiCE] Generating {CF_NUM_CFS} CFs for {n_cf_instances} instances...")
        cf_results = self.dice_exp.generate_batch(
            cf_eval_df, num_cfs=CF_NUM_CFS, desired_class=1
        )

        cf_metrics = evaluate_counterfactuals(cf_eval_X, cf_results, self.feature_ranges)

        results = {
            "CF_Validity": cf_metrics["validity"],
            "CF_Proximity_L1": cf_metrics["proximity_l1"],
            "CF_Proximity_L2": cf_metrics["proximity_l2"],
            "CF_Sparsity": cf_metrics["sparsity"],
            "CF_Diversity": cf_metrics["diversity"],
        }
        for k, v in results.items():
            print(f"    {k} = {v:.4f}" if v is not None and not np.isnan(v) else f"    {k} = N/A")
        return results

    # ------------------------------------------------------------------
    # Step 5: Runtime benchmarks
    # ------------------------------------------------------------------

    def _eval_runtime(self, n_runtime: int = 30) -> dict:
        print("\n--- Runtime Benchmarks ---")
        runtime_X = self.X_eval[:n_runtime]
        runtime_df = self.X_eval_df.iloc[:n_runtime]

        print(f"  Measuring LIME runtime ({n_runtime} instances)...")
        lime_rt = self.lime_exp.measure_runtime(runtime_X)
        print(f"    LIME mean: {lime_rt['mean_time']:.3f}s ± {lime_rt['std_time']:.3f}s")

        print(f"  Measuring SHAP runtime ({n_runtime} instances)...")
        shap_rt = self.shap_exp.measure_runtime(runtime_X)
        print(f"    SHAP mean: {shap_rt['mean_time']:.4f}s (batch: {shap_rt['batch_total_time']:.3f}s total)")

        print(f"  Measuring DiCE runtime ({n_runtime} instances)...")
        dice_rt = self.dice_exp.measure_runtime(runtime_df)
        print(f"    DiCE mean: {dice_rt['mean_time']:.3f}s ± {dice_rt['std_time']:.3f}s")

        return {
            "LIME_MeanTime_s": lime_rt["mean_time"],
            "LIME_StdTime_s": lime_rt["std_time"],
            "SHAP_MeanTime_s": shap_rt["mean_time"],
            "SHAP_StdTime_s": shap_rt["std_time"],
            "SHAP_BatchTotal_s": shap_rt["batch_total_time"],
            "CF_MeanTime_s": dice_rt["mean_time"],
            "CF_StdTime_s": dice_rt["std_time"],
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, skip_cf: bool = False) -> dict:
        """
        Run the full Phase 1 evaluation.

        Parameters
        ----------
        skip_cf : bool
            If True, skip counterfactual generation (slow). Useful for quick testing.

        Returns
        -------
        results_dict : dict with all metric scores
        """
        print("\n" + "=" * 60)
        print("PHASE 1: XAI EVALUATION ON TABULAR DATA")
        print(f"Dataset: Adult Income | Model: XGBoost")
        print(f"Evaluating {len(self.X_eval)} test instances")
        print("=" * 60)

        all_results = {}
        all_results["dataset"] = "Adult Income"
        all_results["model"] = "XGBoost"
        all_results["n_eval_instances"] = len(self.X_eval)

        # Step 1: Compute attributions once (reused across faithfulness + stability)
        lime_attrs, shap_attrs = self._compute_attributions()

        # Step 2: Faithfulness
        faith_results = self._eval_faithfulness(lime_attrs, shap_attrs)
        all_results.update(faith_results)

        # Step 3: Stability
        stab_results = self._eval_stability()
        all_results.update(stab_results)

        # Step 4: Counterfactuals
        if not skip_cf:
            cf_results = self._eval_counterfactuals()
            all_results.update(cf_results)

        # Step 5: Runtime
        runtime_results = self._eval_runtime()
        all_results.update(runtime_results)

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)

        return all_results

    def save_results(self, results: dict) -> None:
        """Save results summary to CSV."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        summary_df = pd.DataFrame([results])
        summary_path = RESULTS_DIR / "metrics_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nResults saved to: {summary_path}")

        # Also print a formatted summary table
        self._print_summary_table(results)

    def _print_summary_table(self, results: dict) -> None:
        print("\n" + "=" * 60)
        print("METRIC SUMMARY TABLE")
        print("=" * 60)

        sections = {
            "Faithfulness (↑ higher is better)": [
                ("AOPC", "LIME_AOPC", "SHAP_AOPC", None),
                ("Comprehensiveness", "LIME_Comprehensiveness", "SHAP_Comprehensiveness", None),
                ("Sufficiency (↓ lower)", "LIME_Sufficiency", "SHAP_Sufficiency", None),
                ("Local Fidelity R²", "LIME_LocalFidelity_R2", None, None),
            ],
            "Stability (↑ Rank Corr | ↓ Sensitivity)": [
                ("Rank Correlation ↑", "LIME_RankCorrelation", "SHAP_RankCorrelation", None),
                ("Avg Sensitivity ↓", "LIME_AvgSensitivity", "SHAP_AvgSensitivity", None),
            ],
            "Counterfactuals": [
                ("Validity ↑", None, None, "CF_Validity"),
                ("Proximity L1 ↓", None, None, "CF_Proximity_L1"),
                ("Proximity L2 ↓", None, None, "CF_Proximity_L2"),
                ("Sparsity ↓", None, None, "CF_Sparsity"),
                ("Diversity ↑", None, None, "CF_Diversity"),
            ],
            "Runtime (seconds/instance)": [
                ("Mean Time ↓", "LIME_MeanTime_s", "SHAP_MeanTime_s", "CF_MeanTime_s"),
            ],
        }

        fmt = "{:<30} {:>10} {:>10} {:>10}"
        print(fmt.format("Metric", "LIME", "SHAP", "CF/DiCE"))
        print("-" * 62)

        for section, items in sections.items():
            print(f"\n{section}")
            for label, lime_key, shap_key, cf_key in items:
                lime_val = f"{results[lime_key]:.4f}" if lime_key and lime_key in results else "—"
                shap_val = f"{results[shap_key]:.4f}" if shap_key and shap_key in results else "—"
                cf_val = f"{results[cf_key]:.4f}" if cf_key and cf_key in results and not (
                    isinstance(results[cf_key], float) and np.isnan(results[cf_key])
                ) else "—"
                print(fmt.format(f"  {label}", lime_val, shap_val, cf_val))
