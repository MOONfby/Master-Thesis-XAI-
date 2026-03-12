"""
Phase 1 Image Evaluation Pipeline.

Orchestrates the full evaluation of LIME-image and GradientSHAP
across all metrics: Faithfulness, Stability, Localisation, Runtime.

Mirrors the structure of phase1/tabular/evaluation/evaluator.py.
"""
import time
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.segmentation import slic

from phase1.image.config import (
    EVAL_SAMPLE_SIZE, STABILITY_SAMPLE_SIZE, RANDOM_STATE,
    FAITHFULNESS_N_STEPS, FAITHFULNESS_TOP_K,
    STABILITY_N_PERTURBATIONS, STABILITY_NOISE_STD,
    N_SEGMENTS, SLIC_COMPACTNESS, SLIC_SIGMA,
    IMAGE_RESULTS_DIR,
)
from phase1.image.metrics.faithfulness import (
    region_aopc_score, region_comprehensiveness, region_sufficiency,
    insertion_deletion_auc_batch,
)
from phase1.image.metrics.stability import (
    rank_correlation_stability, average_sensitivity,
)
from phase1.image.metrics.localization import evaluate_localization


class Phase1ImageEvaluator:
    """
    Runs the complete Phase 1 image evaluation for LIME-image and GradientSHAP.

    Usage
    -----
    evaluator = Phase1ImageEvaluator(data, model, predict_fn, lime_exp, gradshap_exp)
    results   = evaluator.run(skip_localization=False)
    evaluator.save_results(results)
    """

    def __init__(self, data: dict, model,
                 predict_fn,
                 lime_explainer,
                 gradshap_explainer):
        self.data          = data
        self.model         = model
        self.predict_fn    = predict_fn
        self.lime_exp      = lime_explainer
        self.gradshap_exp  = gradshap_explainer

        # Sample a fixed eval subset from test set
        rng    = np.random.default_rng(RANDOM_STATE)
        n_test = len(data["X_test"])
        n_eval = min(EVAL_SAMPLE_SIZE, n_test)
        self.eval_idx = rng.choice(n_test, size=n_eval, replace=False)

        self.X_eval     = data["X_test"][self.eval_idx]       # (N,3,H,W) normalised
        self.X_eval_raw = data["X_test_raw"][self.eval_idx]   # (N,3,H,W) in [0,1]
        self.y_eval     = data["y_test"][self.eval_idx]
        self.eval_ids   = [data["test_ids"][i] for i in self.eval_idx]
        self.annotations = data["annotations"]
        self.baseline   = data["baseline_image"]   # (3,H,W)

        # Stability uses a smaller subset
        n_stab = min(STABILITY_SAMPLE_SIZE, n_eval)
        stab_idx = rng.choice(n_eval, size=n_stab, replace=False)
        self.X_stab     = self.X_eval[stab_idx]
        self.stab_maps  = None   # filled in _precompute_superpixels

        # Storage — populated during run()
        self.segment_maps  = None   # (N, H, W)
        self.lime_attrs    = None   # (N, S)
        self.gradshap_attrs = None  # (N, S)
        self.target_labels = None   # (N,)

    # ------------------------------------------------------------------
    # Step 0: Pre-compute SLIC superpixels once for all eval images
    # ------------------------------------------------------------------

    def _precompute_superpixels(self):
        print("  Computing SLIC superpixels...")
        N, _, H, W = self.X_eval_raw.shape
        self.segment_maps = np.zeros((N, H, W), dtype=np.int32)
        for i in range(N):
            img_hwc = self.X_eval_raw[i].transpose(1, 2, 0)   # (H,W,3)
            self.segment_maps[i] = slic(
                img_hwc,
                n_segments=N_SEGMENTS,
                compactness=SLIC_COMPACTNESS,
                sigma=SLIC_SIGMA,
                start_label=0,
            )
        # Stability subset maps (same SLIC params)
        stab_n = len(self.X_stab)
        self.stab_maps = self.segment_maps[:stab_n]   # first stab_n rows
        print(f"    Superpixels computed for {N} images.")

    # ------------------------------------------------------------------
    # Step 1: Compute attributions (batch)
    # ------------------------------------------------------------------

    def _compute_attributions(self):
        print("\n  Computing target labels...")
        probs = self.predict_fn(self.X_eval)
        self.target_labels = probs.argmax(axis=1)

        print("  Computing LIME-image attributions (batch)...")
        self.lime_attrs = self.lime_exp.explain_batch(
            self.X_eval, self.segment_maps, labels=self.target_labels
        )

        print("  Computing GradientSHAP attributions (batch)...")
        self.gradshap_attrs = self.gradshap_exp.explain_batch(
            self.X_eval, self.segment_maps, labels=self.target_labels
        )

    # ------------------------------------------------------------------
    # Step 2: Faithfulness metrics
    # ------------------------------------------------------------------

    def _eval_faithfulness(self) -> dict:
        print("\n--- Faithfulness Metrics ---")
        results = {}
        kwargs = dict(
            predict_fn    = self.predict_fn,
            images        = self.X_eval,
            segment_maps  = self.segment_maps,
            baseline_image = self.baseline,
            target_labels = self.target_labels,
        )

        for name, attrs in [("LIME", self.lime_attrs),
                             ("GradSHAP", self.gradshap_attrs)]:
            print(f"  [{name}] AOPC...")
            _, results[f"{name}_AOPC"] = region_aopc_score(
                **kwargs, attributions=attrs, n_steps=FAITHFULNESS_N_STEPS
            )
            print(f"    {name} AOPC = {results[f'{name}_AOPC']:.4f}")

            print(f"  [{name}] Comprehensiveness...")
            _, results[f"{name}_Comprehensiveness"] = region_comprehensiveness(
                **kwargs, attributions=attrs, top_k=FAITHFULNESS_TOP_K
            )
            print(f"    {name} Comp = {results[f'{name}_Comprehensiveness']:.4f}")

            print(f"  [{name}] Sufficiency...")
            _, results[f"{name}_Sufficiency"] = region_sufficiency(
                **kwargs, attributions=attrs, top_k=FAITHFULNESS_TOP_K
            )
            print(f"    {name} Suff = {results[f'{name}_Sufficiency']:.4f}")

            print(f"  [{name}] Insertion/Deletion AUC...")
            ins_auc, del_auc = insertion_deletion_auc_batch(
                self.predict_fn, self.X_eval, self.segment_maps,
                attrs, self.baseline, target_labels=self.target_labels
            )
            results[f"{name}_InsertionAUC"] = ins_auc
            results[f"{name}_DeletionAUC"]  = del_auc
            print(f"    {name} InsAUC={ins_auc:.4f}  DelAUC={del_auc:.4f}")

        return results

    # ------------------------------------------------------------------
    # Step 3: Stability metrics
    # ------------------------------------------------------------------

    def _eval_stability(self) -> dict:
        print("\n--- Stability Metrics ---")
        results = {}

        for name, exp in [("LIME", self.lime_exp),
                          ("GradSHAP", self.gradshap_exp)]:
            print(f"  [{name}] Rank Correlation Stability...")
            _, results[f"{name}_RankCorrelation"] = rank_correlation_stability(
                exp, self.X_stab, self.stab_maps,
                STABILITY_N_PERTURBATIONS, STABILITY_NOISE_STD
            )
            print(f"    {name} Rank Corr = {results[f'{name}_RankCorrelation']:.4f}")

            print(f"  [{name}] Average Sensitivity...")
            _, results[f"{name}_AvgSensitivity"] = average_sensitivity(
                exp, self.X_stab, self.stab_maps,
                STABILITY_N_PERTURBATIONS, STABILITY_NOISE_STD
            )
            print(f"    {name} Avg Sens = {results[f'{name}_AvgSensitivity']:.4f}")

        return results

    # ------------------------------------------------------------------
    # Step 4: Localisation metrics
    # ------------------------------------------------------------------

    def _eval_localization(self) -> dict:
        print("\n--- Localisation Metrics ---")
        results = {}

        for name, exp in [("LIME", self.lime_exp),
                          ("GradSHAP", self.gradshap_exp)]:
            print(f"  [{name}] Pointing Game + Segmentation IoU...")
            loc = evaluate_localization(
                exp, self.X_eval, self.segment_maps,
                self.annotations, self.eval_ids,
                target_labels=self.target_labels
            )
            results[f"{name}_PointingGame"]   = loc["pointing_game_acc"]
            results[f"{name}_SegIoU"]         = loc["mean_iou"]
            print(f"    {name} Pointing Game = {loc['pointing_game_acc']:.4f}  "
                  f"(n={loc['n_pointing']})")
            print(f"    {name} Seg IoU       = {loc['mean_iou']:.4f}  "
                  f"(n={loc['n_iou']})")

        return results

    # ------------------------------------------------------------------
    # Step 5: Runtime benchmarks
    # ------------------------------------------------------------------

    def _eval_runtime(self, n_runtime: int = 20) -> dict:
        print("\n--- Runtime Benchmarks ---")
        rt_imgs = self.X_eval[:n_runtime]
        rt_maps = self.segment_maps[:n_runtime]

        lime_rt    = self.lime_exp.measure_runtime(rt_imgs, rt_maps)
        gradshap_rt = self.gradshap_exp.measure_runtime(rt_imgs, rt_maps)

        print(f"    LIME     mean: {lime_rt['mean_time']:.3f}s "
              f"± {lime_rt['std_time']:.3f}s")
        print(f"    GradSHAP mean: {gradshap_rt['mean_time']:.3f}s "
              f"± {gradshap_rt['std_time']:.3f}s")

        return {
            "LIME_MeanTime_s":     lime_rt["mean_time"],
            "LIME_StdTime_s":      lime_rt["std_time"],
            "GradSHAP_MeanTime_s": gradshap_rt["mean_time"],
            "GradSHAP_StdTime_s":  gradshap_rt["std_time"],
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, skip_localization: bool = False) -> dict:
        """
        Run the full Phase 1 image evaluation.

        Parameters
        ----------
        skip_localization : bool
            If True, skip Pointing Game / IoU (requires CUB-200 GT masks).

        Returns
        -------
        dict with all metric scores
        """
        print("\n" + "=" * 60)
        print("PHASE 1: XAI EVALUATION ON IMAGE DATA")
        print("Dataset: CUB-200-2011 | Model: ResNet-50")
        print(f"Methods: LIME-image, GradientSHAP")
        print(f"Evaluating {len(self.X_eval)} test images")
        print("=" * 60)

        all_results = {
            "dataset": "CUB-200-2011",
            "model":   "ResNet-50",
            "n_eval_instances": len(self.X_eval),
        }

        self._precompute_superpixels()
        self._compute_attributions()

        all_results.update(self._eval_faithfulness())
        all_results.update(self._eval_stability())

        if not skip_localization:
            all_results.update(self._eval_localization())

        all_results.update(self._eval_runtime())

        print("\n" + "=" * 60)
        print("IMAGE EVALUATION COMPLETE")
        print("=" * 60)

        return all_results

    def save_results(self, results: dict) -> None:
        """Save results summary to CSV."""
        IMAGE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = IMAGE_RESULTS_DIR / "metrics_summary.csv"
        pd.DataFrame([results]).to_csv(summary_path, index=False)
        print(f"\nResults saved to: {summary_path}")
        self._print_summary_table(results)

    def _print_summary_table(self, results: dict) -> None:
        print("\n" + "=" * 60)
        print("IMAGE METRIC SUMMARY TABLE")
        print("=" * 60)

        sections = {
            "Faithfulness (LIME | GradSHAP)": [
                ("AOPC ↑",           "LIME_AOPC",             "GradSHAP_AOPC"),
                ("Comprehensiveness ↑","LIME_Comprehensiveness","GradSHAP_Comprehensiveness"),
                ("Sufficiency ↓",    "LIME_Sufficiency",       "GradSHAP_Sufficiency"),
                ("Insertion AUC ↑",  "LIME_InsertionAUC",      "GradSHAP_InsertionAUC"),
                ("Deletion AUC ↓",   "LIME_DeletionAUC",       "GradSHAP_DeletionAUC"),
            ],
            "Stability": [
                ("Rank Correlation ↑","LIME_RankCorrelation",  "GradSHAP_RankCorrelation"),
                ("Avg Sensitivity ↓", "LIME_AvgSensitivity",   "GradSHAP_AvgSensitivity"),
            ],
            "Localisation": [
                ("Pointing Game ↑",  "LIME_PointingGame",      "GradSHAP_PointingGame"),
                ("Seg IoU ↑",        "LIME_SegIoU",            "GradSHAP_SegIoU"),
            ],
            "Runtime (s/image)": [
                ("Mean Time ↓",      "LIME_MeanTime_s",        "GradSHAP_MeanTime_s"),
            ],
        }

        fmt = "{:<30} {:>12} {:>12}"
        print(fmt.format("Metric", "LIME", "GradSHAP"))
        print("-" * 56)

        for section, items in sections.items():
            print(f"\n{section}")
            for label, lime_key, shap_key in items:
                lv = f"{results[lime_key]:.4f}" if lime_key in results and not (
                    isinstance(results.get(lime_key), float) and
                    results[lime_key] != results[lime_key]   # isnan check
                ) else "—"
                sv = f"{results[shap_key]:.4f}" if shap_key in results and not (
                    isinstance(results.get(shap_key), float) and
                    results[shap_key] != results[shap_key]
                ) else "—"
                print(fmt.format(f"  {label}", lv, sv))
