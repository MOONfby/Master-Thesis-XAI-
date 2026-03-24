"""
Phase 2 Time Series Evaluation Pipeline.

Orchestrates the full evaluation of LIMESegment, TimeSHAP, and Integrated
Gradients across all metrics: Faithfulness, Stability, Localisation, Runtime.

Mirrors phase1/image/evaluation/evaluator.py exactly:
  - Same __init__ signature pattern
  - Same run() / save_results() / _print_summary_table() structure
  - Superpixel maps → temporal segment maps
  - segment_maps (N, H, W) → segment_maps (N, T)
"""
import time
import numpy as np
import pandas as pd
from pathlib import Path

from phase2.timeseries.config import (
    EVAL_SAMPLE_SIZE, STABILITY_SAMPLE_SIZE, RANDOM_STATE,
    FAITHFULNESS_N_STEPS, FAITHFULNESS_TOP_K,
    STABILITY_N_PERTURBATIONS, STABILITY_NOISE_STD,
    N_SEGMENTS_TS, TS_RESULTS_DIR,
)
from phase2.timeseries.utils import build_segment_map
from phase2.timeseries.metrics.faithfulness import (
    region_aopc_score, region_comprehensiveness, region_sufficiency,
    insertion_deletion_auc_batch,
)
from phase2.timeseries.metrics.stability import (
    rank_correlation_stability, average_sensitivity,
)
from phase2.timeseries.metrics.localization import evaluate_localization


class Phase2TSEvaluator:
    """
    Runs the complete Phase 2 time series evaluation.

    Usage
    -----
    evaluator = Phase2TSEvaluator(data, model, predict_fn,
                                  lime_exp, timeshap_exp, ig_exp)
    results   = evaluator.run(skip_localization=False)
    evaluator.save_results(results)
    """

    def __init__(self, data: dict,
                 model,
                 predict_fn,
                 lime_explainer,
                 timeshap_explainer,
                 ig_explainer):
        self.data          = data
        self.model         = model
        self.predict_fn    = predict_fn
        self.lime_exp      = lime_explainer
        self.timeshap_exp  = timeshap_explainer
        self.ig_exp        = ig_explainer

        # Fixed temporal segment map (same for all series — uniform windows)
        T = data["X_test"].shape[-1]
        self._segment_map_1d = build_segment_map(T, N_SEGMENTS_TS)  # (T,)

        # Sample a fixed eval subset from test set
        rng    = np.random.default_rng(RANDOM_STATE)
        n_test = len(data["X_test"])
        n_eval = min(EVAL_SAMPLE_SIZE, n_test)
        self.eval_idx = rng.choice(n_test, size=n_eval, replace=False)

        self.X_eval     = data["X_test"][self.eval_idx]    # (N, 1, T)
        self.y_eval     = data["y_test"][self.eval_idx]    # (N,)
        self.baseline   = data["baseline_series"]          # (1, T)
        self.qrs_windows = data["qrs_windows"]

        # Stability uses a smaller subset
        n_stab = min(STABILITY_SAMPLE_SIZE, n_eval)
        stab_idx = rng.choice(n_eval, size=n_stab, replace=False)
        self.X_stab = self.X_eval[stab_idx]

        # Segment maps: same 1D map broadcast to (N, T)
        N = len(self.X_eval)
        self.segment_maps = np.tile(self._segment_map_1d, (N, 1))  # (N, T)
        stab_N = len(self.X_stab)
        self.stab_maps = np.tile(self._segment_map_1d, (stab_N, 1))

        # Storage — populated during run()
        self.lime_attrs    = None   # (N, S)
        self.timeshap_attrs = None  # (N, S)
        self.ig_attrs      = None   # (N, S)
        self.target_labels = None   # (N,)

    # ------------------------------------------------------------------
    # Step 1: Compute attributions (batch)
    # ------------------------------------------------------------------

    def _compute_attributions(self):
        print("\n  Computing target labels...")
        probs = self.predict_fn(self.X_eval)
        self.target_labels = probs.argmax(axis=1)

        print("  Computing LIME-TS attributions (batch)...")
        self.lime_attrs = self.lime_exp.explain_batch(
            self.X_eval, self.segment_maps, labels=self.target_labels
        )

        print("  Computing TimeSHAP attributions (batch)...")
        self.timeshap_attrs = self.timeshap_exp.explain_batch(
            self.X_eval, self.segment_maps, labels=self.target_labels
        )

        print("  Computing Integrated Gradients attributions (batch)...")
        self.ig_attrs = self.ig_exp.explain_batch(
            self.X_eval, self.segment_maps, labels=self.target_labels
        )

    # ------------------------------------------------------------------
    # Step 2: Faithfulness metrics
    # ------------------------------------------------------------------

    def _eval_faithfulness(self) -> dict:
        print("\n--- Faithfulness Metrics ---")
        results = {}
        kwargs = dict(
            predict_fn     = self.predict_fn,
            series_batch   = self.X_eval,
            segment_maps   = self.segment_maps,
            baseline_series = self.baseline,
            target_labels  = self.target_labels,
        )

        methods = [
            ("LIME",     self.lime_attrs),
            ("TimeSHAP", self.timeshap_attrs),
            ("IG",       self.ig_attrs),
        ]

        for name, attrs in methods:
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

        methods = [
            ("LIME",     self.lime_exp),
            ("TimeSHAP", self.timeshap_exp),
            ("IG",       self.ig_exp),
        ]

        for name, exp in methods:
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
    # Step 4: Temporal Localisation metrics
    # ------------------------------------------------------------------

    def _eval_localization(self) -> dict:
        print("\n--- Temporal Localisation Metrics ---")
        results = {}

        # Localisation uses test-set indices → map to qrs_windows keys
        # qrs_windows are keyed by train index; for test we detect on-the-fly
        test_qrs = _detect_qrs_for_batch(self.X_eval)
        sample_indices = list(range(len(self.X_eval)))

        methods = [
            ("LIME",     self.lime_exp),
            ("TimeSHAP", self.timeshap_exp),
            ("IG",       self.ig_exp),
        ]

        for name, exp in methods:
            print(f"  [{name}] Temporal Pointing Game + IoU...")
            loc = evaluate_localization(
                exp, self.X_eval, self.segment_maps,
                test_qrs, sample_indices,
                target_labels=self.target_labels
            )
            results[f"{name}_PointingGame"] = loc["pointing_game_acc"]
            results[f"{name}_TemporalIoU"]  = loc["mean_iou"]
            print(f"    {name} Pointing Game = {loc['pointing_game_acc']:.4f}  "
                  f"(n={loc['n_evaluated']})")
            print(f"    {name} Temporal IoU  = {loc['mean_iou']:.4f}")

        return results

    # ------------------------------------------------------------------
    # Step 5: Runtime benchmarks
    # ------------------------------------------------------------------

    def _eval_runtime(self, n_runtime: int = 20) -> dict:
        print("\n--- Runtime Benchmarks ---")
        rt_series = self.X_eval[:n_runtime]
        rt_maps   = self.segment_maps[:n_runtime]

        lime_rt    = self.lime_exp.measure_runtime(rt_series, rt_maps)
        timeshap_rt = self.timeshap_exp.measure_runtime(rt_series, rt_maps)
        ig_rt       = self.ig_exp.measure_runtime(rt_series, rt_maps)

        print(f"    LIME     mean: {lime_rt['mean_time']:.3f}s "
              f"± {lime_rt['std_time']:.3f}s")
        print(f"    TimeSHAP mean: {timeshap_rt['mean_time']:.3f}s "
              f"± {timeshap_rt['std_time']:.3f}s")
        print(f"    IG       mean: {ig_rt['mean_time']:.3f}s "
              f"± {ig_rt['std_time']:.3f}s")

        return {
            "LIME_MeanTime_s":     lime_rt["mean_time"],
            "LIME_StdTime_s":      lime_rt["std_time"],
            "TimeSHAP_MeanTime_s": timeshap_rt["mean_time"],
            "TimeSHAP_StdTime_s":  timeshap_rt["std_time"],
            "IG_MeanTime_s":       ig_rt["mean_time"],
            "IG_StdTime_s":        ig_rt["std_time"],
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, skip_localization: bool = False) -> dict:
        """
        Run the full Phase 2 time series evaluation.

        Parameters
        ----------
        skip_localization : bool
            If True, skip Temporal Pointing Game / IoU.

        Returns
        -------
        dict with all metric scores
        """
        print("\n" + "=" * 60)
        print("PHASE 2: XAI EVALUATION ON TIME SERIES DATA")
        print("Dataset: ECG5000 | Model: InceptionTime")
        print("Methods: LIME-TS, TimeSHAP, Integrated Gradients")
        print(f"Evaluating {len(self.X_eval)} test series")
        print("=" * 60)

        all_results = {
            "dataset": "ECG5000",
            "model":   "InceptionTime",
            "n_eval_instances": len(self.X_eval),
        }

        self._compute_attributions()

        all_results.update(self._eval_faithfulness())
        all_results.update(self._eval_stability())

        if not skip_localization:
            all_results.update(self._eval_localization())

        all_results.update(self._eval_runtime())

        print("\n" + "=" * 60)
        print("TIME SERIES EVALUATION COMPLETE")
        print("=" * 60)

        return all_results

    def save_results(self, results: dict) -> None:
        """Save results summary to CSV."""
        TS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = TS_RESULTS_DIR / "metrics_summary.csv"
        pd.DataFrame([results]).to_csv(summary_path, index=False)
        print(f"\nResults saved to: {summary_path}")
        self._print_summary_table(results)

    def _print_summary_table(self, results: dict) -> None:
        print("\n" + "=" * 70)
        print("TIME SERIES METRIC SUMMARY TABLE")
        print("=" * 70)

        sections = {
            "Faithfulness (LIME | TimeSHAP | IG)": [
                ("AOPC ↑",           "LIME_AOPC",             "TimeSHAP_AOPC",             "IG_AOPC"),
                ("Comprehensiveness ↑","LIME_Comprehensiveness","TimeSHAP_Comprehensiveness","IG_Comprehensiveness"),
                ("Sufficiency ↓",    "LIME_Sufficiency",       "TimeSHAP_Sufficiency",       "IG_Sufficiency"),
                ("Insertion AUC ↑",  "LIME_InsertionAUC",      "TimeSHAP_InsertionAUC",      "IG_InsertionAUC"),
                ("Deletion AUC ↓",   "LIME_DeletionAUC",       "TimeSHAP_DeletionAUC",       "IG_DeletionAUC"),
            ],
            "Stability": [
                ("Rank Correlation ↑","LIME_RankCorrelation",  "TimeSHAP_RankCorrelation",  "IG_RankCorrelation"),
                ("Avg Sensitivity ↓", "LIME_AvgSensitivity",   "TimeSHAP_AvgSensitivity",   "IG_AvgSensitivity"),
            ],
            "Temporal Localisation": [
                ("Pointing Game ↑",  "LIME_PointingGame",      "TimeSHAP_PointingGame",      "IG_PointingGame"),
                ("Temporal IoU ↑",   "LIME_TemporalIoU",       "TimeSHAP_TemporalIoU",       "IG_TemporalIoU"),
            ],
            "Runtime (s/series)": [
                ("Mean Time ↓",      "LIME_MeanTime_s",        "TimeSHAP_MeanTime_s",        "IG_MeanTime_s"),
            ],
        }

        def _fmt(results, key):
            v = results.get(key)
            if v is None or (isinstance(v, float) and v != v):
                return "—"
            return f"{v:.4f}"

        fmt = "{:<30} {:>12} {:>12} {:>10}"
        print(fmt.format("Metric", "LIME", "TimeSHAP", "IG"))
        print("-" * 66)

        for section, items in sections.items():
            print(f"\n{section}")
            for row in items:
                label = row[0]
                vals  = [_fmt(results, k) for k in row[1:]]
                print(fmt.format(f"  {label}", *vals))


# ------------------------------------------------------------------
# Helper: detect QRS windows for test batch
# ------------------------------------------------------------------

def _detect_qrs_for_batch(series_batch: np.ndarray,
                           qrs_half_width: int = 10) -> dict:
    """
    Detect QRS windows for a batch of test series.

    Returns dict {i: (start, end)} keyed by batch index.
    """
    T = series_batch.shape[-1]
    windows = {}
    for i in range(len(series_batch)):
        series  = series_batch[i, 0]
        r_peak  = int(np.abs(series).argmax())
        start   = max(0, r_peak - qrs_half_width)
        end     = min(T - 1, r_peak + qrs_half_width)
        windows[i] = (start, end)
    return windows
