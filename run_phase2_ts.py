"""
Phase 2: XAI Evaluation on Time Series Data.

Dataset  : ECG5000 (UCR archive) — 5-class ECG arrhythmia classification
Model    : InceptionTime (1D-CNN)
Methods  : LIME-TS, TimeSHAP, Integrated Gradients

Usage
-----
    python run_phase2_ts.py                      # full evaluation
    python run_phase2_ts.py --skip-localization  # skip temporal QRS metrics
    python run_phase2_ts.py --force-reload       # re-download + retrain
    python run_phase2_ts.py --no-plots           # skip figure generation

Outputs
-------
    results/phase2_ts/metrics_summary.csv
    results/phase2_ts/figures/*.png
"""
import argparse

from phase2.timeseries.data_loader import load_ecg5000
from phase2.timeseries.model_trainer import (
    load_or_train_inceptiontime, make_predict_fn
)
from phase2.timeseries.utils import build_segment_map, compute_baseline_series
from phase2.timeseries.explainers.lime_ts_explainer import LIMETSExplainer
from phase2.timeseries.explainers.timeshap_explainer import TimeSHAPExplainer
from phase2.timeseries.explainers.ig_explainer import IntegratedGradientsExplainer
from phase2.timeseries.evaluation.evaluator import Phase2TSEvaluator
from phase2.timeseries.visualization.plots import generate_all_ts_plots


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Time Series XAI Evaluation"
    )
    parser.add_argument("--skip-localization", action="store_true",
                        help="Skip temporal QRS localisation metrics")
    parser.add_argument("--force-reload", action="store_true",
                        help="Re-download data and retrain model")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip figure generation")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 2: TIME SERIES XAI EVALUATION")
    print("=" * 60)
    print("\n[1/5] Loading ECG5000 dataset...")
    data = load_ecg5000(force_reload=args.force_reload)

    # ------------------------------------------------------------------
    # 2. Load / train InceptionTime
    # ------------------------------------------------------------------
    print("\n[2/5] Loading / training InceptionTime...")
    model      = load_or_train_inceptiontime(data, force_reload=args.force_reload)
    predict_fn = make_predict_fn(model)

    # Quick accuracy check
    import numpy as np
    probs = predict_fn(data["X_test"])
    acc   = (probs.argmax(axis=1) == data["y_test"]).mean()
    print(f"  Test accuracy: {acc:.4f}")

    # ------------------------------------------------------------------
    # 3. Initialise explainers
    # ------------------------------------------------------------------
    print("\n[3/5] Initialising explainers...")
    baseline = data["baseline_series"]

    lime_exp = LIMETSExplainer(predict_fn, baseline)
    print("  LIME-TS ready.")

    timeshap_exp = TimeSHAPExplainer(predict_fn, baseline)
    print("  TimeSHAP ready.")

    ig_exp = IntegratedGradientsExplainer(model, baseline)
    print("  Integrated Gradients ready.")

    # ------------------------------------------------------------------
    # 4. Run evaluation
    # ------------------------------------------------------------------
    print("\n[4/5] Running evaluation...")
    evaluator = Phase2TSEvaluator(
        data       = data,
        model      = model,
        predict_fn = predict_fn,
        lime_explainer     = lime_exp,
        timeshap_explainer = timeshap_exp,
        ig_explainer       = ig_exp,
    )

    results = evaluator.run(skip_localization=args.skip_localization)
    evaluator.save_results(results)

    # ------------------------------------------------------------------
    # 5. Generate visualisations
    # ------------------------------------------------------------------
    if not args.no_plots:
        print("\n[5/5] Generating figures...")
        generate_all_ts_plots(
            data           = data,
            results        = results,
            lime_attrs     = evaluator.lime_attrs,
            timeshap_attrs = evaluator.timeshap_attrs,
            ig_attrs       = evaluator.ig_attrs,
        )
    else:
        print("\n[5/5] Skipping figure generation (--no-plots).")

    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETE")
    print(f"Results: results/phase2_ts/metrics_summary.csv")
    print(f"Figures: results/phase2_ts/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
