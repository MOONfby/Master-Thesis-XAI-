"""
Phase 1 Image XAI Evaluation Script
Thesis: "Bridging Technical and Regulatory Transparency" — Biying Feng, KTH 2026

Evaluates LIME-image and GradientSHAP on CUB-200-2011 with ResNet-50.

Metrics: Faithfulness (AOPC, Comprehensiveness, Sufficiency, Insertion/Deletion AUC),
         Stability (Rank Correlation, Average Sensitivity),
         Localisation (Pointing Game, Segmentation IoU),
         Runtime

Usage:
    python run_phase1_image.py                     # full evaluation
    python run_phase1_image.py --skip-localization # skip GT-based localization metrics
    python run_phase1_image.py --force-reload      # re-download data and retrain model

Output:
    results/phase1_image/metrics_summary.csv
    results/phase1_image/figures/
"""
import sys
import argparse

from phase1.image.data_loader import load_cub200
from phase1.image.model_trainer import load_or_finetune_resnet50, get_predict_fn
from phase1.image.explainers import LIMEImageExplainer, GradientSHAPExplainer
from phase1.image.evaluation.evaluator import Phase1ImageEvaluator
from phase1.image.visualization.plots import generate_all_image_plots


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1 Image XAI Evaluation")
    parser.add_argument("--skip-localization", action="store_true",
                        help="Skip Pointing Game / IoU (CUB-200 GT masks required)")
    parser.add_argument("--force-reload", action="store_true",
                        help="Force re-download of CUB-200 and model retraining")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("PHASE 1: Multi-Method XAI Evaluation — Image Data")
    print("Dataset: CUB-200-2011 | Methods: LIME-image, GradientSHAP")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading CUB-200-2011 dataset...")
    data = load_cub200(force_reload=args.force_reload)
    print(f"  Train: {len(data['X_train'])} images | "
          f"Test: {len(data['X_test'])} images | "
          f"Classes: {len(data['class_names'])}")

    # ------------------------------------------------------------------
    # Step 2: Model
    # ------------------------------------------------------------------
    print("\n[2/5] Loading / fine-tuning ResNet-50...")
    model      = load_or_finetune_resnet50(data, force_retrain=args.force_reload)
    predict_fn = get_predict_fn(model)

    # ------------------------------------------------------------------
    # Step 3: Explainers
    # ------------------------------------------------------------------
    print("\n[3/5] Initialising explainers...")

    print("  Initialising LIME-image...")
    lime_exp = LIMEImageExplainer(
        predict_fn=predict_fn,
        baseline_image=data["baseline_image"],
    )

    print("  Initialising GradientSHAP...")
    gradshap_exp = GradientSHAPExplainer(
        model=model,
        background_images=data["background_images"],
    )

    # ------------------------------------------------------------------
    # Step 4: Evaluation
    # ------------------------------------------------------------------
    print("\n[4/5] Running evaluation...")
    evaluator = Phase1ImageEvaluator(
        data=data,
        model=model,
        predict_fn=predict_fn,
        lime_explainer=lime_exp,
        gradshap_explainer=gradshap_exp,
    )
    results = evaluator.run(skip_localization=args.skip_localization)
    evaluator.save_results(results)

    # ------------------------------------------------------------------
    # Step 5: Visualisations
    # ------------------------------------------------------------------
    print("\n[5/5] Generating visualisations...")
    generate_all_image_plots(results, evaluator)

    print("\n" + "=" * 60)
    print("Phase 1 image evaluation complete.")
    print("Results : results/phase1_image/metrics_summary.csv")
    print("Figures : results/phase1_image/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
