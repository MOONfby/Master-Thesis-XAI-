"""
Phase 1 Main Experiment Script
Thesis: "Bridging Technical and Regulatory Transparency" — Biying Feng, KTH 2026

Executes the full Phase 1 evaluation pipeline:
  1. Load and preprocess Adult Income dataset
  2. Train XGBoost classifier
  3. Initialize LIME, SHAP (TreeExplainer), and DiCE explainers
  4. Evaluate across all metrics: Faithfulness, Stability, CF quality, Runtime
  5. Generate comparison visualizations and save results

Usage:
    python run_phase1.py                        # full run
    python run_phase1.py --skip-cf              # skip counterfactuals (faster)
    python run_phase1.py --force-reload         # re-download data and retrain model

Output:
    results/phase1/metrics_summary.csv
    results/phase1/figures/
"""
import sys
import argparse
import numpy as np

from phase1.tabular.data_loader import load_adult_income
from phase1.tabular.model_trainer import train_model
from phase1.tabular.explainers import LIMEExplainer, SHAPExplainer, DiCEExplainer
from phase1.tabular.evaluation.evaluator import Phase1Evaluator
from phase1.tabular.visualization.plots import generate_all_plots


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1 XAI Evaluation")
    parser.add_argument("--skip-cf", action="store_true",
                        help="Skip counterfactual generation (faster testing)")
    parser.add_argument("--force-reload", action="store_true",
                        help="Force re-download of data and model retraining")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("PHASE 1: Multi-Method XAI Evaluation — Tabular Data")
    print("Dataset: Adult Income | Methods: LIME, SHAP, DiCE")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading dataset...")
    data = load_adult_income(force_reload=args.force_reload)

    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]
    feature_names = data["feature_names"]
    num_indices   = data["num_indices"]
    cat_indices   = data["cat_indices"]

    # ------------------------------------------------------------------
    # Step 2: Model
    # ------------------------------------------------------------------
    print("\n[2/5] Training XGBoost model...")
    model = train_model(X_train, y_train, X_test, y_test,
                        force_retrain=args.force_reload)

    # ------------------------------------------------------------------
    # Step 3: Initialize explainers
    # ------------------------------------------------------------------
    print("\n[3/5] Initializing explainers...")

    print("  Initializing LIME...")
    lime_exp = LIMEExplainer(
        model=model,
        X_train=X_train,
        cat_indices=cat_indices,
        feature_names=feature_names,
    )

    print("  Initializing SHAP (TreeExplainer)...")
    shap_exp = SHAPExplainer(
        model=model,
        X_train=X_train,
        feature_names=feature_names,
    )

    if not args.skip_cf:
        print("  Initializing DiCE (Counterfactual Explainer)...")
        dice_exp = DiCEExplainer(
            model=model,
            X_train_df=data["X_train_df"],
            y_train=y_train,
            feature_names=feature_names,
        )
    else:
        dice_exp = None
        print("  [Skipped] DiCE (--skip-cf flag set)")

    # ------------------------------------------------------------------
    # Step 4: Run evaluation
    # ------------------------------------------------------------------
    print("\n[4/5] Running evaluation...")
    evaluator = Phase1Evaluator(data, model, lime_exp, shap_exp, dice_exp)
    results = evaluator.run(skip_cf=args.skip_cf)
    evaluator.save_results(results)

    # ------------------------------------------------------------------
    # Step 5: Visualizations
    # ------------------------------------------------------------------
    print("\n[5/5] Generating visualizations...")

    # Recompute attributions for visualization (on eval subset)
    X_eval = evaluator.X_eval
    lime_attrs = lime_exp.explain_batch(X_eval)
    shap_attrs = shap_exp.explain_batch(X_eval)

    generate_all_plots(
        results=results,
        lime_attrs=lime_attrs,
        shap_attrs=shap_attrs,
        X_eval=X_eval,
        baseline=data["baseline_values"],
        predict_fn=model.predict_proba,
        feature_names=feature_names,
    )

    print("\n" + "=" * 60)
    print("Phase 1 complete.")
    print(f"Results: results/phase1/metrics_summary.csv")
    print(f"Figures: results/phase1/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
