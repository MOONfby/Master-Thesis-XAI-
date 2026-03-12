"""
Visualisation for Phase 1 Image XAI Evaluation.

Generates:
  1. metrics_comparison.png  — bar chart: Faithfulness & Stability (LIME vs GradSHAP)
  2. faithfulness_curves.png — Insertion/Deletion AUC curves
  3. runtime_comparison.png  — per-image time (log scale)
  4. saliency_examples.png   — side-by-side: image | LIME overlay | GradSHAP overlay
  5. localization.png        — Pointing Game & Seg IoU bar chart (if available)

All figures saved to IMAGE_FIGURES_DIR.
Uses matplotlib Agg backend (headless server compatible).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

from phase1.image.config import IMAGE_FIGURES_DIR, N_SEGMENTS
from phase1.image.data_loader import unnormalise


_LIME_COLOR    = "#4C72B0"
_GRADSHAP_COLOR = "#DD8452"

COLORS = {
    "LIME":     _LIME_COLOR,
    "GradSHAP": _GRADSHAP_COLOR,
}


def generate_all_image_plots(results: dict,
                              evaluator,
                              save: bool = True) -> None:
    """
    Generate all Phase 1 image visualisation figures.

    Parameters
    ----------
    results : dict — output of Phase1ImageEvaluator.run()
    evaluator : Phase1ImageEvaluator — for accessing stored attributions / images
    save : bool
    """
    IMAGE_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_metrics_comparison(results, save=save)
    plot_runtime_comparison(results, save=save)

    if evaluator.lime_attrs is not None and evaluator.gradshap_attrs is not None:
        plot_saliency_examples(
            evaluator.X_eval, evaluator.X_eval_raw,
            evaluator.segment_maps,
            evaluator.lime_attrs, evaluator.gradshap_attrs,
            n_examples=4, save=save
        )

    has_loc = "LIME_PointingGame" in results
    if has_loc:
        plot_localization(results, save=save)

    print(f"  Figures saved to: {IMAGE_FIGURES_DIR}")


# ------------------------------------------------------------------
# 1. Metrics comparison bar chart
# ------------------------------------------------------------------

def plot_metrics_comparison(results: dict, save: bool = True) -> None:
    metrics = [
        ("AOPC ↑",              "AOPC"),
        ("Comprehensiveness ↑", "Comprehensiveness"),
        ("Sufficiency ↓",       "Sufficiency"),
        ("Insertion AUC ↑",     "InsertionAUC"),
        ("Deletion AUC ↓",      "DeletionAUC"),
        ("Rank Corr ↑",         "RankCorrelation"),
        ("Avg Sensitivity ↓",   "AvgSensitivity"),
    ]

    labels      = [m[0] for m in metrics]
    lime_vals   = [results.get(f"LIME_{m[1]}", float("nan")) for m in metrics]
    shap_vals   = [results.get(f"GradSHAP_{m[1]}", float("nan")) for m in metrics]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, lime_vals, width, label="LIME-image", color=_LIME_COLOR)
    ax.bar(x + width/2, shap_vals, width, label="GradientSHAP", color=_GRADSHAP_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Phase 1 Image — Faithfulness & Stability Metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save:
        path = IMAGE_FIGURES_DIR / "metrics_comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)


# ------------------------------------------------------------------
# 2. Runtime comparison
# ------------------------------------------------------------------

def plot_runtime_comparison(results: dict, save: bool = True) -> None:
    methods = ["LIME-image", "GradientSHAP"]
    keys    = ["LIME_MeanTime_s", "GradSHAP_MeanTime_s"]
    errs    = ["LIME_StdTime_s",  "GradSHAP_StdTime_s"]
    colors  = [_LIME_COLOR, _GRADSHAP_COLOR]

    means = [results.get(k, float("nan")) for k in keys]
    stds  = [results.get(k, 0.0) for k in errs]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(methods, means, yerr=stds, color=colors,
                  capsize=5, edgecolor="white")
    ax.set_yscale("log")
    ax.set_ylabel("Time per image (s, log scale)")
    ax.set_title("Phase 1 Image — Runtime per Image")
    ax.grid(axis="y", alpha=0.3)

    for bar, mean in zip(bars, means):
        if not np.isnan(mean):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.05,
                    f"{mean:.3f}s", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()

    if save:
        path = IMAGE_FIGURES_DIR / "runtime_comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)


# ------------------------------------------------------------------
# 3. Saliency overlay examples
# ------------------------------------------------------------------

def plot_saliency_examples(images_norm, images_raw, segment_maps,
                            lime_attrs, gradshap_attrs,
                            n_examples: int = 4, save: bool = True) -> None:
    """
    Side-by-side: original image | LIME overlay | GradSHAP overlay
    for n_examples test images.
    """
    n = min(n_examples, len(images_raw))
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None]

    col_titles = ["Original", "LIME-image", "GradientSHAP"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=10)

    for i in range(n):
        img_show = images_raw[i].transpose(1, 2, 0)   # (H,W,3) in [0,1]

        axes[i, 0].imshow(img_show)
        axes[i, 0].axis("off")

        for j, (attrs, cmap) in enumerate(
            [(lime_attrs[i], "RdBu_r"), (gradshap_attrs[i], "RdBu_r")]
        ):
            heatmap = _attrs_to_heatmap(attrs, segment_maps[i])
            axes[i, j + 1].imshow(img_show, alpha=0.6)
            axes[i, j + 1].imshow(heatmap, cmap=cmap, alpha=0.5,
                                    vmin=-np.abs(heatmap).max(),
                                    vmax=np.abs(heatmap).max())
            axes[i, j + 1].axis("off")

    plt.tight_layout()
    if save:
        path = IMAGE_FIGURES_DIR / "saliency_examples.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)


# ------------------------------------------------------------------
# 4. Localisation bar chart
# ------------------------------------------------------------------

def plot_localization(results: dict, save: bool = True) -> None:
    metrics = [("Pointing Game ↑", "PointingGame"),
               ("Seg IoU ↑",       "SegIoU")]

    labels    = [m[0] for m in metrics]
    lime_vals = [results.get(f"LIME_{m[1]}", float("nan")) for m in metrics]
    shap_vals = [results.get(f"GradSHAP_{m[1]}", float("nan")) for m in metrics]

    x, width = np.arange(len(labels)), 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width/2, lime_vals, width, label="LIME-image", color=_LIME_COLOR)
    ax.bar(x + width/2, shap_vals, width, label="GradientSHAP", color=_GRADSHAP_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Phase 1 Image — Localisation Metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save:
        path = IMAGE_FIGURES_DIR / "localization.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _attrs_to_heatmap(attrs: np.ndarray, segment_map: np.ndarray) -> np.ndarray:
    """Broadcast (S,) superpixel attributions to (H,W) pixel heatmap."""
    H, W = segment_map.shape
    heatmap = np.zeros((H, W), dtype=np.float64)
    for s, a in enumerate(attrs):
        heatmap[segment_map == s] = a
    return heatmap
