"""
Visualization for Phase 1 XAI evaluation results.

Generates publication-quality figures comparing LIME, SHAP, and DiCE:

1. metrics_comparison.png   — grouped bar chart of all metrics
2. runtime_comparison.png   — runtime comparison (log scale)
3. shap_importance.png      — SHAP feature importance (beeswarm plot)
4. lime_importance.png      — LIME feature importance (bar plot, averaged)
5. cf_example.png           — example counterfactual explanation table
6. perturbation_curves.png  — AOPC perturbation curves for LIME vs SHAP
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from phase1.tabular.config import FIGURES_DIR, FEATURE_NAMES

# Use a consistent, clean style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})
COLORS = {"LIME": "#4C72B0", "SHAP": "#DD8452", "CF": "#55A868"}


# -----------------------------------------------------------------------
# 1. Metrics comparison bar chart
# -----------------------------------------------------------------------

def plot_metrics_comparison(results: dict, save: bool = True) -> None:
    """
    Grouped bar chart comparing LIME vs SHAP across faithfulness and stability metrics.
    Counterfactual metrics shown separately.
    """
    faithfulness_metrics = {
        "AOPC ↑":            ("LIME_AOPC",               "SHAP_AOPC"),
        "Comprehen. ↑":      ("LIME_Comprehensiveness",   "SHAP_Comprehensiveness"),
        "Sufficiency ↓":     ("LIME_Sufficiency",         "SHAP_Sufficiency"),
    }
    stability_metrics = {
        "Rank Corr. ↑":      ("LIME_RankCorrelation",     "SHAP_RankCorrelation"),
        "Avg Sensitivity ↓": ("LIME_AvgSensitivity",       "SHAP_AvgSensitivity"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Phase 1: XAI Method Comparison on Adult Income Dataset", fontsize=14)

    for ax, metric_dict, title in zip(
        axes,
        [faithfulness_metrics, stability_metrics],
        ["Faithfulness Metrics", "Stability Metrics"],
    ):
        labels = list(metric_dict.keys())
        lime_vals = [results.get(v[0], 0) for v in metric_dict.values()]
        shap_vals = [results.get(v[1], 0) for v in metric_dict.values()]

        x = np.arange(len(labels))
        width = 0.35

        bars_lime = ax.bar(x - width / 2, lime_vals, width, label="LIME",
                           color=COLORS["LIME"], alpha=0.85)
        bars_shap = ax.bar(x + width / 2, shap_vals, width, label="SHAP",
                           color=COLORS["SHAP"], alpha=0.85)

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.legend()
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_ylabel("Score")

        # Value labels on bars
        for bar in list(bars_lime) + list(bars_shap):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.001,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "metrics_comparison.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()


def plot_cf_metrics(results: dict, save: bool = True) -> None:
    """Bar chart for counterfactual-specific metrics."""
    cf_metrics = {
        "Validity ↑": results.get("CF_Validity", 0),
        "Sparsity ↓":  results.get("CF_Sparsity", 0),
        "Proximity L1 ↓": results.get("CF_Proximity_L1", 0),
        "Diversity ↑": results.get("CF_Diversity", 0),
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(cf_metrics.keys())
    vals = [v if v is not None and not np.isnan(v) else 0 for v in cf_metrics.values()]

    bars = ax.bar(labels, vals, color=COLORS["CF"], alpha=0.85, width=0.5)
    ax.set_title("Counterfactual Explanation Metrics (DiCE)")
    ax.set_ylabel("Score")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "cf_metrics.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()


# -----------------------------------------------------------------------
# 2. Runtime comparison (log scale)
# -----------------------------------------------------------------------

def plot_runtime_comparison(results: dict, save: bool = True) -> None:
    """Bar chart of mean explanation time per instance (log scale)."""
    methods = ["LIME", "SHAP", "CF (DiCE)"]
    times = [
        results.get("LIME_MeanTime_s", 0),
        results.get("SHAP_MeanTime_s", 0),
        results.get("CF_MeanTime_s", 0),
    ]
    stds = [
        results.get("LIME_StdTime_s", 0),
        results.get("SHAP_StdTime_s", 0),
        results.get("CF_StdTime_s", 0),
    ]
    colors = [COLORS["LIME"], COLORS["SHAP"], COLORS["CF"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(methods, times, yerr=stds, color=colors, alpha=0.85,
                  capsize=5, width=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Time per explanation (seconds, log scale)")
    ax.set_title("Runtime Comparison: LIME vs SHAP vs DiCE")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, t * 1.3,
                f"{t:.4f}s", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "runtime_comparison.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()


# -----------------------------------------------------------------------
# 3. SHAP summary / feature importance
# -----------------------------------------------------------------------

def plot_shap_importance(shap_attributions: np.ndarray,
                         feature_names: list[str] = None,
                         n_top: int = 13,
                         save: bool = True) -> None:
    """
    Bar plot of mean |SHAP value| per feature (global feature importance).
    """
    feature_names = feature_names or FEATURE_NAMES
    mean_abs = np.abs(shap_attributions).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:n_top]

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(n_top)[::-1]
    ax.barh(y_pos, mean_abs[top_idx], color=COLORS["SHAP"], alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in top_idx])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"SHAP Global Feature Importance (Top {n_top})")

    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "shap_importance.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()


def plot_lime_importance(lime_attributions: np.ndarray,
                         feature_names: list[str] = None,
                         n_top: int = 13,
                         save: bool = True) -> None:
    """
    Bar plot of mean |LIME attribution| per feature.
    """
    feature_names = feature_names or FEATURE_NAMES
    mean_abs = np.abs(lime_attributions).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:n_top]

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(n_top)[::-1]
    ax.barh(y_pos, mean_abs[top_idx], color=COLORS["LIME"], alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in top_idx])
    ax.set_xlabel("Mean |LIME attribution|")
    ax.set_title(f"LIME Global Feature Importance (Top {n_top})")

    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "lime_importance.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()


# -----------------------------------------------------------------------
# 4. SHAP beeswarm (requires shap library)
# -----------------------------------------------------------------------

def plot_shap_beeswarm(shap_attributions: np.ndarray, X_eval: np.ndarray,
                       feature_names: list[str] = None,
                       n_top: int = 13, save: bool = True) -> None:
    """SHAP beeswarm plot showing attribution distribution per feature."""
    try:
        import shap
        feature_names = feature_names or FEATURE_NAMES
        expl = shap.Explanation(
            values=shap_attributions,
            data=X_eval,
            feature_names=feature_names,
        )
        fig, ax = plt.subplots(figsize=(9, 7))
        shap.plots.beeswarm(expl, max_display=n_top, show=False)
        plt.title("SHAP Beeswarm Plot (Top Features)")
        plt.tight_layout()
        if save:
            path = FIGURES_DIR / "shap_beeswarm.png"
            plt.savefig(path, bbox_inches="tight")
            print(f"Saved: {path}")
        plt.close()
    except Exception as e:
        print(f"  [Warning] Beeswarm plot failed: {e}")


# -----------------------------------------------------------------------
# 5. Perturbation curves (AOPC visualization)
# -----------------------------------------------------------------------

def plot_perturbation_curves(
    predict_fn,
    X_sample: np.ndarray,
    lime_attrs: np.ndarray,
    shap_attrs: np.ndarray,
    baseline: np.ndarray,
    n_steps: int = 13,
    save: bool = True,
) -> None:
    """
    Plot mean prediction drop as features are progressively removed (AOPC curve).
    Visually shows how quickly LIME and SHAP identify important features.
    """
    def _curve(attrs, X, baseline, n_steps):
        drops = np.zeros(n_steps + 1)
        for i in range(len(X)):
            orig_prob = predict_fn(X[[i]])[0, 1]
            sorted_idx = np.argsort(np.abs(attrs[i]))[::-1]
            x_pert = X[i].copy()
            drops[0] += orig_prob
            for k in range(n_steps):
                x_pert[sorted_idx[k]] = baseline[sorted_idx[k]]
                drops[k + 1] += predict_fn(x_pert[np.newaxis, :])[0, 1]
        return drops / len(X)

    n_sample = min(50, len(X_sample))
    X_s = X_sample[:n_sample]
    l_s = lime_attrs[:n_sample]
    sh_s = shap_attrs[:n_sample]

    lime_curve = _curve(l_s, X_s, baseline, n_steps)
    shap_curve = _curve(sh_s, X_s, baseline, n_steps)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(n_steps + 1), lime_curve, marker="o", color=COLORS["LIME"],
            label="LIME", linewidth=2)
    ax.plot(range(n_steps + 1), shap_curve, marker="s", color=COLORS["SHAP"],
            label="SHAP", linewidth=2)
    ax.set_xlabel("Number of features removed (most important first)")
    ax.set_ylabel("Mean predicted probability (class >50K)")
    ax.set_title("Perturbation Curve (AOPC): Prediction Drop vs Features Removed")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "perturbation_curves.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()


# -----------------------------------------------------------------------
# 6. Example explanation comparison for one instance
# -----------------------------------------------------------------------

def plot_example_explanation(
    instance: np.ndarray,
    lime_attr: np.ndarray,
    shap_attr: np.ndarray,
    cf_df: pd.DataFrame | None,
    feature_names: list[str] = None,
    n_top: int = 10,
    save: bool = True,
) -> None:
    """
    Side-by-side bar chart showing LIME vs SHAP attributions for one instance,
    plus a table with the counterfactual changes.
    """
    feature_names = feature_names or FEATURE_NAMES
    top_idx = np.argsort(np.abs(shap_attr))[::-1][:n_top]
    feat_labels = [feature_names[i] for i in top_idx]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Example Instance: LIME vs SHAP Feature Attribution", fontsize=13)

    for ax, attr, color, title in zip(
        axes,
        [lime_attr[top_idx], shap_attr[top_idx]],
        [COLORS["LIME"], COLORS["SHAP"]],
        ["LIME", "SHAP"],
    ):
        bar_colors = [color if v >= 0 else "#e74c3c" for v in attr]
        ax.barh(range(n_top), attr[::-1], color=bar_colors[::-1], alpha=0.85)
        ax.set_yticks(range(n_top))
        ax.set_yticklabels(feat_labels[::-1])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Attribution value")
        ax.set_title(f"{title} Attributions")

    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "example_explanation.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()


# -----------------------------------------------------------------------
# Generate all plots
# -----------------------------------------------------------------------

def generate_all_plots(
    results: dict,
    lime_attrs: np.ndarray,
    shap_attrs: np.ndarray,
    X_eval: np.ndarray,
    baseline: np.ndarray,
    predict_fn,
    feature_names: list[str] = None,
) -> None:
    """Run all visualization functions."""
    print("\n--- Generating Visualizations ---")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_metrics_comparison(results)
    plot_cf_metrics(results)
    plot_runtime_comparison(results)
    plot_shap_importance(shap_attrs, feature_names)
    plot_lime_importance(lime_attrs, feature_names)
    plot_shap_beeswarm(shap_attrs, X_eval, feature_names)
    plot_perturbation_curves(predict_fn, X_eval, lime_attrs, shap_attrs, baseline)

    # Example for the first test instance
    plot_example_explanation(
        X_eval[0], lime_attrs[0], shap_attrs[0], cf_df=None, feature_names=feature_names
    )
    print(f"\nAll figures saved to: {FIGURES_DIR}")
