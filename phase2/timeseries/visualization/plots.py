"""
Visualisation for Phase 2 Time Series XAI Evaluation.

Generates:
  1. Attribution heatmaps overlaid on ECG waveforms (per method, per class)
  2. Comparison bar chart of all metrics across methods
  3. Temporal attribution profile comparison (LIME vs TimeSHAP vs IG)

Mirrors phase1/image/visualization/plots.py structure.
Saves all figures to TS_FIGURES_DIR.
"""
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from phase2.timeseries.config import (
    TS_FIGURES_DIR, N_SEGMENTS_TS
)
from phase2.timeseries.utils import build_segment_map


def generate_all_ts_plots(data: dict, results: dict,
                           lime_attrs: np.ndarray,
                           timeshap_attrs: np.ndarray,
                           ig_attrs: np.ndarray,
                           n_examples: int = 5) -> None:
    """
    Generate and save all Phase 2 figures.

    Parameters
    ----------
    data : dict — output of load_ecg5000()
    results : dict — output of evaluator.run()
    lime_attrs, timeshap_attrs, ig_attrs : (N, S) attribution arrays
    n_examples : int — number of example series to plot
    """
    TS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X_eval = data["X_test"][:n_examples]
    y_eval = data["y_test"][:n_examples]
    segment_map = build_segment_map()

    print("  Generating attribution heatmap overlays...")
    _plot_attribution_overlays(X_eval, y_eval, segment_map,
                                lime_attrs[:n_examples],
                                timeshap_attrs[:n_examples],
                                ig_attrs[:n_examples])

    print("  Generating metric comparison bar chart...")
    _plot_metric_comparison(results)

    print("  Generating temporal profile comparison...")
    _plot_temporal_profiles(X_eval[:3], y_eval[:3], segment_map,
                             lime_attrs[:3], timeshap_attrs[:3], ig_attrs[:3])

    print(f"  All figures saved to {TS_FIGURES_DIR}")


def _plot_attribution_overlays(X: np.ndarray, y: np.ndarray,
                                segment_map: np.ndarray,
                                lime_a: np.ndarray,
                                timeshap_a: np.ndarray,
                                ig_a: np.ndarray) -> None:
    """
    For each example: plot raw ECG + colour-coded attribution heatmap
    for each of the 3 methods in subplots.
    """
    N = len(X)
    T = X.shape[-1]
    t_axis = np.arange(T)

    for i in range(N):
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"ECG5000 — Sample {i} (Class {y[i]})",
                     fontsize=12, fontweight="bold")

        method_data = [
            ("LIME-TS",    lime_a[i]),
            ("TimeSHAP",   timeshap_a[i]),
            ("Int.Grads",  ig_a[i]),
        ]

        series = X[i, 0]   # (T,)

        for ax, (name, attrs) in zip(axes, method_data):
            # Broadcast segment attributions to timestep level
            ts_attr = np.zeros(T)
            for s, a in enumerate(attrs):
                ts_attr[segment_map == s] = a

            # Normalise for colour map
            a_min, a_max = ts_attr.min(), ts_attr.max()
            if a_max > a_min:
                ts_norm = (ts_attr - a_min) / (a_max - a_min)
            else:
                ts_norm = np.zeros_like(ts_attr)

            # Plot ECG waveform coloured by attribution
            for t in range(T - 1):
                ax.plot([t, t + 1], [series[t], series[t + 1]],
                        color=cm.RdYlGn(ts_norm[t]), linewidth=1.5)

            ax.set_ylabel(f"{name}\nattribution", fontsize=9)
            ax.set_ylim(series.min() - 0.5, series.max() + 0.5)

            # Colour bar
            sm = plt.cm.ScalarMappable(cmap="RdYlGn",
                                        norm=plt.Normalize(a_min, a_max))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, orientation="vertical", pad=0.01,
                         label="Attribution")

        axes[-1].set_xlabel("Timestep", fontsize=10)
        plt.tight_layout()
        save_path = TS_FIGURES_DIR / f"attribution_overlay_sample{i}.png"
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()


def _plot_metric_comparison(results: dict) -> None:
    """
    Grouped bar chart comparing LIME, TimeSHAP, IG across all faithfulness metrics.
    """
    metrics = ["AOPC", "Comprehensiveness", "InsertionAUC"]
    methods = ["LIME", "TimeSHAP", "IG"]
    colors  = ["#2196F3", "#FF9800", "#4CAF50"]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    for j, (method, color) in enumerate(zip(methods, colors)):
        values = [results.get(f"{method}_{m}", 0.0) or 0.0 for m in metrics]
        ax.bar(x + j * width, values, width, label=method, color=color, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Faithfulness Metrics: LIME-TS vs TimeSHAP vs Integrated Gradients",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(TS_FIGURES_DIR / "metric_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()


def _plot_temporal_profiles(X: np.ndarray, y: np.ndarray,
                             segment_map: np.ndarray,
                             lime_a: np.ndarray,
                             timeshap_a: np.ndarray,
                             ig_a: np.ndarray) -> None:
    """
    Side-by-side temporal attribution profiles for 3 examples.
    Shows the ECG waveform (top) and attribution bars per segment (bottom).
    """
    N = len(X)
    T = X.shape[-1]

    for i in range(N):
        fig, axes = plt.subplots(2, 1, figsize=(12, 6),
                                  gridspec_kw={"height_ratios": [2, 1]})
        fig.suptitle(f"Temporal Attribution Profile — Sample {i} (Class {y[i]})",
                     fontsize=12, fontweight="bold")

        # Top: ECG waveform
        axes[0].plot(X[i, 0], color="black", linewidth=1.2, label="ECG")
        axes[0].set_ylabel("Amplitude", fontsize=10)
        axes[0].legend(fontsize=9)
        axes[0].grid(alpha=0.3)

        # Bottom: bar chart of segment attributions
        segs = np.arange(N_SEGMENTS_TS)
        w = 0.25
        axes[1].bar(segs - w, lime_a[i], w,     label="LIME",     color="#2196F3", alpha=0.85)
        axes[1].bar(segs,     timeshap_a[i], w,  label="TimeSHAP", color="#FF9800", alpha=0.85)
        axes[1].bar(segs + w, ig_a[i], w,        label="IG",       color="#4CAF50", alpha=0.85)
        axes[1].axhline(0, color="black", linewidth=0.5)
        axes[1].set_xlabel("Temporal Segment Index", fontsize=10)
        axes[1].set_ylabel("Attribution", fontsize=10)
        axes[1].legend(fontsize=9)
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(TS_FIGURES_DIR / f"temporal_profile_sample{i}.png",
                    dpi=120, bbox_inches="tight")
        plt.close()
