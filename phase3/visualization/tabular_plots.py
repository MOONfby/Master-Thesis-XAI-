"""
Tabular visualization — horizontal bar chart of feature attributions.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phase3.config import ADULT_FEATURE_NAMES


def plot_feature_attribution(
    attributions: np.ndarray,
    feature_names: list[str] = ADULT_FEATURE_NAMES,
    title: str = "Feature Attribution",
) -> plt.Figure:
    """
    Horizontal bar chart sorted by |attribution|, green for positive, red for negative.
    """
    attrs = np.array(attributions)
    names = list(feature_names)

    # Sort by absolute value
    order = np.argsort(np.abs(attrs))
    attrs_sorted = attrs[order]
    names_sorted = [names[i] for i in order]

    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in attrs_sorted]

    fig, ax = plt.subplots(figsize=(7, max(4, len(attrs_sorted) * 0.4)))
    bars = ax.barh(names_sorted, attrs_sorted, color=colors, edgecolor="none")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Attribution score")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)

    # Value labels on bars
    for bar, val in zip(bars, attrs_sorted):
        ha = "left" if val >= 0 else "right"
        offset = 0.001 if val >= 0 else -0.001
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha=ha, fontsize=8)

    fig.tight_layout()
    return fig
