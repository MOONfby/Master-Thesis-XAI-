"""
Time series visualization — stacked ECG line + attribution heatmap.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def plot_ecg_attribution(
    series: np.ndarray,
    attributions: np.ndarray,
    segment_map: np.ndarray,
    predicted_class_name: str,
    confidence: float,
    method: str,
    instance_idx: int,
) -> plt.Figure:
    """
    Two-subplot stacked figure:
      top — ECG waveform with QRS region highlighted
      bottom — per-segment attribution bar (heatmap style)

    Parameters
    ----------
    series : np.ndarray, shape (1, 140)
    attributions : np.ndarray, shape (20,)  — segment-level
    segment_map : np.ndarray, shape (140,)  — segment index per timestep
    predicted_class_name : str
    confidence : float
    method : str
    instance_idx : int
    """
    series_1d = series.squeeze()        # (140,)
    T = len(series_1d)
    n_segments = len(attributions)
    t = np.arange(T)

    # Build timestep-level attribution for colour bar
    ts_attr = np.zeros(T, dtype=float)
    for seg_id, attr_val in enumerate(attributions):
        ts_attr[segment_map == seg_id] = attr_val

    # Normalise for colour mapping
    vmax = np.abs(ts_attr).max()
    ts_attr_norm = ts_attr / vmax if vmax > 0 else ts_attr

    # QRS approximate region highlight (timesteps 50–80)
    qrs_start, qrs_end = 50, 80

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 5),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # ── Top: ECG waveform ──────────────────────────────────────────
    ax_ecg = axes[0]
    ax_ecg.plot(t, series_1d, color="#2c3e50", linewidth=1.2, zorder=3)
    ax_ecg.axvspan(qrs_start, qrs_end, alpha=0.12, color="#e67e22",
                   label=f"QRS region ({qrs_start}–{qrs_end})")
    ax_ecg.set_ylabel("Amplitude")
    ax_ecg.set_title(
        f"ECG5000 — Instance {instance_idx} — "
        f"Predicted: {predicted_class_name} ({confidence*100:.1f}%)",
        fontsize=11, fontweight="bold",
    )
    ax_ecg.legend(fontsize=8, loc="upper right")
    ax_ecg.spines["top"].set_visible(False)
    ax_ecg.spines["right"].set_visible(False)

    # ── Bottom: attribution heatmap bar ───────────────────────────
    ax_attr = axes[1]
    cmap = plt.get_cmap("RdYlGn")
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    for seg_id in range(n_segments):
        mask = segment_map == seg_id
        seg_t = t[mask]
        if len(seg_t) == 0:
            continue
        color = cmap(norm(ts_attr_norm[mask][0]))
        ax_attr.bar(
            seg_t,
            np.ones(len(seg_t)),
            width=1.0,
            color=color,
            align="center",
            edgecolor="none",
        )

    ax_attr.set_yticks([])
    ax_attr.set_xlabel("Timestep")
    ax_attr.set_ylabel("Attribution", fontsize=8)
    ax_attr.set_title(f"{method} — segment attributions", fontsize=9)

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_attr, orientation="vertical",
                        fraction=0.02, pad=0.02)
    cbar.set_label("Attribution", fontsize=7)

    fig.tight_layout()
    return fig
