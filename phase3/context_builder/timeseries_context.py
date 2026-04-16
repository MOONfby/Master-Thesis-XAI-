"""
Context builder for Time Series (ECG5000) explanations.

Produces a structured text block injected into LLM prompts.
"""
from __future__ import annotations

import numpy as np

from phase3.backend.explainer_runner import ExplanationResult


# ECG physiological landmark ranges (approximate, based on typical 140-sample ECG windows)
_QRS_REGION_LABEL = "QRS complex (ventricular depolarisation)"
_P_WAVE_LABEL     = "P-wave region (atrial depolarisation)"
_T_WAVE_LABEL     = "T-wave region (ventricular repolarisation)"


def build_timeseries_context(result: ExplanationResult, metrics_str: str) -> str:
    """
    Build a grounded context block for a time series explanation.

    Parameters
    ----------
    result : ExplanationResult
    metrics_str : str
        Pre-formatted metrics string from format_metrics_for_prompt().

    Returns
    -------
    str — the full context block to inject into the LLM prompt.
    """
    attrs = result.attributions         # (20,)
    seg_map = result.segment_map        # (140,)
    probs = result.all_probabilities    # (5,)
    n_segments = len(attrs)

    # Top segments by absolute attribution
    ranked = np.argsort(np.abs(attrs))[::-1]
    top_n = min(5, n_segments)

    top_lines = []
    for rank, seg_id in enumerate(ranked[:top_n]):
        start = int(np.where(seg_map == seg_id)[0].min())
        end   = int(np.where(seg_map == seg_id)[0].max())
        sign  = "+" if attrs[seg_id] >= 0 else ""
        marker = "  ← highest" if rank == 0 else ""
        top_lines.append(
            f"  Segment {seg_id} (timesteps {start}–{end}): "
            f"{sign}{attrs[seg_id]:.3f}{marker}"
        )

    # Identify QRS region from segment map
    # Approximate: QRS typically spans timesteps 50–80 in ECG5000 normalised series
    # We detect by asking which segments overlap with the central 30-timestep window
    qrs_approx_start, qrs_approx_end = 50, 80
    qrs_segments = sorted({
        int(seg_map[t])
        for t in range(qrs_approx_start, min(qrs_approx_end, len(seg_map)))
    })
    qrs_attr_sum = float(np.sum(np.abs(attrs[[s for s in qrs_segments if s < len(attrs)]])))

    # Probability table
    ecg_names = [
        "Normal", "R-on-T PVC", "PVC",
        "Supra-ventricular Ectopic Beat", "Unclassifiable",
    ]
    prob_parts = []
    for i, name in enumerate(ecg_names):
        if i < len(probs):
            prob_parts.append(f"{name}={probs[i]:.3f}")
    prob_str = ", ".join(prob_parts)

    seg_per_segment = 140 // n_segments if n_segments > 0 else 7

    lines = [
        "=== INSTANCE DATA ===",
        f"Dataset: ECG5000  |  Model: InceptionTime  |  Instance index: {result.instance_index}",
        f"Predicted class: {result.predicted_class_name} (confidence: {result.confidence*100:.1f}%)",
        f"All probabilities: {prob_str}",
        "",
        f"=== ATTRIBUTION ANALYSIS ({result.method}, {n_segments} temporal segments × {seg_per_segment} timesteps) ===",
        f"Top {top_n} segments by |attribution|:",
    ]
    lines.extend(top_lines)
    lines += [
        f"Approximate QRS region (timesteps {qrs_approx_start}–{qrs_approx_end}): "
        f"segments {qrs_segments}",
        f"QRS attribution concentration: {qrs_attr_sum:.3f} "
        f"(sum of |attribution| in QRS region)",
        "",
        f"=== BENCHMARK METRICS (200 test instances) ===",
        metrics_str,
    ]

    return "\n".join(lines)
