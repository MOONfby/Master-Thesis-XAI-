"""
Context builder for Image (CUB-200-2011) explanations.
"""
from __future__ import annotations

import numpy as np

from phase3.backend.explainer_runner import ExplanationResult


def build_image_context(result: ExplanationResult, metrics_str: str) -> str:
    """
    Build a grounded context block for an image explanation.

    Parameters
    ----------
    result : ExplanationResult
    metrics_str : str

    Returns
    -------
    str
    """
    attrs = result.attributions      # (50,)
    seg_map = result.segment_map     # (H, W)
    probs = result.all_probabilities

    n_segments = len(attrs)
    ranked = np.argsort(np.abs(attrs))[::-1]
    top_n = min(5, n_segments)

    # Compute approximate spatial location for top segments
    H, W = seg_map.shape

    top_lines = []
    for rank, seg_id in enumerate(ranked[:top_n]):
        pixels = np.argwhere(seg_map == seg_id)
        if len(pixels) == 0:
            location = "unknown"
        else:
            cy = float(pixels[:, 0].mean()) / H
            cx = float(pixels[:, 1].mean()) / W
            location = _spatial_label(cy, cx)
        sign = "+" if attrs[seg_id] >= 0 else ""
        marker = "  ← highest" if rank == 0 else ""
        top_lines.append(
            f"  Superpixel {seg_id} ({location}): "
            f"{sign}{attrs[seg_id]:.3f}{marker}"
        )

    # Top-3 predicted classes
    top3_idx = np.argsort(probs)[::-1][:3]
    prob_parts = [f"class_{i}={probs[i]:.3f}" for i in top3_idx]
    prob_str = ", ".join(prob_parts)

    # Positive vs negative attribution summary
    pos_mask = attrs > 0
    neg_mask = attrs < 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    sum_pos = float(attrs[pos_mask].sum()) if n_pos > 0 else 0.0
    sum_neg = float(attrs[neg_mask].sum()) if n_neg > 0 else 0.0

    lines = [
        "=== INSTANCE DATA ===",
        f"Dataset: CUB-200-2011  |  Model: ResNet-50  |  Instance index: {result.instance_index}",
        f"Predicted class: {result.predicted_class_name} "
        f"(class index {result.predicted_class}, confidence: {result.confidence*100:.1f}%)",
        f"Top-3 probabilities: {prob_str}",
        "",
        f"=== ATTRIBUTION ANALYSIS ({result.method}, {n_segments} SLIC superpixels) ===",
        f"Top {top_n} superpixels by |attribution|:",
    ]
    lines.extend(top_lines)
    lines += [
        f"Positive superpixels (support prediction): {n_pos} "
        f"(total weight: {sum_pos:.3f})",
        f"Negative superpixels (contradict prediction): {n_neg} "
        f"(total weight: {sum_neg:.3f})",
        "",
        "=== BENCHMARK METRICS (200 test instances) ===",
        metrics_str,
    ]

    return "\n".join(lines)


def _spatial_label(cy: float, cx: float) -> str:
    """Convert normalised (cy, cx) to a human-readable spatial description."""
    vert = "upper" if cy < 0.4 else ("lower" if cy > 0.6 else "central")
    horiz = "left" if cx < 0.4 else ("right" if cx > 0.6 else "central")
    if vert == "central" and horiz == "central":
        return "central body region"
    return f"{vert}-{horiz} region"
