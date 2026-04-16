"""
Context builder for Tabular (Adult Income) explanations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from phase3.backend.explainer_runner import ExplanationResult
from phase3.config import ADULT_FEATURE_NAMES, ADULT_CLASS_NAMES


# Domain-friendly feature name mapping for Domain Expert persona
_FEATURE_LABELS = {
    "age": "age",
    "education-num": "years of education",
    "capital-gain": "capital gains",
    "capital-loss": "capital losses",
    "hours-per-week": "weekly work hours",
    "workclass": "employment type",
    "education": "education level",
    "marital-status": "marital status",
    "occupation": "occupation",
    "relationship": "family relationship",
    "race": "race",
    "sex": "sex",
    "native-country": "country of origin",
}


def build_tabular_context(
    result: ExplanationResult,
    metrics_str: str,
    bundle,
    instance_idx: int,
) -> str:
    """
    Build a grounded context block for a tabular explanation.

    Parameters
    ----------
    result : ExplanationResult
    metrics_str : str
    bundle : TabularBundle
    instance_idx : int

    Returns
    -------
    str
    """
    attrs = result.attributions        # (14,)
    probs = result.all_probabilities   # (2,)
    feature_names = ADULT_FEATURE_NAMES

    # Retrieve actual feature values for this instance
    X_test = bundle.data.get("X_test", bundle.data.get("X_test_df"))
    if hasattr(X_test, "iloc"):
        instance_values = X_test.iloc[instance_idx].to_dict()
    else:
        instance_values = {
            feature_names[i]: float(X_test[instance_idx][i])
            for i in range(len(feature_names))
        }

    # Rank features by |attribution|
    ranked = np.argsort(np.abs(attrs))[::-1]
    top_n = min(8, len(attrs))

    top_lines = []
    for rank, fi in enumerate(ranked[:top_n]):
        fname = feature_names[fi]
        fval  = instance_values.get(fname, "?")
        label = _FEATURE_LABELS.get(fname, fname)
        sign  = "+" if attrs[fi] >= 0 else ""
        direction = "pushes toward >50K" if attrs[fi] > 0 else "pushes toward <=50K"
        marker = "  ← most influential" if rank == 0 else ""
        top_lines.append(
            f"  {label} = {fval}: attribution {sign}{attrs[fi]:.4f} "
            f"({direction}){marker}"
        )

    # Probability row
    prob_parts = [
        f"{ADULT_CLASS_NAMES[i]}={probs[i]:.3f}"
        for i in range(len(probs))
    ]
    prob_str = ", ".join(prob_parts)

    lines = [
        "=== INSTANCE DATA ===",
        f"Dataset: Adult Income (UCI)  |  Model: XGBoost  |  Instance index: {instance_idx}",
        f"Predicted class: {result.predicted_class_name} "
        f"(confidence: {result.confidence*100:.1f}%)",
        f"Class probabilities: {prob_str}",
        "",
        f"=== ATTRIBUTION ANALYSIS ({result.method}, {len(attrs)} features) ===",
        f"Top {top_n} features by |attribution|:",
    ]
    lines.extend(top_lines)
    lines += [
        "",
        "=== BENCHMARK METRICS (300 test instances) ===",
        metrics_str,
    ]

    return "\n".join(lines)
