"""
Context builder for Tabular (Adult Income) explanations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from phase3.backend.explainer_runner import ExplanationResult
from phase3.config import ADULT_FEATURE_NAMES, ADULT_CLASS_NAMES, ADULT_CATEGORICAL_FEATURES


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


# ──────────────────────────────────────────────────────────────────
# What-If context builder
# ──────────────────────────────────────────────────────────────────

def _resolve_display_value(feat: str, val: float, bundle) -> str:
    """Resolve an encoded feature value to a human-readable string."""
    if feat in ADULT_CATEGORICAL_FEATURES:
        cat_labels = getattr(bundle, "cat_labels", {})
        labels = cat_labels.get(feat, [])
        if labels:
            idx = int(round(val))
            if 0 <= idx < len(labels):
                return labels[idx]
        return str(int(round(val)))
    # Numerical
    v = float(val)
    return str(int(v)) if v == int(v) else f"{v:.2f}"


def build_whatif_context(
    original_result: ExplanationResult,
    modified_values: dict,
    new_probs: "np.ndarray",
    bundle,
) -> str:
    """
    Build a grounded context block describing a what-if modification for the LLM.

    Parameters
    ----------
    original_result : ExplanationResult  — the original explanation
    modified_values : dict[str, float]   — current slider/dropdown values
    new_probs : np.ndarray (2,)          — model probabilities for modified instance
    bundle : TabularBundle
    """
    feature_names = ADULT_FEATURE_NAMES

    # Original feature values
    X_test = bundle.data.get("X_test", bundle.data.get("X_test_df"))
    if hasattr(X_test, "iloc"):
        orig_values = X_test.iloc[original_result.instance_index].to_dict()
    else:
        orig_values = {
            feature_names[i]: float(X_test[original_result.instance_index][i])
            for i in range(len(feature_names))
        }

    # Changed features
    changed_lines = []
    for feat in feature_names:
        orig_val = float(orig_values.get(feat, 0.0))
        mod_val  = float(modified_values.get(feat, orig_val))
        if abs(orig_val - mod_val) > 1e-6:
            label = _FEATURE_LABELS.get(feat, feat)
            orig_display = _resolve_display_value(feat, orig_val, bundle)
            mod_display  = _resolve_display_value(feat, mod_val, bundle)
            changed_lines.append(f"  {label}: {orig_display} → {mod_display}")

    # New prediction
    new_pred_cls  = int(new_probs.argmax())
    new_pred_name = ADULT_CLASS_NAMES[new_pred_cls]
    new_conf      = float(new_probs[new_pred_cls])
    pred_changed  = new_pred_cls != original_result.predicted_class

    orig_prob_str = ", ".join(
        f"{ADULT_CLASS_NAMES[i]}={original_result.all_probabilities[i]:.3f}"
        for i in range(len(original_result.all_probabilities))
    )
    new_prob_str = ", ".join(
        f"{ADULT_CLASS_NAMES[i]}={new_probs[i]:.3f}"
        for i in range(len(new_probs))
    )

    lines = [
        "=== WHAT-IF ANALYSIS ===",
        f"Dataset: Adult Income (UCI)  |  Model: XGBoost  "
        f"|  Instance index: {original_result.instance_index}",
        "",
        "=== ORIGINAL PREDICTION ===",
        f"Predicted class: {original_result.predicted_class_name} "
        f"(confidence: {original_result.confidence*100:.1f}%)",
        f"Class probabilities: {orig_prob_str}",
        "",
        f"=== FEATURE MODIFICATIONS ({len(changed_lines)} change(s)) ===",
    ]
    lines.extend(changed_lines if changed_lines else ["  (no changes)"])
    lines += [
        "",
        "=== MODIFIED PREDICTION ===",
        f"Predicted class: {new_pred_name} (confidence: {new_conf*100:.1f}%)",
        f"Class probabilities: {new_prob_str}",
        f"Prediction changed: {'YES' if pred_changed else 'NO'}",
    ]

    return "\n".join(lines)
