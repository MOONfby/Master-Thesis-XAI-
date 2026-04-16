"""
Explainer Runner — on-demand attribution computation.

Dispatches to the correct explainer based on modality + method,
and returns a unified ExplanationResult regardless of modality.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from phase3.config import (
    MODALITY_TS, MODALITY_IMAGE, MODALITY_TABULAR,
    ADULT_FEATURE_NAMES,
)


# ──────────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────────

@dataclass
class ExplanationResult:
    attributions: np.ndarray          # (N_features,) segment/superpixel/feature level
    predicted_class: int
    predicted_class_name: str
    confidence: float
    all_probabilities: np.ndarray     # (N_classes,)
    method: str
    modality: str
    pixel_heatmap: Optional[np.ndarray]   # (H,W) image | (T,) timeseries | None tabular
    segment_map: Optional[np.ndarray]     # (H,W) image | (T,) timeseries | None tabular
    instance_index: int
    runtime_seconds: float


# ──────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────

def run_explanation(
    modality: str,
    method: str,
    instance_idx: int,
    bundle,
) -> ExplanationResult:
    """
    Compute an explanation for a single instance.

    Parameters
    ----------
    modality : str
        One of MODALITY_TS, MODALITY_IMAGE, MODALITY_TABULAR.
    method : str
        Method name string (e.g. "LIME-TS", "TimeSHAP", "LIME", "SHAP", ...).
    instance_idx : int
        Index into the test set.
    bundle : TimeSeriesBundle | ImageBundle | TabularBundle

    Returns
    -------
    ExplanationResult
    """
    if modality == MODALITY_TS:
        return _explain_timeseries(method, instance_idx, bundle)
    elif modality == MODALITY_IMAGE:
        return _explain_image(method, instance_idx, bundle)
    elif modality == MODALITY_TABULAR:
        return _explain_tabular(method, instance_idx, bundle)
    else:
        raise ValueError(f"Unknown modality: {modality}")


# ──────────────────────────────────────────────────────────────────
# Time Series
# ──────────────────────────────────────────────────────────────────

def _explain_timeseries(method: str, idx: int, bundle) -> ExplanationResult:
    series = bundle.data["X_test"][idx]        # (1, 140)
    seg_map = bundle.segment_map               # (140,)

    # Predict
    probs = bundle.predict_fn(series[np.newaxis])[0]   # (5,)
    pred_cls = int(probs.argmax())
    confidence = float(probs[pred_cls])

    # Explain
    t0 = time.perf_counter()
    if method == "LIME-TS":
        attrs = bundle.lime_explainer.explain(series, seg_map, label=pred_cls)
        heatmap = _broadcast_segments_to_timesteps(attrs, seg_map)
    elif method == "TimeSHAP":
        attrs = bundle.timeshap_explainer.explain(series, seg_map, label=pred_cls)
        heatmap = _broadcast_segments_to_timesteps(attrs, seg_map)
    elif method == "Integrated Gradients":
        attrs = bundle.ig_explainer.explain(series, seg_map, label=pred_cls)
        heatmap = bundle.ig_explainer.explain_timestep_level(series, label=pred_cls)
        if hasattr(heatmap, "numpy"):
            heatmap = heatmap.numpy()
        heatmap = np.abs(heatmap.squeeze())   # (140,)
    else:
        raise ValueError(f"Unknown TS method: {method}")
    runtime = time.perf_counter() - t0

    cls_name = bundle.class_names[pred_cls] if pred_cls < len(bundle.class_names) else str(pred_cls)

    return ExplanationResult(
        attributions=attrs,
        predicted_class=pred_cls,
        predicted_class_name=cls_name,
        confidence=confidence,
        all_probabilities=probs,
        method=method,
        modality=MODALITY_TS,
        pixel_heatmap=heatmap,
        segment_map=seg_map,
        instance_index=idx,
        runtime_seconds=runtime,
    )


def _broadcast_segments_to_timesteps(
    attrs: np.ndarray, seg_map: np.ndarray
) -> np.ndarray:
    """Broadcast (S,) segment attributions to (T,) timestep heatmap."""
    heatmap = np.zeros(len(seg_map), dtype=float)
    for seg_id, attr_val in enumerate(attrs):
        heatmap[seg_map == seg_id] = attr_val
    return heatmap


# ──────────────────────────────────────────────────────────────────
# Image
# ──────────────────────────────────────────────────────────────────

def _explain_image(method: str, idx: int, bundle) -> ExplanationResult:
    from skimage.segmentation import slic

    image = bundle.data["X_test"][idx]    # (3, H, W)
    img_hwc = image.transpose(1, 2, 0)   # (H, W, 3)

    # SLIC per-image (must match Phase 1 config: n_segments=50)
    seg_map = slic(img_hwc, n_segments=50, compactness=10, sigma=1, start_label=0)

    # Predict
    probs = bundle.predict_fn(image[np.newaxis])[0]   # (N_classes,)
    pred_cls = int(probs.argmax())
    confidence = float(probs[pred_cls])

    # Explain
    t0 = time.perf_counter()
    if method == "LIME":
        attrs = bundle.lime_explainer.explain(image, seg_map, label=pred_cls)
        heatmap = _build_pixel_heatmap_from_segments(attrs, seg_map)
    elif method == "GradientSHAP":
        attrs = bundle.gradshap_explainer.explain(image, seg_map, label=pred_cls)
        heatmap = bundle.gradshap_explainer.explain_pixel_level(image, label=pred_cls)
        if hasattr(heatmap, "numpy"):
            heatmap = heatmap.numpy()
    else:
        raise ValueError(f"Unknown image method: {method}")
    runtime = time.perf_counter() - t0

    cls_name = bundle.class_names[pred_cls] if pred_cls < len(bundle.class_names) else str(pred_cls)

    return ExplanationResult(
        attributions=attrs,
        predicted_class=pred_cls,
        predicted_class_name=cls_name,
        confidence=confidence,
        all_probabilities=probs,
        method=method,
        modality=MODALITY_IMAGE,
        pixel_heatmap=heatmap,
        segment_map=seg_map,
        instance_index=idx,
        runtime_seconds=runtime,
    )


def _build_pixel_heatmap_from_segments(
    attrs: np.ndarray, seg_map: np.ndarray
) -> np.ndarray:
    """Build (H, W) pixel heatmap by assigning each pixel its segment's attribution."""
    heatmap = np.zeros(seg_map.shape, dtype=float)
    for seg_id, attr_val in enumerate(attrs):
        heatmap[seg_map == seg_id] = attr_val
    return heatmap


# ──────────────────────────────────────────────────────────────────
# Tabular
# ──────────────────────────────────────────────────────────────────

def _explain_tabular(method: str, idx: int, bundle) -> ExplanationResult:
    import pandas as pd

    X_test = bundle.data.get("X_test", bundle.data.get("X_test_df"))
    if hasattr(X_test, "iloc"):
        instance_df = X_test.iloc[idx]
        instance_np = instance_df.values.astype(float)
    else:
        instance_np = np.array(X_test[idx], dtype=float)
        instance_df = pd.Series(instance_np, index=ADULT_FEATURE_NAMES)

    # Predict
    probs = bundle.model.predict_proba(
        pd.DataFrame([instance_np], columns=ADULT_FEATURE_NAMES)
    )[0]
    pred_cls = int(probs.argmax())
    confidence = float(probs[pred_cls])

    # Explain
    t0 = time.perf_counter()
    if method == "LIME":
        exp = bundle.lime_explainer.explain_instance(
            instance_np,
            bundle.model.predict_proba,
            num_features=len(ADULT_FEATURE_NAMES),
            labels=[pred_cls],
        )
        # Map back to feature order
        attrs = np.zeros(len(ADULT_FEATURE_NAMES), dtype=float)
        for feat_desc, weight in exp.as_list(label=pred_cls):
            # LIME feature descriptions can be like "age > 40" — match by feature name
            for fi, fname in enumerate(ADULT_FEATURE_NAMES):
                if fname in feat_desc:
                    attrs[fi] = weight
                    break

    elif method == "SHAP":
        shap_values = bundle.shap_explainer.shap_values(
            pd.DataFrame([instance_np], columns=ADULT_FEATURE_NAMES)
        )
        # shap_values may be list (one per class) or ndarray (N, F) or (N, F, C)
        if isinstance(shap_values, list):
            attrs = np.array(shap_values[pred_cls][0], dtype=float)
        elif shap_values.ndim == 3:
            attrs = shap_values[0, :, pred_cls].astype(float)
        else:
            attrs = shap_values[0].astype(float)
    else:
        raise ValueError(f"Unknown tabular method: {method}")

    runtime = time.perf_counter() - t0

    cls_name = bundle.class_names[pred_cls] if pred_cls < len(bundle.class_names) else str(pred_cls)

    return ExplanationResult(
        attributions=attrs,
        predicted_class=pred_cls,
        predicted_class_name=cls_name,
        confidence=confidence,
        all_probabilities=probs,
        method=method,
        modality=MODALITY_TABULAR,
        pixel_heatmap=None,
        segment_map=None,
        instance_index=idx,
        runtime_seconds=runtime,
    )
