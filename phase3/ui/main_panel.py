"""
Main visualization panel — prediction badge, attribution tabs, metrics.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from phase3.backend.explainer_runner import ExplanationResult
from phase3.config import (
    MODALITY_TS, MODALITY_IMAGE, MODALITY_TABULAR,
    ADULT_FEATURE_NAMES, ECG_CLASS_NAMES,
)


def render_main_panel(result: ExplanationResult, bundle) -> None:
    """
    Render the visualization column: prediction badge + tabs.

    Parameters
    ----------
    result : ExplanationResult
    bundle : TimeSeriesBundle | ImageBundle | TabularBundle
    """
    # ── Prediction badge ─────────────────────────────────────────
    col_cls, col_conf = st.columns([2, 1])
    with col_cls:
        st.metric(
            label="Predicted Class",
            value=result.predicted_class_name,
        )
    with col_conf:
        st.metric(
            label="Confidence",
            value=f"{result.confidence*100:.1f}%",
        )

    st.markdown("---")

    # ── Tabs ─────────────────────────────────────────────────────
    tab_attr, tab_metrics, tab_instance = st.tabs(
        ["Attribution", "Metrics", "Instance Info"]
    )

    with tab_attr:
        _render_attribution(result, bundle)

    with tab_metrics:
        _render_metrics(result)

    with tab_instance:
        _render_instance_info(result, bundle)


def _render_attribution(result: ExplanationResult, bundle) -> None:
    if result.modality == MODALITY_TABULAR:
        _render_tabular_attribution(result, bundle)
    elif result.modality == MODALITY_IMAGE:
        _render_image_attribution(result)
    elif result.modality == MODALITY_TS:
        _render_ts_attribution(result, bundle)


def _render_tabular_attribution(result: ExplanationResult, bundle) -> None:
    from phase3.visualization.tabular_plots import plot_feature_attribution
    fig = plot_feature_attribution(
        result.attributions,
        feature_names=bundle.feature_names,
        title=f"{result.method} — Feature Attribution",
    )
    st.pyplot(fig, use_container_width=True)


def _render_image_attribution(result: ExplanationResult) -> None:
    if result.pixel_heatmap is None:
        st.info("Pixel heatmap not available for this method.")
        return
    X_test = None
    # We need the raw image — passed via result.instance_index in a full run;
    # the plot function is called from app.py which has the bundle.
    st.info("Image visualization is rendered in app.py with bundle context.")


def _render_ts_attribution(result: ExplanationResult, bundle) -> None:
    from phase3.visualization.timeseries_plots import plot_ecg_attribution
    series = bundle.data["X_test"][result.instance_index]
    fig = plot_ecg_attribution(
        series=series,
        attributions=result.attributions,
        segment_map=result.segment_map,
        predicted_class_name=result.predicted_class_name,
        confidence=result.confidence,
        method=result.method,
        instance_idx=result.instance_index,
    )
    st.pyplot(fig, use_container_width=True)


def _render_metrics(result: ExplanationResult) -> None:
    """Show raw benchmark metrics for the selected method."""
    from phase3.backend.metrics_loader import load_metrics_csv, get_method_metrics
    from phase3.config import TS_METRICS_CSV, IMAGE_METRICS_CSV, TABULAR_METRICS_CSV

    csv_map = {
        MODALITY_TS:      TS_METRICS_CSV,
        MODALITY_IMAGE:   IMAGE_METRICS_CSV,
        MODALITY_TABULAR: TABULAR_METRICS_CSV,
    }
    csv_path = csv_map.get(result.modality)
    df = load_metrics_csv(csv_path)

    # Determine CSV column prefix for this method
    prefix_map = {
        "LIME-TS": "LIME",
        "TimeSHAP": "TimeSHAP",
        "Integrated Gradients": "IG",
        "LIME": "LIME",
        "GradientSHAP": "GradSHAP",
        "SHAP": "SHAP",
    }
    prefix = prefix_map.get(result.method, result.method)
    metrics = get_method_metrics(df, prefix)

    if not metrics:
        st.info("No benchmark metrics found for this method.")
        return

    st.caption(f"Benchmark metrics for **{result.method}** (200 test instances)")

    # Group into categories
    faith_keys = ["AOPC", "Comprehensiveness", "Sufficiency", "InsertionAUC", "DeletionAUC",
                  "LocalFidelity_R2"]
    stab_keys  = ["RankCorrelation", "AvgSensitivity"]
    loc_keys   = ["PointingGame", "SegIoU", "TemporalIoU"]
    time_keys  = ["MeanTime_s", "StdTime_s"]

    def _show_group(title: str, keys: list[str]) -> None:
        group = {k: v for k, v in metrics.items() if k in keys}
        if group:
            st.markdown(f"**{title}**")
            cols = st.columns(min(3, len(group)))
            for i, (k, v) in enumerate(group.items()):
                cols[i % len(cols)].metric(k, f"{v:.4f}" if isinstance(v, float) else str(v))

    _show_group("Faithfulness", faith_keys)
    _show_group("Stability", stab_keys)
    _show_group("Localization", loc_keys)
    _show_group("Runtime", time_keys)

    # Show remaining keys
    shown = set(faith_keys + stab_keys + loc_keys + time_keys)
    other = {k: v for k, v in metrics.items() if k not in shown}
    if other:
        st.markdown("**Other**")
        st.json(other)


def _render_instance_info(result: ExplanationResult, bundle) -> None:
    st.caption(f"Instance index: **{result.instance_index}**")
    st.caption(f"Method: **{result.method}**  |  Runtime: **{result.runtime_seconds*1000:.1f} ms**")

    # Probability table
    probs = result.all_probabilities
    if result.modality == MODALITY_TS:
        names = ECG_CLASS_NAMES
    elif result.modality == MODALITY_TABULAR:
        from phase3.config import ADULT_CLASS_NAMES
        names = ADULT_CLASS_NAMES
    else:
        names = [f"class_{i}" for i in range(len(probs))]

    st.markdown("**Class probabilities:**")
    prob_data = {
        "Class": names[:len(probs)],
        "Probability": [f"{p:.4f}" for p in probs],
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(prob_data), hide_index=True, use_container_width=True)
