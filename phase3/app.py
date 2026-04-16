"""
Phase 3 — XAI Explanation Dashboard
Streamlit entry point.

Run from the code/ directory:
    streamlit run phase3/app.py
"""
import sys
from pathlib import Path

# Ensure code/ is on PYTHONPATH when launched via `streamlit run phase3/app.py`
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

from phase3.config import (
    MODALITY_TS, MODALITY_IMAGE, MODALITY_TABULAR,
)
from phase3.ui.sidebar import render_sidebar


# ──────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XAI Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _detect_context_change(modality: str, persona: str, method: str, idx: int) -> bool:
    """Return True if any key explanation parameter has changed since last run."""
    last_key = st.session_state.get("last_context_key")
    new_key  = (modality, persona, method, idx)
    return last_key != new_key


def _load_bundle(modality: str):
    """Load (cached) bundle for the selected modality."""
    from phase3.backend.model_registry import (
        load_timeseries_bundle,
        load_image_bundle,
        load_tabular_bundle,
    )
    if modality == MODALITY_TS:
        return load_timeseries_bundle()
    elif modality == MODALITY_IMAGE:
        return load_image_bundle()
    elif modality == MODALITY_TABULAR:
        return load_tabular_bundle()
    return None


def _build_context(result, modality: str, method: str, bundle) -> str:
    """Build the grounded context block for the LLM prompt."""
    from phase3.backend.metrics_loader import (
        load_metrics_csv, get_method_metrics, format_metrics_for_prompt
    )
    from phase3.config import (
        TS_METRICS_CSV, IMAGE_METRICS_CSV, TABULAR_METRICS_CSV
    )

    csv_map = {
        MODALITY_TS:      TS_METRICS_CSV,
        MODALITY_IMAGE:   IMAGE_METRICS_CSV,
        MODALITY_TABULAR: TABULAR_METRICS_CSV,
    }
    prefix_map = {
        "LIME-TS": "LIME",
        "TimeSHAP": "TimeSHAP",
        "Integrated Gradients": "IG",
        "LIME": "LIME",
        "GradientSHAP": "GradSHAP",
        "SHAP": "SHAP",
    }

    df = load_metrics_csv(csv_map[modality])
    prefix = prefix_map.get(method, method)
    metrics = get_method_metrics(df, prefix)
    metrics_str = format_metrics_for_prompt(metrics)

    if modality == MODALITY_TS:
        from phase3.context_builder.timeseries_context import build_timeseries_context
        return build_timeseries_context(result, metrics_str)
    elif modality == MODALITY_IMAGE:
        from phase3.context_builder.image_context import build_image_context
        return build_image_context(result, metrics_str)
    elif modality == MODALITY_TABULAR:
        from phase3.context_builder.tabular_context import build_tabular_context
        return build_tabular_context(result, metrics_str, bundle, result.instance_index)
    return metrics_str


def main():
    # ── Sidebar controls ───────────────────────────────────────────
    controls = render_sidebar()
    modality    = controls["modality"]
    method      = controls["method"]
    persona     = controls["persona"]
    instance_idx = controls["instance_idx"]
    api_key     = controls["api_key"]
    run_clicked = controls["run_clicked"]

    # ── Persist controls ───────────────────────────────────────────
    st.session_state["modality"]     = modality
    st.session_state["method"]       = method
    st.session_state["persona"]      = persona
    st.session_state["instance_idx"] = instance_idx

    # ── Context change detection ──────────────────────────────────
    if _detect_context_change(modality, persona, method, instance_idx):
        st.session_state["conversation_history"] = []
        st.session_state["current_explanation"]  = None
        st.session_state["last_context_key"] = (modality, persona, method, instance_idx)

    # ── Main layout ───────────────────────────────────────────────
    col_viz, col_chat = st.columns([55, 45])

    with col_viz:
        st.subheader("Explanation Visualization")

        # Load bundle
        bundle = _load_bundle(modality)
        if bundle is None:
            st.error(
                f"Could not load the {modality} bundle. "
                "Make sure models are trained and data files exist. "
                "Run the corresponding run_phaseX.py script first."
            )
            return

        # Run explanation on button click
        if run_clicked:
            with st.spinner(f"Running {method} explanation..."):
                from phase3.backend.explainer_runner import run_explanation
                try:
                    result = run_explanation(modality, method, instance_idx, bundle)
                    st.session_state["current_explanation"] = result
                except Exception as e:
                    st.error(f"Explanation failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    return

        # Render visualization if we have a result
        result = st.session_state.get("current_explanation")
        if result is not None:
            _render_visualization(result, bundle)
        else:
            st.info(
                "Select a dataset, method, and instance, then click "
                "**Run Explanation** to start."
            )

    with col_chat:
        from phase3.ui.chat_panel import render_chat_panel, stream_opening_message
        from phase3.context_builder.persona_templates import (
            get_system_prompt, get_opening_question
        )

        render_chat_panel(persona, api_key)

        # Auto-stream explanation when Run is clicked
        result = st.session_state.get("current_explanation")
        if run_clicked and result is not None and api_key:
            system_prompt  = get_system_prompt(persona, modality)
            opening_q      = get_opening_question(persona, modality)
            context_block  = _build_context(result, modality, method, bundle)

            st.session_state["system_prompt"]        = system_prompt
            st.session_state["conversation_history"] = []

            with col_chat:
                stream_opening_message(context_block, opening_q, system_prompt, api_key)


def _render_visualization(result, bundle) -> None:
    """Render the appropriate visualization for the given result."""
    from phase3.ui.main_panel import render_main_panel

    modality = result.modality

    if modality == MODALITY_IMAGE:
        # Image needs special handling (pass raw image to plot function)
        from phase3.visualization.image_plots import plot_image_attribution
        from phase3.backend.explainer_runner import ExplanationResult

        st.metric("Predicted Class", result.predicted_class_name)
        st.metric("Confidence", f"{result.confidence*100:.1f}%")
        st.markdown("---")

        tab_attr, tab_metrics, tab_instance = st.tabs(
            ["Attribution", "Metrics", "Instance Info"]
        )
        with tab_attr:
            if result.pixel_heatmap is not None:
                image = bundle.data["X_test"][result.instance_index]
                fig = plot_image_attribution(
                    image=image,
                    pixel_heatmap=result.pixel_heatmap,
                    segment_map=result.segment_map,
                    predicted_class_name=result.predicted_class_name,
                    confidence=result.confidence,
                    method=result.method,
                )
                st.pyplot(fig, use_container_width=True)
            else:
                st.info("Pixel heatmap not available.")
        with tab_metrics:
            from phase3.ui.main_panel import _render_metrics
            _render_metrics(result)
        with tab_instance:
            from phase3.ui.main_panel import _render_instance_info
            _render_instance_info(result, bundle)
    else:
        render_main_panel(result, bundle)


if __name__ == "__main__":
    main()
