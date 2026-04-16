"""
Sidebar controls for the dashboard.
"""
from __future__ import annotations

import os

import streamlit as st

from phase3.config import (
    MODALITIES, MODALITY_TS, MODALITY_IMAGE, MODALITY_TABULAR,
    TS_METHODS, IMAGE_METHODS, TABULAR_METHODS,
    PERSONAS, LLM_MODEL,
)


def render_sidebar() -> dict:
    """
    Render all sidebar controls.

    Returns a dict with keys:
        api_key, modality, method, persona, instance_idx, run_clicked
    """
    st.sidebar.title("XAI Dashboard")
    st.sidebar.markdown("---")

    # ── API Key ─────────────────────────────────────────────────
    st.sidebar.subheader("Anthropic API Key")
    default_key = os.environ.get("ANTHROPIC_API_KEY", "")
    api_key = st.sidebar.text_input(
        "API Key",
        value=st.session_state.get("api_key", default_key),
        type="password",
        label_visibility="collapsed",
        placeholder="sk-...",
        key="api_key_input",
    )
    if api_key:
        st.session_state["api_key"] = api_key

    st.sidebar.markdown("---")

    # ── Dataset & Model ─────────────────────────────────────────
    st.sidebar.subheader("Dataset & Model")
    modality = st.sidebar.selectbox(
        "Modality",
        MODALITIES,
        index=MODALITIES.index(st.session_state.get("modality", MODALITY_TS)),
        key="modality_select",
        label_visibility="collapsed",
    )

    # ── Method ──────────────────────────────────────────────────
    st.sidebar.subheader("Explanation Method")
    method_options = {
        MODALITY_TS:      TS_METHODS,
        MODALITY_IMAGE:   IMAGE_METHODS,
        MODALITY_TABULAR: TABULAR_METHODS,
    }[modality]

    prev_method = st.session_state.get("method", method_options[0])
    default_method_idx = method_options.index(prev_method) if prev_method in method_options else 0

    method = st.sidebar.radio(
        "Method",
        method_options,
        index=default_method_idx,
        key="method_radio",
        label_visibility="collapsed",
    )

    # ── Persona ──────────────────────────────────────────────────
    st.sidebar.subheader("User Persona")
    persona = st.sidebar.radio(
        "Persona",
        PERSONAS,
        index=PERSONAS.index(st.session_state.get("persona", PERSONAS[0])),
        key="persona_radio",
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")

    # ── Instance Selection ───────────────────────────────────────
    st.sidebar.subheader("Instance Selection")
    max_instances = {
        MODALITY_TS:      4499,
        MODALITY_IMAGE:   199,
        MODALITY_TABULAR: 299,
    }[modality]

    instance_idx = st.sidebar.number_input(
        "Instance index",
        min_value=0,
        max_value=max_instances,
        value=int(st.session_state.get("instance_idx", 0)),
        step=1,
        label_visibility="collapsed",
        key="instance_input",
    )

    run_clicked = st.sidebar.button(
        "Run Explanation",
        type="primary",
        use_container_width=True,
    )

    st.sidebar.markdown("---")

    # ── About ────────────────────────────────────────────────────
    with st.sidebar.expander("About"):
        st.markdown(
            "**XAI Explanation Dashboard**  \n"
            "KTH Master's Thesis — Biying Feng, 2026  \n\n"
            "Translates XAI attribution outputs into natural language "
            "explanations tailored to three audience personas using "
            f"DeepSeek ({LLM_MODEL})."
        )

    return {
        "api_key": api_key,
        "modality": modality,
        "method": method,
        "persona": persona,
        "instance_idx": int(instance_idx),
        "run_clicked": run_clicked,
    }
