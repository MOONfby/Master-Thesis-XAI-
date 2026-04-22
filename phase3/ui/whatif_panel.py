"""
What-If interactive panel for Tabular (Adult Income) explanations.

Users adjust feature values via sliders/dropdowns and see how the prediction
changes in real time. A "Explain this change" button sends the before/after
context to the LLM chat panel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from phase3.backend.explainer_runner import ExplanationResult
from phase3.config import (
    ADULT_FEATURE_NAMES,
    ADULT_NUMERICAL_FEATURES,
    ADULT_CATEGORICAL_FEATURES,
    ADULT_CLASS_NAMES,
)

_FEATURE_LABELS = {
    "age": "Age",
    "education-num": "Years of education",
    "capital-gain": "Capital gains",
    "capital-loss": "Capital losses",
    "hours-per-week": "Weekly work hours",
    "workclass": "Employment type",
    "education": "Education level",
    "marital-status": "Marital status",
    "occupation": "Occupation",
    "relationship": "Family relationship",
    "race": "Race",
    "sex": "Sex",
    "native-country": "Country of origin",
}

_SLIDER_STEPS = {
    "age": 1.0,
    "education-num": 1.0,
    "capital-gain": 500.0,
    "capital-loss": 100.0,
    "hours-per-week": 1.0,
}


def render_whatif_panel(result: ExplanationResult, bundle) -> None:
    """
    Render the interactive What-If panel.

    Parameters
    ----------
    result : ExplanationResult  — current explanation (used for original values)
    bundle : TabularBundle
    """
    _init_whatif_state(result, bundle)

    # Reset counter — changing this forces all widgets to re-initialize from value=
    rc = st.session_state.get("whatif_reset_count", 0)

    st.caption(
        "Adjust feature values below and observe how the model prediction changes. "
        "Click **Explain this change** to get an LLM interpretation."
    )
    st.markdown("---")

    col_controls, col_pred = st.columns([6, 4])

    # ── Left: feature controls ─────────────────────────────────────
    with col_controls:
        st.markdown("**Feature values**")

        # Numerical sliders
        st.markdown("*Numerical features*")
        for feat in ADULT_NUMERICAL_FEATURES:
            fmin, fmax = bundle.feature_ranges.get(feat, (0.0, 100.0))
            current_val = float(st.session_state["whatif_values"].get(feat, fmin))
            current_val = float(np.clip(current_val, fmin, fmax))
            step = _SLIDER_STEPS.get(feat, 1.0)
            new_val = st.slider(
                label=_FEATURE_LABELS.get(feat, feat),
                min_value=fmin,
                max_value=fmax,
                value=current_val,
                step=step,
                key=f"whatif_slider_{feat}_{rc}",
            )
            st.session_state["whatif_values"][feat] = new_val

        # Categorical dropdowns
        st.markdown("*Categorical features*")
        for feat in ADULT_CATEGORICAL_FEATURES:
            current_int = int(round(float(
                st.session_state["whatif_values"].get(feat, 0)
            )))
            labels = bundle.cat_labels.get(feat, [])

            if labels:
                current_int = min(current_int, len(labels) - 1)
                selected = st.selectbox(
                    label=_FEATURE_LABELS.get(feat, feat),
                    options=labels,
                    index=current_int,
                    key=f"whatif_select_{feat}_{rc}",
                )
                st.session_state["whatif_values"][feat] = float(labels.index(selected))
            else:
                n_cats = _estimate_n_categories(feat, bundle)
                selected_int = st.selectbox(
                    label=_FEATURE_LABELS.get(feat, feat),
                    options=list(range(n_cats)),
                    index=min(current_int, n_cats - 1),
                    key=f"whatif_select_{feat}_{rc}",
                )
                st.session_state["whatif_values"][feat] = float(selected_int)

        st.button(
            "Reset to original values",
            key="whatif_reset_btn",
            on_click=_reset_whatif_state,
            args=(result, bundle),
        )

    # ── Right: live prediction ────────────────────────────────────
    with col_pred:
        st.markdown("**Prediction comparison**")

        mod_cls, mod_cls_name, mod_conf, mod_probs = _compute_whatif_prediction(
            st.session_state["whatif_values"], bundle
        )

        orig_cls_name = result.predicted_class_name
        orig_conf = result.confidence

        st.markdown("*Original*")
        c1, c2 = st.columns(2)
        c1.metric("Class", orig_cls_name)
        c2.metric("Confidence", f"{orig_conf*100:.1f}%")

        st.markdown("*Modified*")
        c3, c4 = st.columns(2)
        cls_delta = "CHANGED ⚠️" if mod_cls_name != orig_cls_name else None
        conf_delta = f"{(mod_conf - orig_conf)*100:+.1f}%"
        c3.metric("Class", mod_cls_name, delta=cls_delta,
                  delta_color="off" if cls_delta else "normal")
        c4.metric("Confidence", f"{mod_conf*100:.1f}%", delta=conf_delta)

        # Changed features summary
        st.markdown("---")
        changes = _compute_changes(
            _get_original_values(result, bundle),
            st.session_state["whatif_values"],
            bundle,
        )
        if changes:
            st.markdown(f"**{len(changes)} feature(s) modified:**")
            st.dataframe(
                pd.DataFrame(changes)[["Feature", "Original", "Modified"]],
                hide_index=True,
                use_container_width=True,
            )

            if st.button("Explain this change", key="whatif_explain_btn", type="primary"):
                from phase3.context_builder.tabular_context import build_whatif_context
                ctx = build_whatif_context(result, st.session_state["whatif_values"],
                                           mod_probs, bundle)
                st.session_state["whatif_context_block"] = ctx
                st.session_state["whatif_explain_requested"] = True
                st.rerun()
        else:
            st.info("Adjust feature values above to explore what-if scenarios.")


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _get_original_values(result: ExplanationResult, bundle) -> dict:
    """Extract feature values for the current instance from bundle."""
    X_test = bundle.data.get("X_test", bundle.data.get("X_test_df"))
    idx = result.instance_index
    if hasattr(X_test, "iloc"):
        return {f: float(v) for f, v in X_test.iloc[idx].items()}
    return {ADULT_FEATURE_NAMES[i]: float(X_test[idx][i])
            for i in range(len(ADULT_FEATURE_NAMES))}


def _init_whatif_state(result: ExplanationResult, bundle) -> None:
    """Populate whatif_values from the current instance if instance changed."""
    if st.session_state.get("whatif_instance_idx") != result.instance_index:
        original = _get_original_values(result, bundle)
        st.session_state["whatif_values"] = original
        st.session_state["whatif_instance_idx"] = result.instance_index
        st.session_state["whatif_reset_count"] = st.session_state.get("whatif_reset_count", 0) + 1


def _reset_whatif_state(result: ExplanationResult, bundle) -> None:
    """Reset all what-if values. Safe to call as on_click callback."""
    original = _get_original_values(result, bundle)
    st.session_state["whatif_values"] = original
    st.session_state["whatif_instance_idx"] = result.instance_index
    # Increment reset counter — widget keys change, forcing re-initialization from value=
    st.session_state["whatif_reset_count"] = st.session_state.get("whatif_reset_count", 0) + 1


def _compute_whatif_prediction(values: dict, bundle) -> tuple:
    """Returns (pred_class, pred_class_name, confidence, all_probs)."""
    row = [values.get(f, 0.0) for f in ADULT_FEATURE_NAMES]
    df = pd.DataFrame([row], columns=ADULT_FEATURE_NAMES)
    probs = bundle.model.predict_proba(df)[0]
    pred_cls = int(probs.argmax())
    return pred_cls, bundle.class_names[pred_cls], float(probs[pred_cls]), probs


def _estimate_n_categories(feat: str, bundle) -> int:
    """Fallback: count unique values for a feature in training data."""
    X_train = bundle.data.get("X_train_df") or bundle.data.get("X_train")
    if X_train is None:
        return 8
    if hasattr(X_train, "columns") and feat in X_train.columns:
        return int(X_train[feat].nunique())
    return 8


def _compute_changes(original: dict, modified: dict, bundle) -> list[dict]:
    """Return list of changed features with display-friendly values."""
    from phase3.context_builder.tabular_context import _resolve_display_value
    changes = []
    for feat in ADULT_FEATURE_NAMES:
        orig_val = float(original.get(feat, 0.0))
        mod_val  = float(modified.get(feat, orig_val))
        if abs(orig_val - mod_val) > 1e-6:
            changes.append({
                "Feature": _FEATURE_LABELS.get(feat, feat),
                "Original": _resolve_display_value(feat, orig_val, bundle),
                "Modified": _resolve_display_value(feat, mod_val, bundle),
            })
    return changes
