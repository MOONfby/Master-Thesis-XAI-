"""
Model Registry — loads all three modality bundles once per server process.

Uses st.cache_resource so each bundle is shared across all Streamlit sessions.
Models are loaded from disk (pre-trained); training is triggered only if
the model file is missing (can take several minutes).
"""
from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import streamlit as st

from phase3.config import (
    TS_MODEL_PATH, IMAGE_MODEL_PATH, TABULAR_MODEL_PATH,
    TS_DATA_PATH, IMAGE_DATA_PATH, TABULAR_DATA_PATH,
    ECG_CLASS_NAMES,
    ADULT_FEATURE_NAMES, ADULT_NUMERICAL_FEATURES, ADULT_CLASS_NAMES,
)


# ──────────────────────────────────────────────────────────────────
# Bundle dataclasses
# ──────────────────────────────────────────────────────────────────

@dataclass
class TimeSeriesBundle:
    data: dict
    model: object
    predict_fn: Callable
    lime_explainer: object
    timeshap_explainer: object
    ig_explainer: object
    segment_map: np.ndarray       # shape (140,) — uniform segments
    class_names: list[str]


@dataclass
class ImageBundle:
    data: dict
    model: object
    predict_fn: Callable
    lime_explainer: object
    gradshap_explainer: object
    class_names: list[str]


@dataclass
class TabularBundle:
    data: dict
    model: object
    lime_explainer: object
    shap_explainer: object
    feature_names: list[str]
    numerical_indices: list[int]
    class_names: list[str]
    cat_labels: dict = field(default_factory=dict)     # {"workclass": ["Private", ...], ...}
    feature_ranges: dict = field(default_factory=dict) # {"age": (17.0, 90.0), ...}


# ──────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Time Series model...")
def load_timeseries_bundle() -> Optional[TimeSeriesBundle]:
    try:
        from phase2.timeseries.data_loader import load_ecg5000
        from phase2.timeseries.model_trainer import (
            load_or_train_inceptiontime, make_predict_fn
        )
        from phase2.timeseries.explainers.lime_ts_explainer import LIMETSExplainer
        from phase2.timeseries.explainers.timeshap_explainer import TimeSHAPExplainer
        from phase2.timeseries.explainers.ig_explainer import IntegratedGradientsExplainer
        from phase2.timeseries.utils import build_segment_map

        data    = load_ecg5000()
        model   = load_or_train_inceptiontime(data)
        p_fn    = make_predict_fn(model)
        seg_map = build_segment_map()   # (140,)

        lime_exp = LIMETSExplainer(
            predict_fn=p_fn,
            baseline_series=data["baseline_series"],
        )
        ts_exp = TimeSHAPExplainer(
            predict_fn=p_fn,
            baseline_series=data["baseline_series"],
        )
        ig_exp = IntegratedGradientsExplainer(
            model=model,
            baseline_series=data["baseline_series"],
        )

        return TimeSeriesBundle(
            data=data,
            model=model,
            predict_fn=p_fn,
            lime_explainer=lime_exp,
            timeshap_explainer=ts_exp,
            ig_explainer=ig_exp,
            segment_map=seg_map,
            class_names=ECG_CLASS_NAMES,
        )
    except Exception:
        traceback.print_exc()
        return None


@st.cache_resource(show_spinner="Loading Image model...")
def load_image_bundle() -> Optional[ImageBundle]:
    try:
        from phase1.image.data_loader import load_cub200
        from phase1.image.model_trainer import load_or_finetune_resnet50, get_predict_fn
        from phase1.image.explainers.lime_image_explainer import LIMEImageExplainer
        from phase1.image.explainers.gradshap_explainer import GradientSHAPExplainer

        data    = load_cub200()
        model   = load_or_finetune_resnet50(data)
        p_fn    = get_predict_fn(model)

        lime_exp = LIMEImageExplainer(
            predict_fn=p_fn,
            baseline_image=data["baseline_image"],
        )
        gs_exp = GradientSHAPExplainer(
            model=model,
            background_images=data["background_images"],
        )

        return ImageBundle(
            data=data,
            model=model,
            predict_fn=p_fn,
            lime_explainer=lime_exp,
            gradshap_explainer=gs_exp,
            class_names=data["class_names"],
        )
    except Exception:
        traceback.print_exc()
        return None


@st.cache_resource(show_spinner="Loading Tabular model...")
def load_tabular_bundle() -> Optional[TabularBundle]:
    try:
        import joblib
        import shap
        import lime.lime_tabular as lime_tab
        import pandas as pd
        import pickle

        # Load pre-processed data
        if not Path(TABULAR_DATA_PATH).exists():
            st.warning("Tabular data not found — run run_phase1.py first.")
            return None

        with open(TABULAR_DATA_PATH, "rb") as f:
            data = pickle.load(f)

        # Load XGBoost model
        if not Path(TABULAR_MODEL_PATH).exists():
            st.warning("Tabular model not found — run run_phase1.py first.")
            return None

        model = joblib.load(TABULAR_MODEL_PATH)

        # Build predict function
        def predict_fn(X: np.ndarray) -> np.ndarray:
            if isinstance(X, pd.DataFrame):
                return model.predict_proba(X)
            df = pd.DataFrame(X, columns=ADULT_FEATURE_NAMES)
            return model.predict_proba(df)

        # LIME explainer using training data
        X_train = data.get("X_train", data.get("X_train_df"))
        if hasattr(X_train, "values"):
            X_train_np = X_train.values
        else:
            X_train_np = np.array(X_train)

        num_idx = list(range(len(ADULT_NUMERICAL_FEATURES)))
        cat_idx = list(range(len(ADULT_NUMERICAL_FEATURES), len(ADULT_FEATURE_NAMES)))

        lime_explainer = lime_tab.LimeTabularExplainer(
            training_data=X_train_np,
            feature_names=ADULT_FEATURE_NAMES,
            class_names=ADULT_CLASS_NAMES,
            categorical_features=cat_idx,
            mode="classification",
            discretize_continuous=False,
            random_state=42,
        )

        # SHAP explainer — TreeExplainer preferred, fall back to KernelExplainer
        # if model was saved with a different XGBoost version
        try:
            shap_explainer = shap.TreeExplainer(model)
        except Exception:
            background = shap.sample(X_train_np, 100)
            shap_explainer = shap.KernelExplainer(model.predict_proba, background)

        # ── Category labels for What-If dropdowns ─────────────────
        cat_labels: dict = {}
        preprocessor = data.get("preprocessor")
        if preprocessor is not None:
            try:
                cat_tf = preprocessor.named_transformers_["cat"]
                for feat, cats in zip(
                    ADULT_CATEGORICAL_FEATURES, cat_tf.categories_
                ):
                    cat_labels[feat] = [str(c) for c in cats.tolist()]
            except (AttributeError, KeyError):
                pass

        # ── Numerical feature ranges for sliders ──────────────────
        _FALLBACK_RANGES = {
            "age": (17.0, 90.0),
            "education-num": (1.0, 16.0),
            "capital-gain": (0.0, 99999.0),
            "capital-loss": (0.0, 4356.0),
            "hours-per-week": (1.0, 99.0),
        }
        feature_ranges: dict = {}
        df_ref = data.get("X_train_df")
        if df_ref is None and data.get("X_train") is not None:
            df_ref = pd.DataFrame(data["X_train"], columns=ADULT_FEATURE_NAMES)
        for feat in ADULT_NUMERICAL_FEATURES:
            if df_ref is not None and hasattr(df_ref, "columns") and feat in df_ref.columns:
                feature_ranges[feat] = (float(df_ref[feat].min()), float(df_ref[feat].max()))
            else:
                feature_ranges[feat] = _FALLBACK_RANGES[feat]

        return TabularBundle(
            data=data,
            model=model,
            lime_explainer=lime_explainer,
            shap_explainer=shap_explainer,
            feature_names=ADULT_FEATURE_NAMES,
            numerical_indices=num_idx,
            class_names=ADULT_CLASS_NAMES,
            cat_labels=cat_labels,
            feature_ranges=feature_ranges,
        )
    except Exception as e:
        traceback.print_exc()
        st.error(f"Tabular bundle failed: {type(e).__name__}: {e}")
        return None
