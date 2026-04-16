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

        # SHAP TreeExplainer
        shap_explainer = shap.TreeExplainer(model)

        return TabularBundle(
            data=data,
            model=model,
            lime_explainer=lime_explainer,
            shap_explainer=shap_explainer,
            feature_names=ADULT_FEATURE_NAMES,
            numerical_indices=num_idx,
            class_names=ADULT_CLASS_NAMES,
        )
    except ImportError:
        traceback.print_exc()
        return None
    except Exception:
        traceback.print_exc()
        return None
