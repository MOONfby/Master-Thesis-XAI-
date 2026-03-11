"""
SHAP (SHapley Additive exPlanations) wrapper using TreeExplainer.

SHAP uses Shapley values from cooperative game theory to attribute the
prediction fairly among features. For tree models (XGBoost), TreeExplainer
computes exact SHAP values in O(TLD²) time — far faster than the model-agnostic
KernelSHAP which requires O(2^n) exponential evaluation.

Reference: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions",
NeurIPS 2017.

Design note:
- TreeExplainer with feature_perturbation='interventional' uses background data
  to approximate E[f(x) | x_S] via conditional sampling, which is theoretically
  correct but slower.
- Default ('tree_path_dependent') is faster and well-suited for tree models.
- We use the default here; interventional can be enabled via parameter.
"""
import re
import time
import warnings
import numpy as np
import shap

from phase1.config import SHAP_BACKGROUND_SAMPLES, FEATURE_NAMES


def _build_tree_explainer(model, X_train: np.ndarray,
                          feature_perturbation: str = "tree_path_dependent",
                          background: np.ndarray = None):
    """
    Build a shap.TreeExplainer, working around the XGBoost 2.0 / SHAP < 0.44
    incompatibility where base_score is serialized as '[2.4783702E-1]' instead
    of a plain float.

    Fix hierarchy:
      1. Try shap.TreeExplainer directly (works when versions are compatible).
      2. If ValueError on base_score, patch the XGBoost booster config in-place
         and retry — preserves exact TreeSHAP without falling back to KernelSHAP.
    """
    try:
        if background is not None:
            return shap.TreeExplainer(model, data=background,
                                      feature_perturbation=feature_perturbation)
        return shap.TreeExplainer(model)
    except ValueError as e:
        if "could not convert string to float" not in str(e):
            raise

    warnings.warn(
        "shap.TreeExplainer failed due to XGBoost 2.0 / SHAP version mismatch "
        "(base_score format changed). Applying automatic patch. "
        "To silence this warning: pip install --upgrade shap",
        UserWarning, stacklevel=3,
    )

    # --- Patch: strip the brackets from base_score in the booster JSON config ---
    import json
    booster = model.get_booster()
    cfg = json.loads(booster.save_config())

    def _fix_base_score(obj):
        """Recursively find and normalize bracketed base_score strings."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "base_score" and isinstance(v, str):
                    # '[2.4783702E-1]'  ->  '0.24783702'
                    cleaned = re.sub(r"[\[\]]", "", v).strip()
                    try:
                        obj[k] = str(float(cleaned))
                    except ValueError:
                        pass
                else:
                    _fix_base_score(v)
        elif isinstance(obj, list):
            for item in obj:
                _fix_base_score(item)

    _fix_base_score(cfg)
    booster.load_config(json.dumps(cfg))

    # Reload the patched booster into a fresh XGBClassifier copy
    import copy, tempfile, os
    model_patched = copy.copy(model)
    with tempfile.NamedTemporaryFile(suffix=".ubj", delete=False) as f:
        tmp = f.name
    booster.save_model(tmp)
    model_patched.load_model(tmp)
    os.unlink(tmp)

    if background is not None:
        return shap.TreeExplainer(model_patched, data=background,
                                  feature_perturbation=feature_perturbation)
    return shap.TreeExplainer(model_patched)


class SHAPExplainer:
    """
    Wrapper around shap.TreeExplainer for XGBoost models.

    Returns SHAP values for class 1 (>50K) by default.
    """

    def __init__(self, model, X_train: np.ndarray,
                 feature_names: list[str] = None,
                 feature_perturbation: str = "tree_path_dependent"):
        """
        Parameters
        ----------
        model       : fitted XGBClassifier
        X_train     : training data for background distribution
        feature_names : feature names list
        feature_perturbation : 'tree_path_dependent' (fast) or 'interventional'
        """
        self.model = model
        self.feature_names = feature_names or FEATURE_NAMES

        if feature_perturbation == "interventional":
            # Sample a background subset for E[f(x) | x_S] estimation
            idx = np.random.choice(len(X_train), size=min(SHAP_BACKGROUND_SAMPLES, len(X_train)),
                                   replace=False)
            background = X_train[idx]
            self.explainer = _build_tree_explainer(
                model, X_train, feature_perturbation="interventional",
                background=background,
            )
        else:
            self.explainer = _build_tree_explainer(model, X_train)

        self.expected_value = self.explainer.expected_value
        # For binary classification, expected_value may be a list [neg, pos]
        if isinstance(self.expected_value, (list, np.ndarray)):
            self.base_value = float(self.expected_value[1])
        else:
            self.base_value = float(self.expected_value)

    def explain(self, instance: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for a single instance.

        Returns
        -------
        shap_values : np.ndarray, shape (n_features,)
            SHAP values for the positive class (>50K).
        """
        sv = self.explainer.shap_values(instance.reshape(1, -1))
        # TreeExplainer on XGBoost returns array of shape (1, n_features)
        # or list [neg_class, pos_class] depending on version
        if isinstance(sv, list):
            return sv[1][0]
        return sv[0]

    def explain_batch(self, instances: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for multiple instances (vectorized, much faster).

        Returns
        -------
        shap_values : np.ndarray, shape (n_instances, n_features)
        """
        sv = self.explainer.shap_values(instances)
        if isinstance(sv, list):
            return sv[1]
        return sv

    def measure_runtime(self, instances: np.ndarray) -> dict:
        """
        Measure per-instance explanation time.

        Note: SHAP TreeExplainer is highly optimized; batch is much faster than
        individual calls due to vectorization.
        """
        # Per-instance timing
        times = []
        for inst in instances:
            t0 = time.perf_counter()
            self.explain(inst)
            times.append(time.perf_counter() - t0)

        # Batch timing
        t0 = time.perf_counter()
        self.explain_batch(instances)
        batch_time = time.perf_counter() - t0

        return {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "total_time": np.sum(times),
            "batch_total_time": batch_time,
            "n_instances": len(instances),
        }
