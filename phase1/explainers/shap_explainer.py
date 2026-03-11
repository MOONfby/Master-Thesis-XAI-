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

    Root cause: SHAP's XGBTreeModelLoader calls booster.save_config(), parses
    the JSON, and does float(config["learner"]["learner_model_param"]["base_score"]).
    XGBoost 2.0 changed this value to a bracketed string, breaking the float().

    Fix: temporarily shadow booster.save_config with a closure that strips the
    brackets before SHAP reads it. This intercepts exactly the right call without
    modifying model weights or reloading any files.
    """
    import json

    def _make_explainer():
        if background is not None:
            return shap.TreeExplainer(model, data=background,
                                      feature_perturbation=feature_perturbation)
        return shap.TreeExplainer(model)

    # First attempt — works when SHAP >= 0.44 or XGBoost < 2.0
    try:
        return _make_explainer()
    except ValueError as e:
        if "could not convert string to float" not in str(e):
            raise

    warnings.warn(
        "XGBoost 2.0 / SHAP version mismatch detected (base_score format). "
        "Applying automatic patch. To silence: pip install --upgrade shap",
        UserWarning, stacklevel=3,
    )

    # Second attempt — patch booster.save_config as a Python instance attribute
    # so SHAP reads a corrected base_score value.
    booster = model.get_booster()
    _orig_save_config = booster.save_config  # bound method (callable)

    def _fixed_save_config():
        cfg = json.loads(_orig_save_config())

        def _fix(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == "base_score" and isinstance(v, str) and "[" in v:
                        # '[2.4783702E-1]'  ->  '0.24783702'
                        cleaned = re.sub(r"[\[\]\s]", "", v)
                        try:
                            obj[k] = str(float(cleaned))
                        except ValueError:
                            pass
                    else:
                        _fix(v)
            elif isinstance(obj, list):
                for item in obj:
                    _fix(item)

        _fix(cfg)
        return json.dumps(cfg)

    # Shadowing: instance attribute takes priority over class method in Python
    booster.save_config = _fixed_save_config
    try:
        return _make_explainer()
    finally:
        # Restore: removing the instance attribute re-exposes the class method
        try:
            del booster.save_config
        except AttributeError:
            pass


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
