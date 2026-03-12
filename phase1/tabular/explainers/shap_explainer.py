"""
SHAP (SHapley Additive exPlanations) wrapper.

Primary computation uses XGBoost's built-in pred_contribs=True, which calls
XGBoost's own C++ TreeSHAP implementation directly — identical algorithm to
shap.TreeExplainer, no Python-layer version dependency.

Fallback: shap.TreeExplainer is tried first for probability-space values;
if it fails due to the XGBoost 2.0 / SHAP < 0.44 base_score format mismatch,
we switch to the native XGBoost path automatically.

Native pred_contribs returns values in log-odds (margin) space for binary
classification, which correctly ranks features and is sufficient for all
evaluation metrics (AOPC, comprehensiveness, sufficiency, stability).

Reference: Lundberg & Lee, "A Unified Approach to Interpreting Model
Predictions", NeurIPS 2017.
"""
import time
import warnings
import numpy as np
import xgboost as xgb
import shap

from phase1.tabular.config import SHAP_BACKGROUND_SAMPLES, FEATURE_NAMES


class SHAPExplainer:
    """
    SHAP explainer for XGBoost, with automatic fallback to native pred_contribs.

    Attributes
    ----------
    base_value   : float  — expected SHAP output (bias term)
    feature_names: list[str]
    _use_native  : bool   — True when using XGBoost native (not shap library)
    """

    def __init__(self, model, X_train: np.ndarray,
                 feature_names: list[str] = None):
        """
        Parameters
        ----------
        model         : fitted XGBClassifier
        X_train       : training data (encoded), used for base value estimation
        feature_names : feature names list
        """
        self.model = model
        self.booster = model.get_booster()
        self.feature_names = feature_names or FEATURE_NAMES

        # --- Attempt 1: shap.TreeExplainer (exact, probability-space values) ---
        try:
            self._shap_exp = shap.TreeExplainer(model)
            self._use_native = False
            ev = self._shap_exp.expected_value
            self.base_value = float(ev[1] if isinstance(ev, (list, np.ndarray)) else ev)
            print("    SHAP: using shap.TreeExplainer")
            return
        except ValueError as e:
            if "could not convert string to float" not in str(e):
                raise

        # --- Attempt 2: XGBoost native pred_contribs (version-independent) ---
        warnings.warn(
            "shap.TreeExplainer unavailable due to XGBoost 2.0 / SHAP version "
            "mismatch. Using XGBoost native TreeSHAP (pred_contribs=True) — "
            "same algorithm, log-odds space. Fix: pip install --upgrade shap",
            UserWarning,
        )
        self._shap_exp = None
        self._use_native = True

        # Estimate base value from training data bias terms
        n_bg = min(SHAP_BACKGROUND_SAMPLES, len(X_train))
        dmat = xgb.DMatrix(X_train[:n_bg], feature_names=self.feature_names)
        contribs = self.booster.predict(dmat, pred_contribs=True)
        # contribs shape: (n_samples, n_features + 1); last col = bias
        self.base_value = float(contribs[:, -1].mean())
        print("    SHAP: using XGBoost native pred_contribs (log-odds space)")

    # ------------------------------------------------------------------

    def explain(self, instance: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for a single instance.

        Returns
        -------
        shap_values : np.ndarray, shape (n_features,)
        """
        if self._use_native:
            dmat = xgb.DMatrix(instance.reshape(1, -1),
                               feature_names=self.feature_names)
            contribs = self.booster.predict(dmat, pred_contribs=True)
            return contribs[0, :-1]  # drop bias column

        sv = self._shap_exp.shap_values(instance.reshape(1, -1))
        if isinstance(sv, list):
            return sv[1][0]
        return sv[0]

    def explain_batch(self, instances: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for multiple instances.

        Returns
        -------
        shap_values : np.ndarray, shape (n_instances, n_features)
        """
        if self._use_native:
            dmat = xgb.DMatrix(instances, feature_names=self.feature_names)
            contribs = self.booster.predict(dmat, pred_contribs=True)
            return contribs[:, :-1]  # drop bias column

        sv = self._shap_exp.shap_values(instances)
        if isinstance(sv, list):
            return sv[1]
        return sv

    def measure_runtime(self, instances: np.ndarray) -> dict:
        """Measure per-instance and batch explanation time."""
        times = []
        for inst in instances:
            t0 = time.perf_counter()
            self.explain(inst)
            times.append(time.perf_counter() - t0)

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
