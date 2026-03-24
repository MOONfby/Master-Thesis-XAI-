"""
TimeSHAP Explainer for Phase 2 Time Series XAI Evaluation.

TimeSHAP (Bento et al. 2021) computes Shapley values for sequential models
by treating each temporal segment as a player in a coalition game.

Approach used here:
  - Treat each temporal segment (not individual timestep) as a SHAP player.
  - Use KernelSHAP (shap.KernelExplainer) on the segment-level binary feature
    space — analogous to how LIME treats segments as binary features, but with
    exact Shapley value guarantees.
  - Background: mean prediction over a subset of training data (SHAP baseline).

This is a pragmatic adaptation: true TimeSHAP uses pruning and event-level
coalitions for RNNs; here we apply KernelSHAP at the segment level which is
model-agnostic and works for InceptionTime.

Phase 1 analogue: SHAPExplainer (tabular, TreeExplainer) and GradientSHAP
(image) — all belong to the SHAP family with Shapley value guarantees.
"""
import time
import numpy as np

from phase2.timeseries.config import N_SEGMENTS_TS, RANDOM_STATE
from phase2.timeseries.utils import apply_temporal_mask


class TimeSHAPExplainer:
    """
    Segment-level KernelSHAP explanation for time series classification.

    Treats each temporal segment as a binary coalition player.
    Present = original segment values; absent = baseline values.

    Parameters
    ----------
    predict_fn : callable
        (np.ndarray (N, 1, T)) -> np.ndarray (N, num_classes)
    baseline_series : np.ndarray, shape (1, T), float32
        Per-channel training mean series (SHAP background/reference).
    n_segments : int
        Number of temporal segments.
    n_background : int
        Number of background samples for KernelSHAP expected value estimation.
    """

    def __init__(self, predict_fn, baseline_series: np.ndarray,
                 n_segments: int = N_SEGMENTS_TS,
                 n_background: int = 50):
        self.predict_fn      = predict_fn
        self.baseline_series = baseline_series
        self.n_segments      = n_segments
        self.n_background    = n_background
        self._current_series = None     # set per-call for the predict wrapper
        self._current_smap   = None

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def explain(self, series: np.ndarray, segment_map: np.ndarray,
                label: int = None) -> np.ndarray:
        """
        Explain a single time series using KernelSHAP over segments.

        Parameters
        ----------
        series : np.ndarray, shape (1, T), normalised float32
        segment_map : np.ndarray, shape (T,), int
        label : int or None — class to explain; None = argmax

        Returns
        -------
        np.ndarray, shape (N_SEGMENTS_TS,)
            Shapley values per temporal segment.
        """
        try:
            import shap
        except ImportError:
            raise ImportError("shap is required. Install with: pip install shap")

        S = self.n_segments

        if label is None:
            probs = self.predict_fn(series[np.newaxis])
            label = int(probs[0].argmax())

        # Store for the predict wrapper
        self._current_series = series
        self._current_smap   = segment_map
        self._target_label   = label

        # Background: single all-zeros binary vector (all segments absent)
        # This means the baseline prediction = f(baseline_series)
        background = np.zeros((1, S), dtype=np.float32)

        explainer = shap.KernelExplainer(
            self._segment_predict_fn,
            background,
        )

        # The instance to explain: all segments present
        instance = np.ones((1, S), dtype=np.float32)

        shap_values = explainer.shap_values(
            instance,
            nsamples=self.n_background,
            silent=True,
        )

        # shap_values may be list (multi-output) or array
        if isinstance(shap_values, list):
            sv = shap_values[label][0]    # class label, first instance
        else:
            sv = shap_values[0]

        return sv.astype(np.float64)

    def explain_batch(self, series_batch: np.ndarray,
                      segment_maps: np.ndarray,
                      labels: np.ndarray = None) -> np.ndarray:
        """
        Explain a batch of time series.

        Parameters
        ----------
        series_batch : np.ndarray, shape (N, 1, T)
        segment_maps : np.ndarray, shape (N, T)
        labels : np.ndarray (N,) int or None

        Returns
        -------
        np.ndarray, shape (N, N_SEGMENTS_TS)
        """
        N = len(series_batch)
        results = []
        for i in range(N):
            lbl = int(labels[i]) if labels is not None else None
            attrs = self.explain(series_batch[i], segment_maps[i], label=lbl)
            results.append(attrs)
            if (i + 1) % 10 == 0:
                print(f"    TimeSHAP: {i+1}/{N} done")
        return np.stack(results)

    def explain_timestep_level(self, series: np.ndarray,
                                segment_map: np.ndarray,
                                label: int = None) -> np.ndarray:
        """
        Return timestep-level heatmap by broadcasting segment SHAP values.

        Used for temporal localisation metrics.

        Returns
        -------
        np.ndarray, shape (T,)
        """
        attrs = self.explain(series, segment_map, label=label)
        T = series.shape[-1]
        heatmap = np.zeros(T, dtype=np.float64)
        for s, a in enumerate(attrs):
            heatmap[segment_map == s] = a
        return heatmap

    def measure_runtime(self, series_batch: np.ndarray,
                        segment_maps: np.ndarray) -> dict:
        """Measure per-series explanation time."""
        times = []
        for i in range(len(series_batch)):
            t0 = time.perf_counter()
            self.explain(series_batch[i], segment_maps[i])
            times.append(time.perf_counter() - t0)
        times = np.array(times)
        return {
            "mean_time":   float(times.mean()),
            "std_time":    float(times.std()),
            "total_time":  float(times.sum()),
            "n_instances": len(times),
        }

    # ------------------------------------------------------------------
    # Internal: wraps predict_fn for KernelSHAP's binary segment input
    # ------------------------------------------------------------------

    def _segment_predict_fn(self, z_batch: np.ndarray) -> np.ndarray:
        """
        KernelSHAP calls this with binary segment masks z ∈ {0,1}^S.

        Maps each binary mask to a perturbed series and calls predict_fn.

        Parameters
        ----------
        z_batch : np.ndarray, shape (M, S)

        Returns
        -------
        np.ndarray, shape (M, num_classes) — probabilities
        """
        M = len(z_batch)
        perturbed = []
        for z in z_batch:
            absent = np.where(z == 0)[0].tolist()
            s = apply_temporal_mask(
                self._current_series,
                self._current_smap,
                absent,
                self.baseline_series,
            )
            perturbed.append(s)
        perturbed_batch = np.stack(perturbed)   # (M, 1, T)
        return self.predict_fn(perturbed_batch)  # (M, C)
