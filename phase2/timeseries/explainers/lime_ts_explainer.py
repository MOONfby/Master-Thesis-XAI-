"""
LIME Time Series Explainer for Phase 2 Time Series XAI Evaluation.

Adapts lime.lime_tabular.LimeTabularExplainer (or a custom perturbation loop)
to explain time series classifiers at the temporal segment level.

Design: Each temporal segment is treated as a binary "feature" (present/absent).
LIME perturbs by setting absent segments to their baseline values, fitting a
sparse linear model on the resulting neighbourhood. This is the same principle
as LIMEImageExplainer but over temporal segments instead of superpixels.

Phase 1 analogue: LIMEImageExplainer — identical interface, temporal masking
replaces superpixel masking.
"""
import time
import numpy as np

from phase2.timeseries.config import (
    N_SEGMENTS_TS, LIME_N_SAMPLES, RANDOM_STATE
)
from phase2.timeseries.utils import apply_temporal_mask


class LIMETSExplainer:
    """
    LIME explanation for time series classification via temporal segmentation.

    Produces temporal-segment-level attribution scores in (S,) format,
    directly comparable with IntegratedGradientsExplainer output.

    Parameters
    ----------
    predict_fn : callable
        (np.ndarray (N, 1, T)) -> np.ndarray (N, num_classes)
    baseline_series : np.ndarray, shape (1, T), float32
        Per-channel mean series used to mask (hide) temporal segments.
    n_segments : int
        Number of temporal segments (must match segment_map used at explain time).
    """

    def __init__(self, predict_fn, baseline_series: np.ndarray,
                 n_segments: int = N_SEGMENTS_TS):
        self.predict_fn      = predict_fn
        self.baseline_series = baseline_series   # (1, T)
        self.n_segments      = n_segments
        self._rng            = np.random.default_rng(RANDOM_STATE)

    # ------------------------------------------------------------------
    # Core interface — mirrors LIMEImageExplainer
    # ------------------------------------------------------------------

    def explain(self, series: np.ndarray, segment_map: np.ndarray,
                label: int = None,
                num_samples: int = LIME_N_SAMPLES) -> np.ndarray:
        """
        Explain a single time series.

        Parameters
        ----------
        series : np.ndarray, shape (1, T), normalised float32
        segment_map : np.ndarray, shape (T,), int — temporal segment labels
        label : int or None — class to explain; None = argmax
        num_samples : int — number of LIME neighbourhood samples

        Returns
        -------
        np.ndarray, shape (N_SEGMENTS_TS,)
            LIME attribution weights per temporal segment.
        """
        S = self.n_segments

        if label is None:
            probs = self.predict_fn(series[np.newaxis])   # (1, C)
            label = int(probs[0].argmax())

        # Build neighbourhood: binary masks over segments
        # z[i] ∈ {0,1}^S where 1 = segment present, 0 = replaced by baseline
        z_on = np.ones((1, S), dtype=np.float32)   # original (all present)

        # Sample random binary vectors (neighbourhood)
        z_samples = self._rng.integers(0, 2, size=(num_samples, S)).astype(np.float32)
        z_all = np.vstack([z_on, z_samples])   # (num_samples+1, S)

        # Map binary masks to perturbed series
        perturbed_series = []
        for z in z_all:
            absent = np.where(z == 0)[0].tolist()
            s = apply_temporal_mask(series, segment_map, absent,
                                    self.baseline_series)
            perturbed_series.append(s)

        perturbed_batch = np.stack(perturbed_series)   # (num_samples+1, 1, T)

        # Get model predictions for target class
        preds = self.predict_fn(perturbed_batch)       # (num_samples+1, C)
        y_local = preds[:, label]                      # (num_samples+1,)

        # Kernel weights: cosine similarity to the original (z_on)
        # Simplified: use exponential kernel on Hamming distance
        weights = self._kernel_weights(z_all, z_on[0])

        # Fit weighted ridge regression: z → y_local
        from sklearn.linear_model import Ridge
        reg = Ridge(alpha=1.0, fit_intercept=True)
        reg.fit(z_all, y_local, sample_weight=weights)

        attrs = reg.coef_.astype(np.float64)   # (S,)
        return attrs

    def explain_batch(self, series_batch: np.ndarray,
                      segment_maps: np.ndarray,
                      labels: np.ndarray = None,
                      num_samples: int = LIME_N_SAMPLES) -> np.ndarray:
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
            attrs = self.explain(series_batch[i], segment_maps[i],
                                 label=lbl, num_samples=num_samples)
            results.append(attrs)
            if (i + 1) % 10 == 0:
                print(f"    LIME-TS: {i+1}/{N} done")
        return np.stack(results)

    def explain_timestep_level(self, series: np.ndarray,
                                segment_map: np.ndarray,
                                label: int = None) -> np.ndarray:
        """
        Return timestep-level heatmap by broadcasting segment attributions.

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
                        segment_maps: np.ndarray,
                        num_samples: int = LIME_N_SAMPLES) -> dict:
        """Measure per-series explanation time."""
        times = []
        for i in range(len(series_batch)):
            t0 = time.perf_counter()
            self.explain(series_batch[i], segment_maps[i],
                         num_samples=num_samples)
            times.append(time.perf_counter() - t0)
        times = np.array(times)
        return {
            "mean_time":   float(times.mean()),
            "std_time":    float(times.std()),
            "total_time":  float(times.sum()),
            "n_instances": len(times),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _kernel_weights(self, z_samples: np.ndarray,
                        z_ref: np.ndarray,
                        kernel_width: float = None) -> np.ndarray:
        """
        Exponential kernel on Hamming distance from reference.

        w = exp(-d² / (2 * kernel_width²))
        kernel_width defaults to sqrt(n_segments) * 0.75 (LIME default).
        """
        S = z_ref.shape[0]
        if kernel_width is None:
            kernel_width = np.sqrt(S) * 0.75
        distances = np.sum(z_samples != z_ref, axis=1).astype(float)
        weights = np.exp(-(distances ** 2) / (2 * kernel_width ** 2))
        return weights
