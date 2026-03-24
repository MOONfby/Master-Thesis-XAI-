"""
Integrated Gradients Explainer for Time Series Classification.

Uses captum.attr.IntegratedGradients to compute per-timestep attributions
for InceptionTime. Attributions are aggregated to temporal segment level.

Integrated Gradients (Sundararajan et al. 2017):
    IG(x)_i = (x_i - x'_i) * integral_0^1 [ dF/dx_i at x' + alpha*(x-x') ] dalpha

Baseline x' = zero series (or per-channel training mean series).

Phase 1 analogue: GradientSHAP (image) — both are gradient-based, fast,
model-specific methods requiring differentiable architectures.
"""
import time
import numpy as np
import torch

from phase2.timeseries.config import N_SEGMENTS_TS, IG_N_STEPS
from phase2.timeseries.utils import aggregate_to_segments, build_segment_map


class IntegratedGradientsExplainer:
    """
    Integrated Gradients explanation for time series classification.

    Produces temporal-segment-level attribution scores in (S,) format,
    comparable with LIMETSExplainer and TimeSHAPExplainer output.

    Parameters
    ----------
    model : torch.nn.Module
        InceptionTime in eval mode.
    baseline_series : np.ndarray, shape (1, T), float32
        Baseline input (per-channel training mean). Attribution is computed
        relative to this reference.
    n_steps : int
        Number of Riemann approximation steps for the integral.
    """

    def __init__(self, model: torch.nn.Module,
                 baseline_series: np.ndarray,
                 n_steps: int = IG_N_STEPS):
        self.model    = model
        self.device   = next(model.parameters()).device
        self._baseline = torch.tensor(baseline_series[np.newaxis],
                                      dtype=torch.float32,
                                      device=self.device)   # (1, 1, T)
        self.n_steps   = n_steps

    # ------------------------------------------------------------------
    # Core interface — mirrors GradientSHAPExplainer
    # ------------------------------------------------------------------

    def explain(self, series: np.ndarray, segment_map: np.ndarray,
                label: int = None) -> np.ndarray:
        """
        Explain a single time series.

        Parameters
        ----------
        series : np.ndarray, shape (1, T), normalised float32
        segment_map : np.ndarray, shape (T,), int
        label : int or None — class to explain; None = argmax

        Returns
        -------
        np.ndarray, shape (N_SEGMENTS_TS,)
            Segment-level IG attributions (absolute, mean per segment).
        """
        ts_attr = self.explain_timestep_level(series, label=label)
        return aggregate_to_segments(np.abs(ts_attr), segment_map,
                                     n_segments=N_SEGMENTS_TS)

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
                print(f"    IntegratedGradients: {i+1}/{N} done")
        return np.stack(results)

    def explain_timestep_level(self, series: np.ndarray,
                                segment_map: np.ndarray = None,
                                label: int = None) -> np.ndarray:
        """
        Return per-timestep IG attributions, shape (T,).

        Sums absolute values across channels to produce a scalar per timestep.
        """
        try:
            from captum.attr import IntegratedGradients
        except ImportError:
            raise ImportError(
                "captum is required for IntegratedGradients. "
                "Install with: pip install captum"
            )

        x = torch.tensor(series[np.newaxis], dtype=torch.float32,
                         device=self.device)   # (1, 1, T)

        if label is None:
            with torch.no_grad():
                logits = self.model(x)
            label = int(logits[0].argmax().item())

        # captum requires gradient-enabled input
        x.requires_grad_(True)

        ig = IntegratedGradients(self.model)
        attr = ig.attribute(
            x,
            baselines=self._baseline,
            target=label,
            n_steps=self.n_steps,
            method="gausslegendre",
        )   # (1, 1, T)

        attr_np = attr.detach().cpu().numpy()[0]   # (1, T)
        # Sum absolute values across channels → (T,)
        ts_attr = np.abs(attr_np).sum(axis=0)
        return ts_attr

    def measure_runtime(self, series_batch: np.ndarray,
                        segment_maps: np.ndarray) -> dict:
        """
        Measure per-series explanation time.

        Returns
        -------
        dict with mean_time, std_time, total_time, n_instances
        """
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
