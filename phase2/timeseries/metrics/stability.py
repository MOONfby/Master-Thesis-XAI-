"""
Stability Metrics for Time Series XAI Evaluation.

Direct port of phase1/image/metrics/stability.py with:
  - perturb_image() → perturb_series()
  - Images (N, 3, H, W) → Series (N, 1, T)
  - segment_maps (N, H, W) → segment_maps (N, T)

The core Spearman ρ and Lipschitz formulas are IDENTICAL to Phase 1.
"""
import numpy as np
from scipy.stats import spearmanr

from phase2.timeseries.config import (
    STABILITY_N_PERTURBATIONS, STABILITY_NOISE_STD, RANDOM_STATE
)
from phase2.timeseries.utils import perturb_series


def rank_correlation_stability(explainer,
                                series_batch: np.ndarray,
                                segment_maps: np.ndarray,
                                n_perturbations: int = STABILITY_N_PERTURBATIONS,
                                noise_std: float = STABILITY_NOISE_STD):
    """
    Rank Correlation Stability under Gaussian noise perturbations.

    For each series, generates n_perturbations noisy copies, re-explains,
    and measures Spearman ρ between original and perturbed attribution ranks.

    Identical formula to phase1/image/metrics/stability.py.

    Parameters
    ----------
    explainer : LIMETSExplainer, TimeSHAPExplainer, or IntegratedGradientsExplainer
    series_batch : (N, 1, T)
    segment_maps : (N, T)
    n_perturbations : int
    noise_std : float

    Returns
    -------
    per_instance : np.ndarray (N,) — mean ρ per series
    mean_stability : float
    """
    rng = np.random.default_rng(RANDOM_STATE)
    N = len(series_batch)
    per_instance = np.zeros(N)

    for i in range(N):
        original_attrs = explainer.explain(series_batch[i], segment_maps[i])
        rhos = []
        for _ in range(n_perturbations):
            perturbed = perturb_series(series_batch[i], noise_std, rng)
            perturbed_attrs = explainer.explain(perturbed, segment_maps[i])
            rho, _ = spearmanr(original_attrs, perturbed_attrs)
            if not np.isnan(rho):
                rhos.append(rho)
        per_instance[i] = float(np.mean(rhos)) if rhos else 0.0

        if (i + 1) % 10 == 0:
            print(f"    Rank Correlation: {i+1}/{N} done "
                  f"(mean ρ so far = {per_instance[:i+1].mean():.4f})")

    return per_instance, float(per_instance.mean())


def average_sensitivity(explainer,
                        series_batch: np.ndarray,
                        segment_maps: np.ndarray,
                        n_perturbations: int = STABILITY_N_PERTURBATIONS,
                        noise_std: float = STABILITY_NOISE_STD):
    """
    Average Sensitivity (empirical Lipschitz constant).

    AS = mean[ ||E(x) - E(x')||_2 / ||x - x'||_2 ]

    Identical formula to phase1/image/metrics/stability.py.
    Input norms computed over all timesteps (flattened).

    Parameters
    ----------
    explainer : any TS explainer
    series_batch : (N, 1, T)
    segment_maps : (N, T)
    n_perturbations : int
    noise_std : float

    Returns
    -------
    per_instance : np.ndarray (N,)
    mean_sensitivity : float
    """
    rng = np.random.default_rng(RANDOM_STATE)
    N = len(series_batch)
    per_instance = np.zeros(N)

    for i in range(N):
        original_attrs = explainer.explain(series_batch[i], segment_maps[i])
        sensitivities = []
        for _ in range(n_perturbations):
            perturbed = perturb_series(series_batch[i], noise_std, rng)
            perturbed_attrs = explainer.explain(perturbed, segment_maps[i])

            delta_attr  = np.linalg.norm(original_attrs - perturbed_attrs)
            delta_input = np.linalg.norm(
                series_batch[i].flatten() - perturbed.flatten()
            )

            if delta_input > 1e-8:
                sensitivities.append(delta_attr / delta_input)

        per_instance[i] = float(np.mean(sensitivities)) if sensitivities else 0.0

        if (i + 1) % 10 == 0:
            print(f"    Avg Sensitivity: {i+1}/{N} done "
                  f"(mean so far = {per_instance[:i+1].mean():.4f})")

    return per_instance, float(per_instance.mean())
