"""
Stability Metrics for Image XAI Evaluation.

Directly adapts Phase 1 tabular stability metrics to image data.
The core formulas (Spearman rank correlation, average sensitivity)
are IDENTICAL — only the perturbation function changes:
  - Tabular: Gaussian noise on numerical feature columns only
  - Image:   Gaussian noise on ALL pixels, clipped to [0, 1]

This module imports the tabular metric functions directly and provides
an image-compatible wrapper for the perturbation step.
"""
import numpy as np
from scipy.stats import spearmanr

from phase1.image.config import (
    STABILITY_N_PERTURBATIONS, STABILITY_NOISE_STD, RANDOM_STATE
)
from phase1.image.utils import perturb_image


def rank_correlation_stability(explainer,
                                images: np.ndarray,
                                segment_maps: np.ndarray,
                                n_perturbations: int = STABILITY_N_PERTURBATIONS,
                                noise_std: float = STABILITY_NOISE_STD):
    """
    Rank Correlation Stability under pixel noise perturbations.

    For each image, generates n_perturbations noisy copies, re-explains,
    and measures Spearman ρ between original and perturbed attribution ranks.

    Identical formula to phase1/tabular/metrics/stability.py:rank_correlation_stability().
    Difference: noise applied to all pixels (no num_indices filtering).

    Parameters
    ----------
    explainer : LIMEImageExplainer or GradientSHAPExplainer
    images : (N, 3, H, W)
    segment_maps : (N, H, W)
    n_perturbations : int
    noise_std : float

    Returns
    -------
    per_instance : np.ndarray (N,)  — mean ρ per image
    mean_stability : float
    """
    rng = np.random.default_rng(RANDOM_STATE)
    N = len(images)
    per_instance = np.zeros(N)

    for i in range(N):
        original_attrs = explainer.explain(images[i], segment_maps[i])
        rhos = []
        for _ in range(n_perturbations):
            perturbed = perturb_image(images[i], noise_std, rng)
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
                        images: np.ndarray,
                        segment_maps: np.ndarray,
                        n_perturbations: int = STABILITY_N_PERTURBATIONS,
                        noise_std: float = STABILITY_NOISE_STD):
    """
    Average Sensitivity (empirical Lipschitz constant).

    AS = mean[ ||E(x) - E(x')||_2 / ||x - x'||_2 ]

    where x' = x + noise, and E is the explainer.

    Identical formula to phase1/tabular/metrics/stability.py:average_sensitivity().
    Input norms computed over all pixels (flattened).

    Parameters
    ----------
    explainer : LIMEImageExplainer or GradientSHAPExplainer
    images : (N, 3, H, W)
    segment_maps : (N, H, W)
    n_perturbations : int
    noise_std : float

    Returns
    -------
    per_instance : np.ndarray (N,)
    mean_sensitivity : float
    """
    rng = np.random.default_rng(RANDOM_STATE)
    N = len(images)
    per_instance = np.zeros(N)

    for i in range(N):
        original_attrs = explainer.explain(images[i], segment_maps[i])
        sensitivities = []
        for _ in range(n_perturbations):
            perturbed = perturb_image(images[i], noise_std, rng)
            perturbed_attrs = explainer.explain(perturbed, segment_maps[i])

            delta_attr  = np.linalg.norm(original_attrs - perturbed_attrs)
            delta_input = np.linalg.norm(images[i].flatten() - perturbed.flatten())

            if delta_input > 1e-8:
                sensitivities.append(delta_attr / delta_input)

        per_instance[i] = float(np.mean(sensitivities)) if sensitivities else 0.0

        if (i + 1) % 10 == 0:
            print(f"    Avg Sensitivity: {i+1}/{N} done "
                  f"(mean so far = {per_instance[:i+1].mean():.4f})")

    return per_instance, float(per_instance.mean())
