"""
Counterfactual Explanation Metrics.

Counterfactual explanations are evaluated differently from feature attribution
methods (LIME/SHAP) because they provide contrastive rather than contributive
explanations. The quality criteria are:

1. Validity — Are the counterfactuals actually valid (do they flip the prediction)?
   validity = (# CFs where predicted_class == desired_class) / total_CFs
   Higher is better; invalid CFs are useless.

2. Proximity (L1, L2) — How close is the counterfactual to the original instance?
   Measures the cost of the change. Lower = more actionable.
   - L1 normalized: mean |CF_i - x_i| / feature_range_i  (robust to scale)
   - L2 normalized: sqrt( mean (CF_i - x_i)^2 / feature_range_i^2 )

3. Sparsity — How many features need to change?
   sparsity = (# features changed) / (# total features)
   Lower = fewer changes needed = simpler, more interpretable explanation.
   A change is detected when |CF_i - x_i| > ε (small threshold).

4. Diversity — How diverse are the generated counterfactuals?
   diversity = mean pairwise L2 distance between CFs / range
   Higher = CFs cover different parts of the input space (more informative).
   DiCE is specifically designed to maximize diversity by construction.

References:
  Mothilal et al. (2020) "Explaining Machine Learning Classifiers through
  Diverse Counterfactual Explanations." FAccT 2020.
  Wachter et al. (2017) "Counterfactual Explanations Without Opening the
  Black Box." Harvard JLAT.
"""
import numpy as np
import pandas as pd
from typing import Optional


def _normalize(values: np.ndarray, feature_ranges: np.ndarray) -> np.ndarray:
    """Normalize feature values by feature range (max - min of training data)."""
    return values / (feature_ranges + 1e-8)


def cf_validity(cf_results: list[dict]) -> float:
    """
    Fraction of instances for which at least one valid CF was found.

    Parameters
    ----------
    cf_results : list of dicts from DiCEExplainer.generate_batch()

    Returns
    -------
    validity_rate : float in [0, 1]
    """
    valid = sum(1 for r in cf_results if r.get("success") and r.get("validity", 0) > 0)
    return valid / len(cf_results) if cf_results else 0.0


def cf_proximity_l1(
    instances: np.ndarray,
    cf_results: list[dict],
    feature_ranges: Optional[np.ndarray] = None,
) -> tuple[float, np.ndarray]:
    """
    Mean normalized L1 proximity between originals and counterfactuals.

    Returns
    -------
    mean_l1  : float (lower = more actionable)
    l1_per_instance : np.ndarray
    """
    l1_scores = []
    for i, result in enumerate(cf_results):
        if not result.get("success") or result.get("cf_df") is None:
            continue
        cf_array = result["cf_df"].values.astype(np.float64)
        instance = instances[i]
        diffs = np.abs(cf_array - instance)  # (num_cfs, n_features)
        if feature_ranges is not None:
            diffs = _normalize(diffs, feature_ranges)
        l1_per_cf = diffs.mean(axis=1)       # mean across features
        l1_scores.append(l1_per_cf.mean())   # mean across CFs

    return (float(np.mean(l1_scores)) if l1_scores else np.nan,
            np.array(l1_scores))


def cf_proximity_l2(
    instances: np.ndarray,
    cf_results: list[dict],
    feature_ranges: Optional[np.ndarray] = None,
) -> tuple[float, np.ndarray]:
    """
    Mean normalized L2 proximity between originals and counterfactuals.

    Returns
    -------
    mean_l2  : float (lower = more actionable)
    l2_per_instance : np.ndarray
    """
    l2_scores = []
    for i, result in enumerate(cf_results):
        if not result.get("success") or result.get("cf_df") is None:
            continue
        cf_array = result["cf_df"].values.astype(np.float64)
        instance = instances[i]
        diffs = cf_array - instance
        if feature_ranges is not None:
            diffs = _normalize(diffs, feature_ranges)
        l2_per_cf = np.linalg.norm(diffs, axis=1)
        l2_scores.append(l2_per_cf.mean())

    return (float(np.mean(l2_scores)) if l2_scores else np.nan,
            np.array(l2_scores))


def cf_sparsity(
    instances: np.ndarray,
    cf_results: list[dict],
    epsilon: float = 1e-3,
) -> tuple[float, np.ndarray]:
    """
    Mean fraction of features changed in counterfactuals.

    Parameters
    ----------
    epsilon : threshold below which a change is considered zero

    Returns
    -------
    mean_sparsity : float in [0, 1] (lower = fewer features changed)
    sparsity_per_instance : np.ndarray
    """
    n_features = instances.shape[1]
    sparsity_scores = []
    for i, result in enumerate(cf_results):
        if not result.get("success") or result.get("cf_df") is None:
            continue
        cf_array = result["cf_df"].values.astype(np.float64)
        instance = instances[i]
        changed = (np.abs(cf_array - instance) > epsilon).mean(axis=1)  # per CF
        sparsity_scores.append(changed.mean())

    return (float(np.mean(sparsity_scores)) if sparsity_scores else np.nan,
            np.array(sparsity_scores))


def cf_diversity(
    cf_results: list[dict],
    feature_ranges: Optional[np.ndarray] = None,
) -> tuple[float, np.ndarray]:
    """
    Mean pairwise normalized L2 distance between counterfactuals.

    Measures how spread out the generated CFs are in feature space.
    Higher diversity = more informative set of alternatives.

    Returns
    -------
    mean_diversity : float (higher = more diverse)
    diversity_per_instance : np.ndarray
    """
    diversity_scores = []
    for result in cf_results:
        if not result.get("success") or result.get("cf_df") is None:
            continue
        cf_array = result["cf_df"].values.astype(np.float64)
        if len(cf_array) < 2:
            diversity_scores.append(0.0)
            continue
        # Pairwise L2
        n = len(cf_array)
        pairwise = []
        for a in range(n):
            for b in range(a + 1, n):
                diff = cf_array[a] - cf_array[b]
                if feature_ranges is not None:
                    diff = _normalize(diff, feature_ranges)
                pairwise.append(np.linalg.norm(diff))
        diversity_scores.append(np.mean(pairwise))

    return (float(np.mean(diversity_scores)) if diversity_scores else np.nan,
            np.array(diversity_scores))


def evaluate_counterfactuals(
    instances: np.ndarray,
    cf_results: list[dict],
    feature_ranges: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute all counterfactual metrics in one call.

    Returns
    -------
    dict with keys: validity, proximity_l1, proximity_l2, sparsity, diversity
    """
    val = cf_validity(cf_results)
    l1, _ = cf_proximity_l1(instances, cf_results, feature_ranges)
    l2, _ = cf_proximity_l2(instances, cf_results, feature_ranges)
    spar, _ = cf_sparsity(instances, cf_results)
    div, _ = cf_diversity(cf_results, feature_ranges)

    return {
        "validity": val,
        "proximity_l1": l1,
        "proximity_l2": l2,
        "sparsity": spar,
        "diversity": div,
    }
