"""
Stability / Robustness Metrics for XAI evaluation.

Stability measures whether an explanation method produces consistent attributions
for similar inputs. An unstable explainer gives different feature rankings for
nearly identical instances, which undermines trust and reproducibility.

Formally: good explanations should satisfy Lipschitz continuity —
  ||E(x) - E(x')|| / ||x - x'|| ≤ L for some constant L

Implemented metrics:

1. Rank Correlation Stability (Spearman's ρ)
   - For each test instance x, generate N perturbed versions x' = x + ε
     where ε ~ N(0, σ²·diag(feature_std²)) (applied to numerical features only)
   - Compute feature attribution rankings for x and each x'
   - Stability = mean Spearman's ρ between rank(E(x)) and rank(E(x'))
   - Range [-1, 1]; higher → more stable

2. Average Sensitivity (Yeh et al., 2019)
   - AS = mean_over_perturbations [ ||E(x) - E(x')||₂ / ||x - x'||₂ ]
   - Measures the Lipschitz constant empirically
   - Lower → more stable (less sensitive to input perturbations)

Both metrics perturb only numerical features (gaussian noise); categorical
features are held fixed because discrete perturbations require different handling.

Reference:
  Yeh et al. (2019) "On the (In)Fidelity and Sensitivity of Explanations."
  NeurIPS 2019.
"""
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

from phase1.config import (
    STABILITY_N_PERTURBATIONS, STABILITY_NOISE_STD, RANDOM_STATE,
    NUMERICAL_FEATURES
)


def _perturb_instance(
    instance: np.ndarray,
    num_indices: list[int],
    feature_std: np.ndarray,
    noise_std: float,
    n_perturbations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate perturbed versions of an instance.

    Only numerical features are perturbed (Gaussian noise scaled by feature std).
    Categorical features remain fixed to avoid invalid category values.

    Returns
    -------
    perturbed : np.ndarray, shape (n_perturbations, n_features)
    """
    perturbed = np.tile(instance, (n_perturbations, 1))
    noise = rng.normal(
        loc=0.0,
        scale=noise_std * feature_std[num_indices],
        size=(n_perturbations, len(num_indices))
    )
    perturbed[:, num_indices] += noise
    return perturbed


def rank_correlation_stability(
    explainer,
    instances: np.ndarray,
    num_indices: list[int],
    feature_std: np.ndarray,
    n_perturbations: int = STABILITY_N_PERTURBATIONS,
    noise_std: float = STABILITY_NOISE_STD,
) -> tuple[np.ndarray, float]:
    """
    Compute Spearman rank correlation stability.

    For each instance:
      1. Compute attribution E(x)
      2. Generate N perturbed x' and compute E(x') for each
      3. Measure Spearman ρ between rank(E(x)) and rank(E(x'))
      4. Instance stability = mean ρ over N perturbations

    Parameters
    ----------
    explainer       : object with explain_batch(instances) method
    instances       : np.ndarray, shape (n_samples, n_features)
    num_indices     : indices of numerical features
    feature_std     : per-feature standard deviation, shape (n_features,)
    n_perturbations : number of perturbations per instance
    noise_std       : noise level (relative to feature std)

    Returns
    -------
    stability_scores : np.ndarray, shape (n_samples,)  — per-instance mean ρ
    mean_stability   : float
    """
    rng = np.random.default_rng(RANDOM_STATE)
    stability_scores = np.zeros(len(instances))

    for i, inst in enumerate(tqdm(instances, desc="Rank Correlation Stability", leave=False)):
        # Attribution for original instance
        orig_attr = explainer.explain(inst)

        # Perturbations
        perturbed = _perturb_instance(inst, num_indices, feature_std, noise_std,
                                      n_perturbations, rng)
        perturbed_attrs = explainer.explain_batch(perturbed)

        # Spearman rank correlation between original and each perturbation
        rho_values = []
        for j in range(n_perturbations):
            rho, _ = spearmanr(orig_attr, perturbed_attrs[j])
            if not np.isnan(rho):
                rho_values.append(rho)

        stability_scores[i] = np.mean(rho_values) if rho_values else 0.0

    return stability_scores, float(np.mean(stability_scores))


def average_sensitivity(
    explainer,
    instances: np.ndarray,
    num_indices: list[int],
    feature_std: np.ndarray,
    n_perturbations: int = STABILITY_N_PERTURBATIONS,
    noise_std: float = STABILITY_NOISE_STD,
) -> tuple[np.ndarray, float]:
    """
    Compute Average Sensitivity (empirical Lipschitz constant).

    AS_i = mean_j [ ||E(x_i) - E(x_i')_j||₂ / (||x_i - x_i'_j||₂ + ε) ]

    Lower values indicate a more stable explanation method.

    Returns
    -------
    sensitivity_scores : np.ndarray, shape (n_samples,)
    mean_sensitivity   : float
    """
    rng = np.random.default_rng(RANDOM_STATE + 1)
    sensitivity_scores = np.zeros(len(instances))

    for i, inst in enumerate(tqdm(instances, desc="Average Sensitivity", leave=False)):
        orig_attr = explainer.explain(inst)

        perturbed = _perturb_instance(inst, num_indices, feature_std, noise_std,
                                      n_perturbations, rng)
        perturbed_attrs = explainer.explain_batch(perturbed)

        sensitivities = []
        for j in range(n_perturbations):
            delta_attr = np.linalg.norm(orig_attr - perturbed_attrs[j])
            delta_input = np.linalg.norm(inst - perturbed[j]) + 1e-10
            sensitivities.append(delta_attr / delta_input)

        sensitivity_scores[i] = np.mean(sensitivities)

    return sensitivity_scores, float(np.mean(sensitivity_scores))
