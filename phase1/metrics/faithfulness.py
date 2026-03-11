"""
Faithfulness / Fidelity Metrics for XAI evaluation.

Faithfulness measures whether the explanation accurately reflects the model's
actual decision-making process — i.e., if the explanation says feature X is
important, removing X should significantly change the prediction.

Implemented metrics:

1. AOPC (Area Over the Perturbation Curve) — Samek et al., 2017
   - Sort features by |attribution| (most important first)
   - Progressively replace top-k features with baseline values
   - AOPC = (1/K) * Σ_{k=1}^{K} [f(x) - f(x_k)]
   - Higher AOPC → explanation correctly identifies impactful features

2. Comprehensiveness — DeYoung et al., 2020
   - comp(k) = f(x) - f(x with top-k features replaced by baseline)
   - Measures: does removing top-k features substantially reduce prediction?
   - Higher → explanation identifies the truly important features

3. Sufficiency — DeYoung et al., 2020
   - suff(k) = f(x) - f(x with ONLY top-k features kept, rest = baseline)
   - Measures: do top-k features alone suffice to maintain the prediction?
   - Lower → top-k features alone are sufficient (explanation is concise)

4. LIME Local Fidelity (R²) — Ribeiro et al., 2016
   - R² of LIME's linear surrogate on its neighborhood
   - LIME-specific; higher → surrogate approximates model well locally

All perturbation metrics operate on the model's input space (encoded features).
Baseline = training set mean (standard choice for tabular data).

References:
  Samek et al. (2017) "Evaluating the Visualization of What a Deep Neural Network
  has Learned." IEEE TNNLS.
  DeYoung et al. (2020) "ERASER: A Benchmark to Evaluate Rationalized NLP Models."
  ACL 2020.
"""
import numpy as np
from tqdm import tqdm

from phase1.config import FAITHFULNESS_N_STEPS, FAITHFULNESS_TOP_K


def aopc_score(
    predict_fn,
    X: np.ndarray,
    attributions: np.ndarray,
    baseline: np.ndarray,
    n_steps: int = FAITHFULNESS_N_STEPS,
) -> tuple[np.ndarray, float]:
    """
    Compute AOPC (Area Over the Perturbation Curve).

    Parameters
    ----------
    predict_fn    : callable, returns probabilities array of shape (n, 2)
    X             : test instances, shape (n_samples, n_features)
    attributions  : feature attributions, shape (n_samples, n_features)
    baseline      : baseline values for masking, shape (n_features,)
    n_steps       : number of features to progressively remove

    Returns
    -------
    aopc_per_instance : np.ndarray, shape (n_samples,)
    mean_aopc         : float, mean over instances
    """
    n_samples, n_features = X.shape
    n_steps = min(n_steps, n_features)
    aopc_per_instance = np.zeros(n_samples)

    for i in tqdm(range(n_samples), desc="AOPC", leave=False):
        # Sort features by absolute importance (descending)
        sorted_idx = np.argsort(np.abs(attributions[i]))[::-1]

        orig_prob = predict_fn(X[[i]])[0, 1]
        x_perturbed = X[i].copy()
        cumulative = 0.0

        for k in range(n_steps):
            feat = sorted_idx[k]
            x_perturbed[feat] = baseline[feat]
            perturbed_prob = predict_fn(x_perturbed[np.newaxis, :])[0, 1]
            cumulative += (orig_prob - perturbed_prob)

        aopc_per_instance[i] = cumulative / n_steps

    return aopc_per_instance, float(np.mean(aopc_per_instance))


def comprehensiveness(
    predict_fn,
    X: np.ndarray,
    attributions: np.ndarray,
    baseline: np.ndarray,
    top_k: int = FAITHFULNESS_TOP_K,
) -> tuple[np.ndarray, float]:
    """
    Compute Comprehensiveness score.

    comp = f(x) - f(x with top-k features replaced by baseline)

    Higher is better: removing important features should reduce the prediction.

    Returns
    -------
    comp_per_instance : np.ndarray, shape (n_samples,)
    mean_comp         : float
    """
    n_samples = len(X)
    comp_per_instance = np.zeros(n_samples)

    for i in range(n_samples):
        sorted_idx = np.argsort(np.abs(attributions[i]))[::-1][:top_k]

        orig_prob = predict_fn(X[[i]])[0, 1]

        x_masked = X[i].copy()
        x_masked[sorted_idx] = baseline[sorted_idx]
        masked_prob = predict_fn(x_masked[np.newaxis, :])[0, 1]

        comp_per_instance[i] = orig_prob - masked_prob

    return comp_per_instance, float(np.mean(comp_per_instance))


def sufficiency(
    predict_fn,
    X: np.ndarray,
    attributions: np.ndarray,
    baseline: np.ndarray,
    top_k: int = FAITHFULNESS_TOP_K,
) -> tuple[np.ndarray, float]:
    """
    Compute Sufficiency score.

    suff = f(x) - f(x with ONLY top-k features; rest = baseline)

    Lower is better: top-k features alone should maintain the original prediction.
    A low sufficiency score means the top-k features are sufficient to reconstruct
    the model's decision.

    Returns
    -------
    suff_per_instance : np.ndarray, shape (n_samples,)
    mean_suff         : float
    """
    n_samples, n_features = X.shape
    suff_per_instance = np.zeros(n_samples)

    for i in range(n_samples):
        sorted_idx = np.argsort(np.abs(attributions[i]))[::-1][:top_k]

        orig_prob = predict_fn(X[[i]])[0, 1]

        # Keep only top-k features; replace the rest with baseline
        x_sufficient = np.full(n_features, fill_value=np.nan)
        x_sufficient[:] = baseline[:]
        x_sufficient[sorted_idx] = X[i][sorted_idx]
        suff_prob = predict_fn(x_sufficient[np.newaxis, :])[0, 1]

        suff_per_instance[i] = orig_prob - suff_prob

    return suff_per_instance, float(np.mean(suff_per_instance))


def lime_local_fidelity_batch(
    lime_explainer,
    instances: np.ndarray,
    n_samples: int = 1000,
    n_features: int = 10,
) -> tuple[np.ndarray, float]:
    """
    Compute LIME local fidelity (R²) for multiple instances.

    This measures how well the linear surrogate approximates the black-box
    model in the local neighborhood — an intrinsic measure of LIME's faithfulness.

    Returns
    -------
    r2_scores : np.ndarray, shape (n_instances,)
    mean_r2   : float
    """
    r2_scores = np.zeros(len(instances))
    for i, inst in enumerate(tqdm(instances, desc="LIME Local Fidelity", leave=False)):
        r2_scores[i] = lime_explainer.get_local_fidelity(inst, n_samples, n_features)
    return r2_scores, float(np.mean(r2_scores))
