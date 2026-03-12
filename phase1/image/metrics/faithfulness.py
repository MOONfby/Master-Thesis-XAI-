"""
Faithfulness Metrics for Image XAI Evaluation.

Adapts Phase 1 tabular faithfulness metrics (AOPC, Comprehensiveness,
Sufficiency) to image data by replacing column-level masking with
superpixel-level masking via apply_superpixel_mask().

Also implements Insertion/Deletion AUC (Petsiuk et al. RISE, BMVC 2018),
which have no direct tabular equivalent.

All functions follow the same signature pattern as phase1/tabular/metrics/faithfulness.py.
"""
import numpy as np
from sklearn.metrics import auc as sklearn_auc

from phase1.image.utils import apply_superpixel_mask


# ------------------------------------------------------------------
# AOPC (Area Over Perturbation Curve)
# Adapted from: Samek et al. 2017
# ------------------------------------------------------------------

def region_aopc_score(predict_fn,
                      images: np.ndarray,
                      segment_maps: np.ndarray,
                      attributions: np.ndarray,
                      baseline_image: np.ndarray,
                      n_steps: int = 10,
                      target_labels: np.ndarray = None):
    """
    Compute Region AOPC for a batch of images.

    Progressively removes top-k superpixels (most important first)
    and measures the drop in predicted probability for the target class.

    Parameters
    ----------
    predict_fn : callable (np.ndarray (N,3,H,W)) -> (N, num_classes)
    images : (N, 3, H, W) normalised float32
    segment_maps : (N, H, W) int
    attributions : (N, S) superpixel attribution scores
    baseline_image : (3, H, W) float32 — replacement for masked segments
    n_steps : int — number of progressive removal steps
    target_labels : (N,) int or None — class to measure; None = argmax

    Returns
    -------
    per_instance : np.ndarray (N,)
    mean_aopc    : float
    """
    N, S = attributions.shape

    if target_labels is None:
        probs0 = predict_fn(images)          # (N, C)
        target_labels = probs0.argmax(axis=1)

    # Original predictions for target class
    probs0 = predict_fn(images)
    f_orig = probs0[np.arange(N), target_labels]  # (N,)

    step_drops = []
    for k in range(1, n_steps + 1):
        n_mask = max(1, int(S * k / n_steps))
        masked_images = []
        for i in range(N):
            top_k = np.argsort(attributions[i])[::-1][:n_mask]
            masked = apply_superpixel_mask(
                images[i], segment_maps[i], top_k, baseline_image
            )
            masked_images.append(masked)
        masked_batch = np.stack(masked_images)
        probs_k = predict_fn(masked_batch)
        f_k = probs_k[np.arange(N), target_labels]
        step_drops.append(f_orig - f_k)   # (N,)

    # AOPC = mean drop across steps
    per_instance = np.mean(step_drops, axis=0)   # (N,)
    return per_instance, float(per_instance.mean())


# ------------------------------------------------------------------
# Comprehensiveness
# Adapted from: DeYoung et al. 2020
# ------------------------------------------------------------------

def region_comprehensiveness(predict_fn,
                              images: np.ndarray,
                              segment_maps: np.ndarray,
                              attributions: np.ndarray,
                              baseline_image: np.ndarray,
                              top_k: int = 10,
                              target_labels: np.ndarray = None):
    """
    Comprehensiveness: remove top-k superpixels, measure prediction drop.

    comp(k) = f(x) - f(x with top-k removed)
    Higher is better — important features being removed matters.

    Returns
    -------
    per_instance : np.ndarray (N,)
    mean_comp    : float
    """
    N, S = attributions.shape
    k = min(top_k, S)

    probs0 = predict_fn(images)
    if target_labels is None:
        target_labels = probs0.argmax(axis=1)
    f_orig = probs0[np.arange(N), target_labels]

    masked_images = []
    for i in range(N):
        top_k_idx = np.argsort(attributions[i])[::-1][:k]
        masked = apply_superpixel_mask(
            images[i], segment_maps[i], top_k_idx, baseline_image
        )
        masked_images.append(masked)

    probs_masked = predict_fn(np.stack(masked_images))
    f_masked = probs_masked[np.arange(N), target_labels]

    per_instance = f_orig - f_masked
    return per_instance, float(per_instance.mean())


# ------------------------------------------------------------------
# Sufficiency
# Adapted from: DeYoung et al. 2020
# ------------------------------------------------------------------

def region_sufficiency(predict_fn,
                       images: np.ndarray,
                       segment_maps: np.ndarray,
                       attributions: np.ndarray,
                       baseline_image: np.ndarray,
                       top_k: int = 10,
                       target_labels: np.ndarray = None):
    """
    Sufficiency: keep only top-k superpixels, measure retained prediction.

    suff(k) = f(x) - f(x with ONLY top-k kept)
    Lower is better — top-k features alone should suffice.

    Returns
    -------
    per_instance : np.ndarray (N,)
    mean_suff    : float
    """
    N, S = attributions.shape
    k = min(top_k, S)

    probs0 = predict_fn(images)
    if target_labels is None:
        target_labels = probs0.argmax(axis=1)
    f_orig = probs0[np.arange(N), target_labels]

    masked_images = []
    for i in range(N):
        top_k_idx = set(np.argsort(attributions[i])[::-1][:k].tolist())
        # Mask all segments EXCEPT top-k
        non_top_k = [s for s in range(S) if s not in top_k_idx]
        masked = apply_superpixel_mask(
            images[i], segment_maps[i], non_top_k, baseline_image
        )
        masked_images.append(masked)

    probs_masked = predict_fn(np.stack(masked_images))
    f_masked = probs_masked[np.arange(N), target_labels]

    per_instance = f_orig - f_masked
    return per_instance, float(per_instance.mean())


# ------------------------------------------------------------------
# Insertion / Deletion AUC
# Petsiuk et al. RISE, BMVC 2018
# ------------------------------------------------------------------

def insertion_deletion_auc(predict_fn,
                            image: np.ndarray,
                            segment_map: np.ndarray,
                            attributions: np.ndarray,
                            baseline_image: np.ndarray,
                            target_label: int = None):
    """
    Compute Insertion AUC and Deletion AUC for a single image.

    Insertion: start from baseline, progressively add top-k segments.
    Deletion:  start from full image, progressively remove top-k segments.

    AUC computed with trapezoidal rule over fraction-of-segments axis.
    Higher Insertion AUC = better; Lower Deletion AUC = better.

    Parameters
    ----------
    image : (3, H, W)
    segment_map : (H, W)
    attributions : (S,)
    baseline_image : (3, H, W)
    target_label : int or None

    Returns
    -------
    ins_auc : float
    del_auc : float
    """
    S = len(attributions)
    rank = np.argsort(attributions)[::-1]   # most important first

    if target_label is None:
        probs = predict_fn(image[None])
        target_label = int(probs[0].argmax())

    ins_scores = []
    del_scores = []
    fractions  = []

    for k in range(0, S + 1):
        frac = k / S
        fractions.append(frac)

        top_k    = rank[:k].tolist()
        non_top_k = rank[k:].tolist()

        # Insertion: only top-k segments visible (rest = baseline)
        ins_img = apply_superpixel_mask(
            baseline_image.copy(), segment_map, non_top_k, baseline_image
        )
        # Actually build insertion image: start from baseline, add top-k from original
        ins_img = baseline_image.copy()
        for s in top_k:
            mask = segment_map == s
            ins_img[:, mask] = image[:, mask]

        # Deletion: full image with top-k removed
        del_img = apply_superpixel_mask(image, segment_map, top_k, baseline_image)

        ins_prob = predict_fn(ins_img[None])[0, target_label]
        del_prob = predict_fn(del_img[None])[0, target_label]

        ins_scores.append(float(ins_prob))
        del_scores.append(float(del_prob))

    fractions  = np.array(fractions)
    ins_scores = np.array(ins_scores)
    del_scores = np.array(del_scores)

    ins_auc = float(np.trapz(ins_scores, fractions))
    del_auc = float(np.trapz(del_scores, fractions))

    return ins_auc, del_auc


def insertion_deletion_auc_batch(predict_fn,
                                  images: np.ndarray,
                                  segment_maps: np.ndarray,
                                  attributions: np.ndarray,
                                  baseline_image: np.ndarray,
                                  target_labels: np.ndarray = None):
    """
    Batch version of insertion_deletion_auc.

    Returns
    -------
    mean_ins_auc : float
    mean_del_auc : float
    """
    N = len(images)
    if target_labels is None:
        probs = predict_fn(images)
        target_labels = probs.argmax(axis=1)

    ins_aucs, del_aucs = [], []
    for i in range(N):
        ins, del_ = insertion_deletion_auc(
            predict_fn, images[i], segment_maps[i],
            attributions[i], baseline_image, target_labels[i]
        )
        ins_aucs.append(ins)
        del_aucs.append(del_)
        if (i + 1) % 10 == 0:
            print(f"    Ins/Del AUC: {i+1}/{N} done")

    return float(np.mean(ins_aucs)), float(np.mean(del_aucs))
