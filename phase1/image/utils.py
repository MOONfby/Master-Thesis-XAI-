"""
Shared utilities for image XAI evaluation.

Superpixel abstraction: treats image segments as the feature unit,
mirroring Phase 1 tabular's (n_features,) attribution vector.
"""
import numpy as np


def aggregate_to_superpixels(pixel_heatmap: np.ndarray,
                              segment_map: np.ndarray) -> np.ndarray:
    """
    Aggregate a pixel-level attribution heatmap to superpixel-level attributions.

    Parameters
    ----------
    pixel_heatmap : np.ndarray, shape (H, W)
        Per-pixel attribution values (summed/averaged across channels if needed).
    segment_map : np.ndarray, shape (H, W), dtype int
        Integer superpixel label per pixel (from SLIC). Labels in [0, S-1].

    Returns
    -------
    np.ndarray, shape (S,)
        Mean attribution per superpixel segment.
    """
    S = int(segment_map.max()) + 1
    attrs = np.zeros(S, dtype=np.float64)
    for s in range(S):
        mask = segment_map == s
        if mask.any():
            attrs[s] = pixel_heatmap[mask].mean()
    return attrs


def apply_superpixel_mask(image: np.ndarray,
                           segment_map: np.ndarray,
                           mask_indices,
                           baseline_image: np.ndarray) -> np.ndarray:
    """
    Replace selected superpixels in an image with baseline values.

    Analogous to the tabular operation:
        x_masked[top_indices] = baseline_values[top_indices]

    Parameters
    ----------
    image : np.ndarray, shape (3, H, W) or (H, W, 3)
        Input image (float, any range).
    segment_map : np.ndarray, shape (H, W)
        Superpixel label map.
    mask_indices : array-like of int
        Segment indices to replace with baseline.
    baseline_image : np.ndarray, same shape as image
        Replacement values (e.g. per-channel training mean).

    Returns
    -------
    np.ndarray, same shape as image
        Image with specified segments replaced by baseline.
    """
    masked = image.copy()
    channel_first = (image.ndim == 3 and image.shape[0] == 3)

    for s in mask_indices:
        seg_mask = segment_map == s   # (H, W) bool
        if channel_first:
            masked[:, seg_mask] = baseline_image[:, seg_mask]
        else:
            masked[seg_mask] = baseline_image[seg_mask]
    return masked


def compute_baseline_image(images: np.ndarray) -> np.ndarray:
    """
    Compute per-channel mean image from a set of training images.

    Analogous to tabular baseline_values = training set column means.

    Parameters
    ----------
    images : np.ndarray, shape (N, 3, H, W), float in [0, 1]

    Returns
    -------
    np.ndarray, shape (3, H, W)
        Per-channel mean image.
    """
    return images.mean(axis=0)


def perturb_image(image: np.ndarray, noise_std: float,
                  rng: np.random.Generator) -> np.ndarray:
    """
    Add Gaussian pixel noise for stability evaluation.

    Analogous to tabular perturbation (noise on numerical features),
    but applied to ALL pixels (images have no categorical features).

    Parameters
    ----------
    image : np.ndarray, shape (3, H, W), float in [0, 1]
    noise_std : float
        Standard deviation of Gaussian noise (in normalised pixel space).
    rng : np.random.Generator

    Returns
    -------
    np.ndarray, shape (3, H, W), clipped to [0, 1]
    """
    noise = rng.normal(0, noise_std, image.shape).astype(np.float32)
    return np.clip(image + noise, 0.0, 1.0)
