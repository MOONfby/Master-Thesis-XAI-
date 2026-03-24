"""
Shared utilities for time series XAI evaluation.

Temporal segment abstraction: treats fixed-length time windows as the feature
unit, mirroring Phase 1's superpixel abstraction for images and per-feature
attributions for tabular data.

Key design: segment_map is a (T,) array mapping each timestep to a segment
index in [0, N_SEGMENTS_TS-1]. For ECG5000 with T=140, N=20 segments of 7
timesteps each, segment_map = [0,0,0,0,0,0,0, 1,1,1,1,1,1,1, ..., 19,...].
"""
import numpy as np

from phase2.timeseries.config import N_SEGMENTS_TS, SEGMENT_LENGTH, SERIES_LENGTH


def build_segment_map(series_length: int = SERIES_LENGTH,
                      n_segments: int = N_SEGMENTS_TS) -> np.ndarray:
    """
    Build a uniform temporal segment map.

    Parameters
    ----------
    series_length : int — number of timesteps
    n_segments : int   — number of equal-length segments

    Returns
    -------
    np.ndarray, shape (series_length,), dtype int
        segment_map[t] = segment index containing timestep t.
        Last segment absorbs any remainder timesteps.
    """
    seg_len = series_length // n_segments
    segment_map = np.zeros(series_length, dtype=np.int32)
    for s in range(n_segments):
        start = s * seg_len
        end   = (s + 1) * seg_len if s < n_segments - 1 else series_length
        segment_map[start:end] = s
    return segment_map


def aggregate_to_segments(timestep_attributions: np.ndarray,
                           segment_map: np.ndarray,
                           n_segments: int = N_SEGMENTS_TS) -> np.ndarray:
    """
    Aggregate per-timestep attributions to segment-level.

    Mirrors phase1/image/utils.py:aggregate_to_superpixels().

    Parameters
    ----------
    timestep_attributions : np.ndarray, shape (T,)
        Per-timestep attribution values (absolute or signed).
    segment_map : np.ndarray, shape (T,), dtype int
    n_segments : int — output size (fixed for consistency across samples)

    Returns
    -------
    np.ndarray, shape (n_segments,)
        Mean attribution per temporal segment.
    """
    S = n_segments
    attrs = np.zeros(S, dtype=np.float64)
    for s in range(S):
        mask = segment_map == s
        if mask.any():
            attrs[s] = timestep_attributions[mask].mean()
    return attrs


def apply_temporal_mask(series: np.ndarray,
                         segment_map: np.ndarray,
                         mask_indices,
                         baseline_series: np.ndarray) -> np.ndarray:
    """
    Replace selected temporal segments with baseline values.

    Analogous to phase1/image/utils.py:apply_superpixel_mask().
    Tabular analogue: x_masked[top_indices] = baseline_values[top_indices].

    Parameters
    ----------
    series : np.ndarray, shape (1, T) or (C, T)
        Input time series (float, normalised).
    segment_map : np.ndarray, shape (T,), dtype int
        Temporal segment label per timestep.
    mask_indices : array-like of int
        Segment indices to replace with baseline.
    baseline_series : np.ndarray, same shape as series
        Replacement values (per-channel training mean series).

    Returns
    -------
    np.ndarray, same shape as series
        Series with specified segments replaced by baseline.
    """
    masked = series.copy()
    for s in mask_indices:
        seg_mask = segment_map == s   # (T,) bool
        masked[:, seg_mask] = baseline_series[:, seg_mask]
    return masked


def perturb_series(series: np.ndarray, noise_std: float,
                   rng: np.random.Generator) -> np.ndarray:
    """
    Add Gaussian noise for stability evaluation.

    Analogous to phase1/image/utils.py:perturb_image().
    Applied to all timesteps (no categorical exclusion needed for ECG).

    Parameters
    ----------
    series : np.ndarray, shape (1, T) or (C, T), normalised float32
    noise_std : float — Gaussian σ in normalised space
    rng : np.random.Generator

    Returns
    -------
    np.ndarray, same shape — noise added, no clipping (normalised ECG
    can exceed [0,1] range; clipping would distort the waveform)
    """
    noise = rng.normal(0, noise_std, series.shape).astype(np.float32)
    return series + noise


def compute_baseline_series(X_train: np.ndarray) -> np.ndarray:
    """
    Compute per-channel mean series from training data.

    Analogous to phase1/image/utils.py:compute_baseline_image().

    Parameters
    ----------
    X_train : np.ndarray, shape (N, C, T)

    Returns
    -------
    np.ndarray, shape (C, T)
    """
    return X_train.mean(axis=0)
