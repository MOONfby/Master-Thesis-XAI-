"""
Temporal Localisation Metrics for Time Series XAI Evaluation.

Analogous to phase1/image/metrics/localization.py (Pointing Game, Seg IoU),
but adapted for 1D temporal data using ECG5000's QRS complex windows as
ground truth.

ECG domain:
  - QRS complex = the most diagnostically significant part of an ECG beat
  - A good XAI method should assign highest attribution to the QRS window
  - QRS windows are detected from the training set via R-peak detection
    in data_loader.py

Metrics:
  - Temporal Pointing Game: is the peak attribution timestep inside the QRS window?
  - Temporal IoU: overlap between thresholded attribution and QRS window mask
"""
import numpy as np


# ------------------------------------------------------------------
# Per-instance metrics
# ------------------------------------------------------------------

def temporal_pointing_game(ts_heatmap: np.ndarray,
                            qrs_window: tuple) -> bool:
    """
    Temporal Pointing Game for a single time series.

    Parameters
    ----------
    ts_heatmap : np.ndarray, shape (T,)
        Per-timestep attribution values. Obtain via explainer.explain_timestep_level().
    qrs_window : tuple (start_t, end_t)
        Ground-truth QRS window boundaries (inclusive).

    Returns
    -------
    bool : True if argmax attribution is inside the QRS window.
    """
    peak_t = int(ts_heatmap.argmax())
    start, end = qrs_window
    return start <= peak_t <= end


def temporal_iou(ts_heatmap: np.ndarray,
                 qrs_window: tuple,
                 threshold: float = None) -> float:
    """
    Temporal Intersection-over-Union between thresholded attribution and QRS window.

    Parameters
    ----------
    ts_heatmap : np.ndarray, shape (T,)
        Per-timestep attribution values, non-negative.
    qrs_window : tuple (start_t, end_t)
        Ground-truth QRS window boundaries (inclusive).
    threshold : float or None
        Binarisation threshold. If None, uses mean + std of heatmap (adaptive).

    Returns
    -------
    float in [0, 1]
    """
    T = len(ts_heatmap)
    start, end = qrs_window

    # Ground-truth mask
    gt_mask = np.zeros(T, dtype=bool)
    gt_mask[start:end + 1] = True

    # Predicted mask
    if threshold is None:
        threshold = ts_heatmap.mean() + ts_heatmap.std()
    pred_mask = ts_heatmap > threshold

    intersection = (pred_mask & gt_mask).sum()
    union        = (pred_mask | gt_mask).sum()

    if union == 0:
        return 0.0
    return float(intersection / union)


# ------------------------------------------------------------------
# Batch evaluation
# ------------------------------------------------------------------

def evaluate_localization(explainer,
                           series_batch: np.ndarray,
                           segment_maps: np.ndarray,
                           qrs_windows: dict,
                           sample_indices: list,
                           target_labels: np.ndarray = None) -> dict:
    """
    Evaluate temporal localisation metrics for a batch of series.

    Parameters
    ----------
    explainer : LIMETSExplainer, TimeSHAPExplainer, or IntegratedGradientsExplainer
        Must implement explain_timestep_level(series, segment_map, label).
    series_batch : (N, 1, T) normalised float32
    segment_maps : (N, T)
    qrs_windows : dict {sample_idx: (start_t, end_t)}
        From load_ecg5000()['qrs_windows']. Keyed by original training index.
    sample_indices : list of int, length N
        Original dataset indices for each series (to look up QRS windows).
    target_labels : (N,) int or None

    Returns
    -------
    dict:
        pointing_game_acc : float
        mean_iou          : float
        n_evaluated       : int
    """
    N = len(series_batch)
    pg_results  = []
    iou_results = []

    for i in range(N):
        idx = sample_indices[i]
        qrs = qrs_windows.get(idx)
        if qrs is None:
            continue

        lbl = int(target_labels[i]) if target_labels is not None else None

        # Per-timestep heatmap
        heatmap = explainer.explain_timestep_level(
            series_batch[i], segment_maps[i], label=lbl
        )   # (T,)

        # Ensure non-negative for IoU threshold logic
        heatmap_abs = np.abs(heatmap)

        pg_results.append(temporal_pointing_game(heatmap_abs, qrs))
        iou_results.append(temporal_iou(heatmap_abs, qrs))

        if (i + 1) % 10 == 0:
            print(f"    Temporal Localisation: {i+1}/{N} done")

    return {
        "pointing_game_acc": float(np.mean(pg_results)) if pg_results else float("nan"),
        "mean_iou":          float(np.mean(iou_results)) if iou_results else float("nan"),
        "n_evaluated":       len(pg_results),
    }
