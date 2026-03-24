"""
Data loader for ECG5000 (UCR Time Series Classification Archive).

ECG5000 properties:
  - 5 classes: 1=Normal, 2-5=arrhythmia variants
  - 500 training samples, 4500 test samples
  - 140 timesteps per series (univariate)
  - Source: http://www.timeseriesclassification.com/description.php?Dataset=ECG5000

Download: fetched from UCR archive via sktime/aeon or directly as .ts/.arff files.
Falls back to aeonml download API if local files absent.

Returns a dict mirroring phase1/image/data_loader.py structure:
    X_train, y_train   : (500, 1, 140) float32,  (500,) int
    X_test,  y_test    : (4500, 1, 140) float32, (4500,) int
    X_train_raw        : same as X_train (no separate raw for 1D data)
    baseline_series    : (1, 140) float32 — per-channel training mean
    qrs_windows        : dict {sample_idx: (start, end)} — detected QRS windows
                         for temporal localisation metrics
"""
import pickle
import numpy as np
from pathlib import Path

from phase2.timeseries.config import (
    TS_DATA_DIR, RANDOM_STATE, SERIES_LENGTH, NUM_CLASSES
)


def load_ecg5000(force_reload: bool = False) -> dict:
    """
    Load ECG5000 dataset, with caching.

    Parameters
    ----------
    force_reload : bool
        If True, ignore cache and re-download / re-process.

    Returns
    -------
    dict with keys:
        X_train, y_train, X_test, y_test, baseline_series, qrs_windows
    """
    cache_path = TS_DATA_DIR / "ecg5000_processed.pkl"
    TS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force_reload:
        print(f"Loading ECG5000 from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("Downloading / loading ECG5000 from UCR archive...")
    X_train, y_train, X_test, y_test = _fetch_ecg5000()

    # Normalise: z-score per training set statistics
    mean = X_train.mean(axis=(0, 2), keepdims=True)   # (1, 1, 1)
    std  = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    X_train = ((X_train - mean) / std).astype(np.float32)
    X_test  = ((X_test  - mean) / std).astype(np.float32)

    # Baseline: per-channel training mean (shape (1, 140))
    baseline_series = X_train.mean(axis=0)   # (1, 140)

    # QRS window detection on raw ECG (heuristic: R-peak neighbourhood)
    print("  Detecting QRS windows for localisation metrics...")
    qrs_windows = _detect_qrs_windows(X_train)

    data = {
        "X_train":        X_train,        # (500, 1, 140)
        "y_train":        y_train,         # (500,) int in [0, NUM_CLASSES-1]
        "X_test":         X_test,          # (4500, 1, 140)
        "y_test":         y_test,          # (4500,) int
        "baseline_series": baseline_series, # (1, 140)
        "qrs_windows":    qrs_windows,     # dict {idx: (start, end)}
        "norm_mean":      mean,
        "norm_std":       std,
    }

    print(f"  Saving cache to {cache_path}")
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)

    _print_summary(data)
    return data


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _fetch_ecg5000():
    """
    Fetch ECG5000 via aeon (preferred) or sktime fallback.

    Returns X_train (N,1,T), y_train (N,), X_test (N,1,T), y_test (N,).
    Labels are 0-indexed integers.
    """
    try:
        from aeon.datasets import load_classification
        X_tr, y_tr = load_classification("ECG5000", split="train")
        X_te, y_te = load_classification("ECG5000", split="test")
    except Exception:
        try:
            from sktime.datasets import load_UCR_UEA_dataset
            X_tr_df, y_tr = load_UCR_UEA_dataset("ECG5000", split="train",
                                                   return_X_y=True)
            X_te_df, y_te = load_UCR_UEA_dataset("ECG5000", split="test",
                                                   return_X_y=True)
            X_tr = _sktime_to_numpy(X_tr_df)
            X_te = _sktime_to_numpy(X_te_df)
        except Exception as e:
            raise RuntimeError(
                "Could not load ECG5000. Install 'aeon' or 'sktime':\n"
                "  pip install aeon\n"
                f"Original error: {e}"
            )

    # Ensure (N, 1, T) shape
    if X_tr.ndim == 2:
        X_tr = X_tr[:, np.newaxis, :]
        X_te = X_te[:, np.newaxis, :]
    elif X_tr.ndim == 3 and X_tr.shape[1] != 1:
        # (N, T, C) → (N, C, T)  — aeon sometimes returns this
        X_tr = X_tr.transpose(0, 2, 1)
        X_te = X_te.transpose(0, 2, 1)

    # 0-index labels
    y_tr = _encode_labels(y_tr)
    y_te = _encode_labels(y_te)

    X_tr = X_tr.astype(np.float32)
    X_te = X_te.astype(np.float32)

    print(f"  ECG5000 loaded: train={X_tr.shape}, test={X_te.shape}")
    return X_tr, y_tr, X_te, y_te


def _sktime_to_numpy(X_df) -> np.ndarray:
    """Convert sktime nested DataFrame to (N, 1, T) numpy array."""
    n_samples = len(X_df)
    n_time    = len(X_df.iloc[0, 0])
    arr = np.zeros((n_samples, 1, n_time), dtype=np.float32)
    for i in range(n_samples):
        arr[i, 0, :] = X_df.iloc[i, 0].values
    return arr


def _encode_labels(y) -> np.ndarray:
    """Map string or 1-indexed integer labels to 0-indexed integers."""
    import pandas as pd
    y = np.array(y)
    if y.dtype.kind in ("U", "S", "O"):   # string labels
        unique = sorted(set(y.tolist()))
        mapping = {v: i for i, v in enumerate(unique)}
        return np.array([mapping[v] for v in y], dtype=np.int64)
    # Numeric: shift to 0-indexed
    y = y.astype(np.int64)
    y -= y.min()
    return y


def _detect_qrs_windows(X_train: np.ndarray) -> dict:
    """
    Detect approximate QRS complex window per training sample.

    Strategy: R-peak = argmax of absolute value across the series.
    QRS window = R-peak ± 10 timesteps (capped at series boundaries).

    In standard ECG5000 (normalised 140-step heartbeats), the R-peak
    is typically near the centre (timestep ~60-80).

    Returns
    -------
    dict {sample_idx: (start_t, end_t)} for all training samples.
    """
    QRS_HALF_WIDTH = 10
    T = X_train.shape[-1]
    windows = {}
    for i in range(len(X_train)):
        series = X_train[i, 0]          # (140,)
        r_peak = int(np.abs(series).argmax())
        start  = max(0, r_peak - QRS_HALF_WIDTH)
        end    = min(T - 1, r_peak + QRS_HALF_WIDTH)
        windows[i] = (start, end)
    return windows


def _print_summary(data: dict) -> None:
    print("\n" + "=" * 50)
    print("ECG5000 Dataset Summary")
    print("=" * 50)
    print(f"  Train: {data['X_train'].shape}  Labels: {np.unique(data['y_train'])}")
    print(f"  Test:  {data['X_test'].shape}   Labels: {np.unique(data['y_test'])}")
    print(f"  Baseline series shape: {data['baseline_series'].shape}")
    print(f"  QRS windows computed for {len(data['qrs_windows'])} train samples")
    print("=" * 50)
