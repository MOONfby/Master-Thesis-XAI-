"""
Faithfulness Metrics for Time Series XAI Evaluation.

Direct port of phase1/image/metrics/faithfulness.py with:
  - apply_superpixel_mask() → apply_temporal_mask()
  - Images (N, 3, H, W) → Series (N, 1, T)
  - segment_maps (N, H, W) → segment_maps (N, T)

All five metrics are implemented identically to the image version.
"""
import numpy as np

from phase2.timeseries.utils import apply_temporal_mask


def region_aopc_score(predict_fn,
                      series_batch: np.ndarray,
                      segment_maps: np.ndarray,
                      attributions: np.ndarray,
                      baseline_series: np.ndarray,
                      n_steps: int = 10,
                      target_labels: np.ndarray = None):
    """
    Region AOPC for a batch of time series.

    Progressively removes top-k temporal segments and measures the drop
    in predicted probability for the target class.

    Parameters
    ----------
    predict_fn    : callable (N, 1, T) -> (N, C)
    series_batch  : (N, 1, T)
    segment_maps  : (N, T) int
    attributions  : (N, S) temporal segment attribution scores
    baseline_series : (1, T) float32
    n_steps       : int
    target_labels : (N,) int or None

    Returns
    -------
    per_instance : np.ndarray (N,)
    mean_aopc    : float
    """
    N, S = attributions.shape

    if target_labels is None:
        probs0 = predict_fn(series_batch)
        target_labels = probs0.argmax(axis=1)

    probs0 = predict_fn(series_batch)
    f_orig = probs0[np.arange(N), target_labels]   # (N,)

    step_drops = []
    for k in range(1, n_steps + 1):
        n_mask = max(1, int(S * k / n_steps))
        masked_series = []
        for i in range(N):
            top_k = np.argsort(attributions[i])[::-1][:n_mask]
            masked = apply_temporal_mask(
                series_batch[i], segment_maps[i], top_k, baseline_series
            )
            masked_series.append(masked)
        masked_batch = np.stack(masked_series)
        probs_k = predict_fn(masked_batch)
        f_k = probs_k[np.arange(N), target_labels]
        step_drops.append(f_orig - f_k)

    per_instance = np.mean(step_drops, axis=0)   # (N,)
    return per_instance, float(per_instance.mean())


def region_comprehensiveness(predict_fn,
                              series_batch: np.ndarray,
                              segment_maps: np.ndarray,
                              attributions: np.ndarray,
                              baseline_series: np.ndarray,
                              top_k: int = 4,
                              target_labels: np.ndarray = None):
    """
    Comprehensiveness: remove top-k segments, measure prediction drop.

    comp(k) = f(x) - f(x with top-k removed)
    Higher is better.

    Returns
    -------
    per_instance : np.ndarray (N,)
    mean_comp    : float
    """
    N, S = attributions.shape
    k = min(top_k, S)

    probs0 = predict_fn(series_batch)
    if target_labels is None:
        target_labels = probs0.argmax(axis=1)
    f_orig = probs0[np.arange(N), target_labels]

    masked_series = []
    for i in range(N):
        top_k_idx = np.argsort(attributions[i])[::-1][:k]
        masked = apply_temporal_mask(
            series_batch[i], segment_maps[i], top_k_idx, baseline_series
        )
        masked_series.append(masked)

    probs_masked = predict_fn(np.stack(masked_series))
    f_masked = probs_masked[np.arange(N), target_labels]

    per_instance = f_orig - f_masked
    return per_instance, float(per_instance.mean())


def region_sufficiency(predict_fn,
                       series_batch: np.ndarray,
                       segment_maps: np.ndarray,
                       attributions: np.ndarray,
                       baseline_series: np.ndarray,
                       top_k: int = 4,
                       target_labels: np.ndarray = None):
    """
    Sufficiency: keep only top-k segments, measure retained prediction.

    suff(k) = f(x) - f(x with ONLY top-k kept)
    Lower is better.

    Returns
    -------
    per_instance : np.ndarray (N,)
    mean_suff    : float
    """
    N, S = attributions.shape
    k = min(top_k, S)

    probs0 = predict_fn(series_batch)
    if target_labels is None:
        target_labels = probs0.argmax(axis=1)
    f_orig = probs0[np.arange(N), target_labels]

    masked_series = []
    for i in range(N):
        top_k_set = set(np.argsort(attributions[i])[::-1][:k].tolist())
        non_top_k = [s for s in range(S) if s not in top_k_set]
        masked = apply_temporal_mask(
            series_batch[i], segment_maps[i], non_top_k, baseline_series
        )
        masked_series.append(masked)

    probs_masked = predict_fn(np.stack(masked_series))
    f_masked = probs_masked[np.arange(N), target_labels]

    per_instance = f_orig - f_masked
    return per_instance, float(per_instance.mean())


def insertion_deletion_auc(predict_fn,
                            series: np.ndarray,
                            segment_map: np.ndarray,
                            attributions: np.ndarray,
                            baseline_series: np.ndarray,
                            target_label: int = None):
    """
    Insertion AUC and Deletion AUC for a single time series.

    Insertion: start from baseline, progressively add top-k segments.
    Deletion:  start from full series, progressively remove top-k segments.

    Returns
    -------
    ins_auc : float — Higher is better
    del_auc : float — Lower is better
    """
    S = len(attributions)
    rank = np.argsort(attributions)[::-1]   # most important first

    if target_label is None:
        probs = predict_fn(series[np.newaxis])
        target_label = int(probs[0].argmax())

    ins_scores = []
    del_scores = []
    fractions  = []

    for k in range(0, S + 1):
        fractions.append(k / S)
        top_k     = rank[:k].tolist()
        non_top_k = rank[k:].tolist()

        # Insertion: start from baseline, add top-k from original
        ins_s = baseline_series.copy()
        for seg in top_k:
            mask = segment_map == seg
            ins_s[:, mask] = series[:, mask]

        # Deletion: full series with top-k replaced by baseline
        del_s = apply_temporal_mask(series, segment_map, top_k, baseline_series)

        ins_p = predict_fn(ins_s[np.newaxis])[0, target_label]
        del_p = predict_fn(del_s[np.newaxis])[0, target_label]

        ins_scores.append(float(ins_p))
        del_scores.append(float(del_p))

    fractions  = np.array(fractions)
    ins_auc = float(np.trapz(ins_scores, fractions))
    del_auc = float(np.trapz(del_scores, fractions))
    return ins_auc, del_auc


def insertion_deletion_auc_batch(predict_fn,
                                  series_batch: np.ndarray,
                                  segment_maps: np.ndarray,
                                  attributions: np.ndarray,
                                  baseline_series: np.ndarray,
                                  target_labels: np.ndarray = None):
    """
    Batch version of insertion_deletion_auc.

    Returns
    -------
    mean_ins_auc : float
    mean_del_auc : float
    """
    N = len(series_batch)
    if target_labels is None:
        probs = predict_fn(series_batch)
        target_labels = probs.argmax(axis=1)

    ins_aucs, del_aucs = [], []
    for i in range(N):
        ins, del_ = insertion_deletion_auc(
            predict_fn, series_batch[i], segment_maps[i],
            attributions[i], baseline_series, target_labels[i]
        )
        ins_aucs.append(ins)
        del_aucs.append(del_)
        if (i + 1) % 10 == 0:
            print(f"    Ins/Del AUC: {i+1}/{N} done")

    return float(np.mean(ins_aucs)), float(np.mean(del_aucs))
