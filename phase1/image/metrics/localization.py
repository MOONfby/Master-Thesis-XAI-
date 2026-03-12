"""
Localisation Metrics for Image XAI Evaluation.

These metrics are unique to image data — they use CUB-200-2011 ground-truth
annotations to evaluate whether explanations correctly identify object locations.
No tabular equivalent exists.

Methods:
  - Pointing Game (Zhang et al. 2016): is the highest-attribution pixel
    inside the ground-truth bounding box?
  - Segmentation IoU: overlap between thresholded saliency map and GT mask.
"""
import numpy as np
from pathlib import Path
from PIL import Image


# ------------------------------------------------------------------
# Per-instance metrics
# ------------------------------------------------------------------

def pointing_game(pixel_heatmap: np.ndarray, bbox: tuple) -> bool:
    """
    Pointing Game accuracy for a single image.

    Parameters
    ----------
    pixel_heatmap : np.ndarray, shape (H, W)
        Pixel-level attribution values. Obtain via explainer.explain_pixel_level().
    bbox : tuple (x, y, w, h)
        Ground-truth bounding box in resized image coordinates
        (scaled to IMAGE_SIZE × IMAGE_SIZE).
        x, y = top-left corner; w, h = width, height.

    Returns
    -------
    bool : True if argmax is inside the bounding box.
    """
    row, col = np.unravel_index(pixel_heatmap.argmax(), pixel_heatmap.shape)
    x, y, w, h = bbox
    return (x <= col <= x + w) and (y <= row <= y + h)


def segmentation_iou(pixel_heatmap: np.ndarray,
                     gt_mask: np.ndarray,
                     threshold: float = None) -> float:
    """
    Intersection-over-Union between thresholded saliency and GT mask.

    Parameters
    ----------
    pixel_heatmap : np.ndarray, shape (H, W), non-negative
    gt_mask : np.ndarray, shape (H, W), bool or {0,1}
        Ground-truth binary segmentation mask.
    threshold : float or None
        Binarisation threshold for pixel_heatmap.
        If None, uses mean + std of the heatmap (adaptive).

    Returns
    -------
    float in [0, 1]
    """
    gt_bool = gt_mask.astype(bool)

    if threshold is None:
        threshold = pixel_heatmap.mean() + pixel_heatmap.std()
    pred_bool = pixel_heatmap > threshold

    intersection = (pred_bool & gt_bool).sum()
    union        = (pred_bool | gt_bool).sum()

    if union == 0:
        return 0.0
    return float(intersection / union)


# ------------------------------------------------------------------
# Batch evaluation
# ------------------------------------------------------------------

def evaluate_localization(explainer,
                           images: np.ndarray,
                           segment_maps: np.ndarray,
                           annotations: dict,
                           image_ids: list,
                           target_labels: np.ndarray = None) -> dict:
    """
    Evaluate localisation metrics for a batch of images.

    Parameters
    ----------
    explainer : LIMEImageExplainer or GradientSHAPExplainer
        Must implement explain_pixel_level(image, segment_map, label).
    images : (N, 3, H, W) normalised float32
    segment_maps : (N, H, W)
    annotations : dict
        Output of load_cub200()['annotations']. Keyed by image_id.
    image_ids : list of int, length N
        CUB-200 image ids aligned with images array.
    target_labels : (N,) int or None

    Returns
    -------
    dict:
        pointing_game_acc : float — fraction of images correctly pointed
        mean_iou          : float — mean IoU (only for images with seg masks)
        n_pointing        : int   — images evaluated for Pointing Game
        n_iou             : int   — images evaluated for IoU
    """
    N = len(images)
    pg_results  = []
    iou_results = []

    for i in range(N):
        img_id = image_ids[i]
        ann    = annotations.get(img_id, {})
        lbl    = int(target_labels[i]) if target_labels is not None else None

        # Pixel-level heatmap
        heatmap = explainer.explain_pixel_level(
            images[i], segment_maps[i], label=lbl
        )   # (H, W)

        # Pointing Game
        bbox = ann.get("bbox_scaled") or ann.get("bbox")
        if bbox is not None:
            pg_results.append(pointing_game(heatmap, bbox))

        # Segmentation IoU
        seg_path = ann.get("seg_path")
        if seg_path is not None and Path(seg_path).exists():
            gt_mask = _load_seg_mask(seg_path, heatmap.shape)
            iou_results.append(segmentation_iou(heatmap, gt_mask))

        if (i + 1) % 10 == 0:
            print(f"    Localisation: {i+1}/{N} done")

    result = {
        "pointing_game_acc": float(np.mean(pg_results)) if pg_results else float("nan"),
        "mean_iou":          float(np.mean(iou_results)) if iou_results else float("nan"),
        "n_pointing":        len(pg_results),
        "n_iou":             len(iou_results),
    }
    return result


def _load_seg_mask(seg_path, target_shape: tuple) -> np.ndarray:
    """Load and resize a CUB-200 segmentation PNG to target_shape (H, W)."""
    mask = Image.open(seg_path).convert("L")
    H, W = target_shape
    mask = mask.resize((W, H), Image.NEAREST)
    return (np.array(mask) > 128).astype(bool)
