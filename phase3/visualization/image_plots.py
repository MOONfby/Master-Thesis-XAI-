"""
Image visualization — raw image + heatmap overlay.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ImageNet normalisation statistics (used in CUB-200-2011 preprocessing)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def plot_image_attribution(
    image: np.ndarray,
    pixel_heatmap: np.ndarray,
    segment_map: np.ndarray,
    predicted_class_name: str,
    confidence: float,
    method: str,
) -> plt.Figure:
    """
    Two-subplot figure: (left) raw image, (right) heatmap overlay.

    Parameters
    ----------
    image : np.ndarray, shape (3, H, W), normalised float32
    pixel_heatmap : np.ndarray, shape (H, W)
    segment_map : np.ndarray, shape (H, W)   [unused here, kept for interface consistency]
    predicted_class_name : str
    confidence : float   [0, 1]
    method : str
    """
    # Denormalise for display
    img_hwc = image.transpose(1, 2, 0)
    img_display = img_hwc * _IMAGENET_STD + _IMAGENET_MEAN
    img_display = np.clip(img_display, 0, 1)

    # Normalise heatmap to [-1, 1] for symmetric colourmap
    hmap = pixel_heatmap.copy().astype(float)
    vmax = np.abs(hmap).max()
    if vmax > 0:
        hmap /= vmax

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: raw image
    axes[0].imshow(img_display)
    axes[0].set_title(
        f"Prediction: {predicted_class_name}\n(confidence: {confidence*100:.1f}%)",
        fontsize=10,
    )
    axes[0].axis("off")

    # Right: heatmap overlay
    axes[1].imshow(img_display)
    im = axes[1].imshow(hmap, cmap="RdYlGn", alpha=0.5, vmin=-1, vmax=1)
    axes[1].set_title(f"{method} attribution overlay", fontsize=10)
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04,
                 label="Attribution (normalised)")

    fig.tight_layout()
    return fig
