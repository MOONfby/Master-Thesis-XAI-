"""
GradientSHAP Explainer wrapper for Phase 1 Image XAI Evaluation.

Uses shap.GradientExplainer (GradientSHAP) to compute pixel-level SHAP
values for a PyTorch ResNet-50 model, then aggregates to superpixel level.

GradientSHAP is the image-domain analogue of Phase 1 tabular SHAPExplainer
(TreeExplainer). Both belong to the SHAP family and bridge Phase 1 → Phase 2
(the ExplainerNetwork will be trained to approximate GradientSHAP outputs).
"""
import time
import numpy as np
import torch
import shap

from phase1.image.config import (
    N_SEGMENTS, GRADSHAP_N_SAMPLES, GRADSHAP_STDEV_NOISE, RANDOM_STATE
)
from phase1.image.utils import aggregate_to_superpixels


class GradientSHAPExplainer:
    """
    GradientSHAP explanation for image classification.

    Produces superpixel-level SHAP values in (S,) format,
    comparable with LIMEImageExplainer output.

    Parameters
    ----------
    model : torch.nn.Module
        ResNet-50 in eval mode. Must accept (N, 3, H, W) tensors.
    background_images : np.ndarray, shape (B, 3, H, W), normalised float32
        Reference distribution for SHAP baseline.
        Analogous to tabular SHAP background samples.
    """

    def __init__(self, model: torch.nn.Module,
                 background_images: np.ndarray):
        self.model  = model
        self.device = next(model.parameters()).device
        self._bg    = torch.tensor(background_images,
                                   dtype=torch.float32,
                                   device=self.device)
        self._explainer = shap.GradientExplainer(model, self._bg)

    # ------------------------------------------------------------------
    # Core interface — mirrors phase1/tabular SHAPExplainer
    # ------------------------------------------------------------------

    def explain(self, image: np.ndarray, segment_map: np.ndarray,
                label: int = None) -> np.ndarray:
        """
        Explain a single image.

        Parameters
        ----------
        image : np.ndarray, shape (3, H, W), normalised float32
        segment_map : np.ndarray, shape (H, W), int
        label : int or None
            Class to explain. If None, uses argmax of model prediction.

        Returns
        -------
        np.ndarray, shape (S,)
            Superpixel-level SHAP values (mean of |pixel SHAP| per segment).
        """
        pixel_heatmap = self.explain_pixel_level(image, label=label)
        return aggregate_to_superpixels(pixel_heatmap, segment_map, n_segments=N_SEGMENTS)

    def explain_batch(self, images: np.ndarray, segment_maps: np.ndarray,
                      labels: np.ndarray = None) -> np.ndarray:
        """
        Explain a batch of images.

        Parameters
        ----------
        images : np.ndarray, shape (N, 3, H, W)
        segment_maps : np.ndarray, shape (N, H, W)
        labels : np.ndarray (N,) int or None

        Returns
        -------
        np.ndarray, shape (N, S)
        """
        N = len(images)
        results = []
        for i in range(N):
            lbl = int(labels[i]) if labels is not None else None
            attrs = self.explain(images[i], segment_maps[i], label=lbl)
            results.append(attrs)
            if (i + 1) % 10 == 0:
                print(f"    GradientSHAP: {i+1}/{N} done")
        return np.stack(results)

    def explain_pixel_level(self, image: np.ndarray,
                             label: int = None) -> np.ndarray:
        """
        Return pixel-level SHAP heatmap (H, W).

        Aggregates across colour channels by summing absolute values,
        giving a single importance score per spatial location.

        Used for localisation metrics and for aggregation to superpixels.

        Returns
        -------
        np.ndarray, shape (H, W), non-negative
        """
        t = torch.tensor(image[None], dtype=torch.float32,
                         device=self.device)   # (1, 3, H, W)

        with torch.no_grad():
            logits = self.model(t)
        if label is None:
            label = int(logits[0].argmax().item())

        shap_values = self._explainer.shap_values(
            t, ranked_outputs=None
        )
        if isinstance(shap_values, list):
            # Old shap API: list of arrays, one per class → index by label
            sv = shap_values[label][0]   # (3, H, W)
        else:
            # New shap API: single ndarray (N, 3, H, W) for top predicted class
            sv = shap_values[0]          # (3, H, W)

        sv = np.abs(sv)
        heatmap = sv.sum(axis=0)        # (H, W)
        return heatmap

    def measure_runtime(self, images: np.ndarray,
                        segment_maps: np.ndarray) -> dict:
        """
        Measure per-image explanation time.

        Returns
        -------
        dict with mean_time, std_time, total_time, n_instances
        """
        times = []
        for i in range(len(images)):
            t0 = time.perf_counter()
            self.explain(images[i], segment_maps[i])
            times.append(time.perf_counter() - t0)
        times = np.array(times)
        return {
            "mean_time":  float(times.mean()),
            "std_time":   float(times.std()),
            "total_time": float(times.sum()),
            "n_instances": len(times),
        }
