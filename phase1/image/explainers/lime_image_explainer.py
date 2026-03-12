"""
LIME Image Explainer wrapper for Phase 1 Image XAI Evaluation.

Wraps lime.lime_image.LimeImageExplainer to implement the unified
explainer interface: explain() returns (S,) superpixel-level attributions,
matching the Phase 1 tabular LIMEExplainer interface contract.
"""
import time
import numpy as np
from lime.lime_image import LimeImageExplainer as _LimeImageExplainer

from phase1.image.config import (
    N_SEGMENTS, LIME_NUM_SAMPLES, LIME_NUM_FEATURES, RANDOM_STATE
)
from phase1.image.utils import aggregate_to_superpixels


class LIMEImageExplainer:
    """
    LIME explanation for image classification.

    Produces superpixel-level attribution scores in (S,) format,
    directly comparable with GradientSHAPExplainer output.

    Parameters
    ----------
    predict_fn : callable
        (np.ndarray (N, 3, H, W)) -> np.ndarray (N, num_classes)
        Must accept batches of normalised images.
    baseline_image : np.ndarray, shape (3, H, W), float in [0, 1]
        Per-channel mean image used to hide (mask) superpixels.
        Analogous to tabular baseline_values.
    """

    def __init__(self, predict_fn, baseline_image: np.ndarray):
        self.predict_fn    = predict_fn
        self.baseline_image = baseline_image  # (3, H, W), raw [0,1]
        self._explainer    = _LimeImageExplainer(random_state=RANDOM_STATE)

    # ------------------------------------------------------------------
    # Core interface — mirrors phase1/tabular LIMEExplainer
    # ------------------------------------------------------------------

    def explain(self, image: np.ndarray, segment_map: np.ndarray,
                label: int = None,
                num_samples: int = LIME_NUM_SAMPLES) -> np.ndarray:
        """
        Explain a single image.

        Parameters
        ----------
        image : np.ndarray, shape (3, H, W), normalised float32
        segment_map : np.ndarray, shape (H, W), int — SLIC labels
        label : int or None
            Class to explain. If None, uses argmax of model prediction.
        num_samples : int

        Returns
        -------
        np.ndarray, shape (S,)
            Superpixel attribution scores for the specified class.
        """
        # LIME expects (H, W, 3) uint8 or float
        img_hwc = image.transpose(1, 2, 0)          # (H, W, 3)
        baseline_hwc = self.baseline_image.transpose(1, 2, 0)

        if label is None:
            probs = self.predict_fn(image[None])     # (1, C)
            label = int(probs[0].argmax())

        def _predict_lime(images_hwc):
            # images_hwc: (N, H, W, 3) float in any range → convert to (N,3,H,W)
            imgs = images_hwc.transpose(0, 3, 1, 2).astype(np.float32)
            return self.predict_fn(imgs)

        explanation = self._explainer.explain_instance(
            img_hwc,
            _predict_lime,
            top_labels=None,
            labels=[label],
            hide_color=None,          # will use mean of segments
            num_samples=num_samples,
            segmentation_fn=lambda x: segment_map,  # use pre-computed SLIC
        )

        # Extract weights for the target label: list of (segment_id, weight)
        seg_weights = dict(explanation.local_exp[label])
        S = int(segment_map.max()) + 1
        attrs = np.array([seg_weights.get(s, 0.0) for s in range(S)],
                         dtype=np.float64)
        return attrs

    def explain_batch(self, images: np.ndarray, segment_maps: np.ndarray,
                      labels: np.ndarray = None,
                      num_samples: int = LIME_NUM_SAMPLES) -> np.ndarray:
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
            attrs = self.explain(images[i], segment_maps[i],
                                 label=lbl, num_samples=num_samples)
            results.append(attrs)
            if (i + 1) % 10 == 0:
                print(f"    LIME-image: {i+1}/{N} done")
        return np.stack(results)

    def explain_pixel_level(self, image: np.ndarray, segment_map: np.ndarray,
                             label: int = None,
                             num_samples: int = LIME_NUM_SAMPLES) -> np.ndarray:
        """
        Return pixel-level heatmap by broadcasting superpixel attributions.

        Used for localisation metrics (Pointing Game, Segmentation IoU).

        Returns
        -------
        np.ndarray, shape (H, W)
        """
        attrs = self.explain(image, segment_map, label=label,
                             num_samples=num_samples)
        H, W = segment_map.shape
        heatmap = np.zeros((H, W), dtype=np.float64)
        for s, a in enumerate(attrs):
            heatmap[segment_map == s] = a
        return heatmap

    def measure_runtime(self, images: np.ndarray, segment_maps: np.ndarray,
                        num_samples: int = LIME_NUM_SAMPLES) -> dict:
        """
        Measure per-image explanation time.

        Returns
        -------
        dict with mean_time, std_time, total_time, n_instances
        """
        times = []
        for i in range(len(images)):
            t0 = time.perf_counter()
            self.explain(images[i], segment_maps[i], num_samples=num_samples)
            times.append(time.perf_counter() - t0)
        times = np.array(times)
        return {
            "mean_time":  float(times.mean()),
            "std_time":   float(times.std()),
            "total_time": float(times.sum()),
            "n_instances": len(times),
        }
