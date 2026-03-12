"""
CUB-200-2011 Dataset Loader for Phase 1 Image XAI Evaluation.

Downloads and preprocesses the Caltech-UCSD Birds 200-2011 dataset.
Returns train/test splits with ImageNet normalisation for ResNet-50,
plus ground-truth annotations for localisation metrics.

Dataset structure after download:
    data/cub200/CUB_200_2011/
        images/                  # 11,788 JPEG images
        bounding_boxes.txt       # (image_id, x, y, width, height)
        image_class_labels.txt   # (image_id, class_id)
        train_test_split.txt     # (image_id, is_training_image)
        images.txt               # (image_id, filename)
        segmentations/           # binary PNG segmentation masks (same stem as images)
"""
import os
import io
import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from phase1.image.config import (
    IMAGE_DATA_DIR, IMAGE_SIZE, EVAL_SAMPLE_SIZE, RANDOM_STATE,
    GRADSHAP_BACKGROUND
)
from phase1.image.utils import compute_baseline_image

# CUB-200-2011 official download URL
_CUB_URL = (
    "https://data.caltech.edu/records/65de6-vp158/files/"
    "CUB_200_2011.tgz"
)
_CACHE_FILE = IMAGE_DATA_DIR / "cub200_processed.pkl"

# ImageNet normalisation constants (for ResNet-50)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def load_cub200(force_reload: bool = False) -> dict:
    """
    Load CUB-200-2011 dataset, downloading if necessary.

    Returns
    -------
    dict with keys:
        X_train      : np.ndarray (N_train, 3, H, W) float32, normalised
        X_test       : np.ndarray (N_test,  3, H, W) float32, normalised
        X_train_raw  : np.ndarray (N_train, 3, H, W) float32, in [0,1] (un-normalised)
        X_test_raw   : np.ndarray (N_test,  3, H, W) float32, in [0,1]
        y_train      : np.ndarray (N_train,) int
        y_test       : np.ndarray (N_test,)  int
        baseline_image    : np.ndarray (3, H, W) per-channel training mean (raw)
        background_images : np.ndarray (GRADSHAP_BACKGROUND, 3, H, W) normalised
        annotations  : dict  image_id (int) -> {
                           'bbox': (x, y, w, h),
                           'seg_path': Path | None,
                           'class_id': int,
                           'split': 'train'|'test',
                       }
        test_ids     : list[int]  image_ids for test split (aligned with X_test)
        class_names  : list[str]
    """
    IMAGE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if _CACHE_FILE.exists() and not force_reload:
        print(f"  Loading CUB-200-2011 from cache: {_CACHE_FILE}")
        with open(_CACHE_FILE, "rb") as f:
            return pickle.load(f)

    cub_root = IMAGE_DATA_DIR / "CUB_200_2011"
    if not cub_root.exists() or force_reload:
        _download_and_extract()

    print("  Parsing CUB-200-2011 annotations...")
    annotations, class_names = _parse_annotations(cub_root)

    print("  Loading and resizing images...")
    data = _build_arrays(cub_root, annotations, class_names)

    print(f"  Caching to {_CACHE_FILE}...")
    with open(_CACHE_FILE, "wb") as f:
        pickle.dump(data, f)

    return data


def unnormalise(images: np.ndarray) -> np.ndarray:
    """
    Reverse ImageNet normalisation.

    Parameters
    ----------
    images : (N, 3, H, W) or (3, H, W), normalised float32

    Returns
    -------
    np.ndarray, float32 in [0, 1]
    """
    mean = _IMAGENET_MEAN[:, None, None]
    std  = _IMAGENET_STD[:, None, None]
    return np.clip(images * std + mean, 0.0, 1.0)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _download_and_extract():
    tgz_path = IMAGE_DATA_DIR / "CUB_200_2011.tgz"
    if not tgz_path.exists():
        print(f"  Downloading CUB-200-2011 (~1.1 GB) from {_CUB_URL} ...")
        urllib.request.urlretrieve(_CUB_URL, tgz_path,
                                   reporthook=_download_progress)
        print()
    print(f"  Extracting {tgz_path} ...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(IMAGE_DATA_DIR)
    print("  Extraction complete.")


def _download_progress(count, block_size, total_size):
    pct = min(count * block_size / total_size * 100, 100)
    print(f"\r  Progress: {pct:.1f}%", end="", flush=True)


def _parse_annotations(cub_root: Path):
    # image id → filename
    id2file = {}
    with open(cub_root / "images.txt") as f:
        for line in f:
            img_id, fname = line.strip().split()
            id2file[int(img_id)] = fname

    # image id → class id (1-indexed)
    id2class = {}
    with open(cub_root / "image_class_labels.txt") as f:
        for line in f:
            img_id, cls_id = line.strip().split()
            id2class[int(img_id)] = int(cls_id) - 1  # 0-indexed

    # image id → train/test split
    id2split = {}
    with open(cub_root / "train_test_split.txt") as f:
        for line in f:
            img_id, is_train = line.strip().split()
            id2split[int(img_id)] = "train" if int(is_train) == 1 else "test"

    # image id → bounding box (x, y, width, height) — original pixel coords
    id2bbox = {}
    with open(cub_root / "bounding_boxes.txt") as f:
        for line in f:
            parts = line.strip().split()
            img_id = int(parts[0])
            id2bbox[img_id] = tuple(float(v) for v in parts[1:])  # x,y,w,h

    # class names
    class_names = []
    with open(cub_root / "classes.txt") as f:
        for line in f:
            class_names.append(line.strip().split(None, 1)[1])

    # segmentation mask paths (optional — may not be present in all versions)
    seg_dir = cub_root / "segmentations"

    annotations = {}
    for img_id, fname in id2file.items():
        seg_path = seg_dir / fname.replace(".jpg", ".png")
        annotations[img_id] = {
            "filename": fname,
            "bbox": id2bbox.get(img_id),
            "seg_path": seg_path if seg_path.exists() else None,
            "class_id": id2class[img_id],
            "split": id2split[img_id],
        }

    return annotations, class_names


def _load_image(img_path: Path) -> np.ndarray:
    """Load and resize to (3, IMAGE_SIZE, IMAGE_SIZE), return float32 in [0,1]."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0   # (H, W, 3)
    return arr.transpose(2, 0, 1)                    # (3, H, W)


def _normalise(images: np.ndarray) -> np.ndarray:
    """Apply ImageNet normalisation. images: (N,3,H,W) or (3,H,W) in [0,1]."""
    mean = _IMAGENET_MEAN[:, None, None]
    std  = _IMAGENET_STD[:, None, None]
    return (images - mean) / std


def _scale_bbox(bbox, orig_w: int, orig_h: int) -> tuple:
    """Rescale bounding box from original image size to IMAGE_SIZE x IMAGE_SIZE."""
    x, y, w, h = bbox
    sx = IMAGE_SIZE / orig_w
    sy = IMAGE_SIZE / orig_h
    return (x * sx, y * sy, w * sx, h * sy)


def _build_arrays(cub_root: Path, annotations: dict, class_names: list) -> dict:
    train_ids, test_ids = [], []
    for img_id, ann in annotations.items():
        if ann["split"] == "train":
            train_ids.append(img_id)
        else:
            test_ids.append(img_id)

    train_ids.sort()
    test_ids.sort()

    def load_split(ids):
        imgs_raw, labels = [], []
        bboxes_scaled = {}
        for img_id in ids:
            ann = annotations[img_id]
            img_path = cub_root / "images" / ann["filename"]
            raw = _load_image(img_path)                 # (3, H, W) in [0,1]

            # Scale bbox to resized image coordinates
            if ann["bbox"] is not None:
                orig = Image.open(img_path)
                bboxes_scaled[img_id] = _scale_bbox(
                    ann["bbox"], orig.width, orig.height
                )
            imgs_raw.append(raw)
            labels.append(ann["class_id"])
        return np.stack(imgs_raw), np.array(labels, dtype=np.int64), bboxes_scaled

    print("    Loading training images...")
    X_train_raw, y_train, _ = load_split(train_ids)
    print("    Loading test images...")
    X_test_raw,  y_test,  test_bboxes = load_split(test_ids)

    # Compute per-channel mean baseline from training set (raw, un-normalised)
    baseline_image = compute_baseline_image(X_train_raw)

    # Normalise for ResNet-50
    X_train = _normalise(X_train_raw)
    X_test  = _normalise(X_test_raw)

    # Background images for GradientSHAP reference distribution
    rng = np.random.default_rng(RANDOM_STATE)
    bg_idx = rng.choice(len(X_train), size=GRADSHAP_BACKGROUND, replace=False)
    background_images = X_train[bg_idx]

    # Update annotations with scaled bboxes
    for img_id, bbox in test_bboxes.items():
        annotations[img_id]["bbox_scaled"] = bbox

    return {
        "X_train":         X_train,
        "X_test":          X_test,
        "X_train_raw":     X_train_raw,
        "X_test_raw":      X_test_raw,
        "y_train":         y_train,
        "y_test":          y_test,
        "baseline_image":  baseline_image,
        "background_images": background_images,
        "annotations":     annotations,
        "test_ids":        test_ids,
        "class_names":     class_names,
    }
