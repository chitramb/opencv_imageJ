"""Preprocessing helpers. Display normalization must never feed quantitative measurements."""
from __future__ import annotations

import numpy as np
from skimage import exposure, filters, morphology


def display_uint8(image: np.ndarray) -> np.ndarray:
    """Create an 8-bit display copy using percentile contrast; source is unchanged."""
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError("display_uint8 expects a single 2-D channel")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, (1, 99))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = exposure.rescale_intensity(arr.astype(np.float32), in_range=(lo, hi), out_range=(0, 255))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def background_subtract(image: np.ndarray, radius: int = 35) -> np.ndarray:
    """Rolling morphological opening background estimate; returns float32 residual."""
    if image.ndim != 2:
        raise ValueError("background_subtract expects a 2-D grayscale image")
    image_f = image.astype(np.float32, copy=False)
    footprint = morphology.disk(max(1, int(radius)))
    background = morphology.opening(image_f, footprint)
    return np.clip(image_f - background, 0, None)


def otsu_mask(image: np.ndarray, min_object_size: int = 80) -> np.ndarray:
    """Generic foreground mask for QC; not a semantic cell/oocyte segmentation."""
    corrected = background_subtract(image)
    if not np.isfinite(corrected).any() or float(corrected.max()) <= 0:
        return np.zeros(image.shape, dtype=bool)
    threshold = filters.threshold_otsu(corrected)
    mask = corrected > threshold
    mask = morphology.remove_small_objects(mask, min_size=max(1, min_object_size))
    mask = morphology.remove_small_holes(mask, area_threshold=max(1, min_object_size))
    return np.asarray(mask, dtype=bool)
