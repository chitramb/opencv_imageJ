"""Fluorescence measurements for original grayscale channel acquisitions."""
from __future__ import annotations

import numpy as np


def quantify_region(image: np.ndarray, roi_mask: np.ndarray,
                    background_mask: np.ndarray | None = None) -> dict:
    """Measure raw ROI intensity and optional local-background-corrected intensity.

    Use only original single-channel acquisition data, not pseudocolour composites.
    ROI and background masks must be spatially registered to the image.
    """
    if image.ndim != 2:
        raise ValueError("Fluorescence quantification requires a single 2-D channel")
    roi = np.asarray(roi_mask, dtype=bool)
    if roi.shape != image.shape:
        raise ValueError("ROI mask shape must match image")
    values = image[roi].astype(np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("ROI contains no finite pixels")
    result = {
        "roi_area_px2": int(values.size),
        "mean_intensity_raw": float(values.mean()),
        "median_intensity_raw": float(np.median(values)),
        "integrated_intensity_raw": float(values.sum()),
        "max_intensity_raw": float(values.max()),
        "saturated_pixel_fraction": float(np.mean(values >= np.iinfo(image.dtype).max))
            if np.issubdtype(image.dtype, np.integer) else np.nan,
    }
    if background_mask is not None:
        bg = np.asarray(background_mask, dtype=bool)
        if bg.shape != image.shape:
            raise ValueError("Background mask shape must match image")
        bg_values = image[bg].astype(np.float64)
        bg_values = bg_values[np.isfinite(bg_values)]
        if bg_values.size:
            bg_mean = float(bg_values.mean())
            result["background_mean"] = bg_mean
            result["mean_intensity_corrected"] = result["mean_intensity_raw"] - bg_mean
            result["integrated_intensity_corrected"] = (
                result["mean_intensity_raw"] - bg_mean
            ) * result["roi_area_px2"]
    return result
