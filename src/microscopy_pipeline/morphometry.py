"""Geometric measurements for labelled regions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.measure import label, regionprops_table


def measure_components(mask: np.ndarray, pixel_size_um: float | None = None) -> pd.DataFrame:
    labels = label(np.asarray(mask, dtype=bool), connectivity=2)
    props = regionprops_table(
        labels,
        properties=("label", "area", "centroid", "axis_major_length",
                    "axis_minor_length", "equivalent_diameter_area",
                    "eccentricity", "solidity", "bbox"),
    )
    df = pd.DataFrame(props)
    if df.empty:
        return df
    df = df.rename(columns={
        "area": "area_px2",
        "centroid-0": "centroid_y_px",
        "centroid-1": "centroid_x_px",
        "axis_major_length": "major_axis_px",
        "axis_minor_length": "minor_axis_px",
        "equivalent_diameter_area": "equivalent_diameter_px",
        "bbox-0": "bbox_min_y",
        "bbox-1": "bbox_min_x",
        "bbox-2": "bbox_max_y",
        "bbox-3": "bbox_max_x",
    })
    if pixel_size_um is not None:
        scale = float(pixel_size_um)
        if scale <= 0:
            raise ValueError("pixel_size_um must be positive")
        df["area_um2"] = df["area_px2"] * scale**2
        for col in ("major_axis_px", "minor_axis_px", "equivalent_diameter_px"):
            df[col.replace("_px", "_um")] = df[col] * scale
        df["centroid_y_um"] = df["centroid_y_px"] * scale
        df["centroid_x_um"] = df["centroid_x_px"] * scale
    return df


def mask_area(mask: np.ndarray, pixel_size_um: float | None = None) -> dict:
    area_px2 = int(np.count_nonzero(mask))
    result = {"foreground_area_px2": area_px2}
    if pixel_size_um is not None:
        result["foreground_area_um2"] = area_px2 * float(pixel_size_um) ** 2
    return result
