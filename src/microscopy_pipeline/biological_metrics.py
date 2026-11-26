"""Biological endpoint calculations from validated segmentation masks.

These functions intentionally do not perform semantic segmentation. Supply masks
validated by a human or a validated model, registered to the same image grid.
"""
from __future__ import annotations

import numpy as np


def oocyte_count_density(oocyte_mask: np.ndarray,
                         count_objects: int,
                         pixel_size_um: float | None = None) -> dict:
    """Return object count and count per projected oocyte area.

    count_objects must be a count of the biological objects defined by the study
    (for example, nuclei) inside the oocyte ROI. It is not inferred from area.
    """
    mask = np.asarray(oocyte_mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("oocyte_mask must be a non-empty 2-D mask")
    if count_objects < 0:
        raise ValueError("count_objects cannot be negative")
    area_px2 = int(mask.sum())
    result = {
        "oocyte_area_px2": area_px2,
        "oocyte_object_count": int(count_objects),
        "oocyte_count_density_per_px2": float(count_objects / area_px2),
    }
    if pixel_size_um is not None:
        if pixel_size_um <= 0:
            raise ValueError("pixel_size_um must be positive")
        area_um2 = area_px2 * float(pixel_size_um) ** 2
        result["oocyte_area_um2"] = area_um2
        result["oocyte_count_density_per_um2"] = float(count_objects / area_um2)
    return result


def border_cluster_area_fraction(border_cluster_mask: np.ndarray,
                                 egg_chamber_mask: np.ndarray) -> dict:
    """Return border-cell cluster projected area and fraction of egg chamber area."""
    cluster = np.asarray(border_cluster_mask, dtype=bool)
    chamber = np.asarray(egg_chamber_mask, dtype=bool)
    if cluster.ndim != 2 or chamber.ndim != 2 or cluster.shape != chamber.shape:
        raise ValueError("Masks must be 2-D and have identical shapes")
    chamber_area = int(chamber.sum())
    if chamber_area == 0:
        raise ValueError("egg_chamber_mask must be non-empty")
    cluster_area = int(cluster.sum())
    outside = int(np.count_nonzero(cluster & ~chamber))
    fraction = float(np.count_nonzero(cluster & chamber) / chamber_area)
    return {
        "border_cluster_area_px2": cluster_area,
        "egg_chamber_area_px2": chamber_area,
        "border_cluster_area_fraction_of_chamber": fraction,
        "border_cluster_percent_of_chamber": 100.0 * fraction,
        "border_cluster_pixels_outside_chamber": outside,
    }
