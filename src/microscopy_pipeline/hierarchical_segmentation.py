"""Hierarchical, annotation-guided segmentation for Drosophila egg chambers.

Biological compartments (egg chamber, oocyte and border-cell search region)
are supplied as polygons. DAPI nuclei and the DE-cadherin-positive cluster
are segmented automatically inside those constrained regions.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_local
from skimage.measure import label, regionprops
from skimage.morphology import disk, binary_closing, binary_opening, remove_small_objects
from skimage.segmentation import watershed


def polygon_mask(shape: tuple[int, int], points: list[list[float]]) -> np.ndarray:
    """Rasterise an [x, y] polygon into a boolean mask."""
    if len(points) < 3:
        raise ValueError("A compartment polygon needs at least three [x, y] points.")
    canvas = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(canvas, [np.asarray(points, dtype=np.float32).round().astype(np.int32)], 1)
    return canvas.astype(bool)


def segment_dapi_nuclei(
    dapi: np.ndarray,
    oocyte_mask: np.ndarray,
    *,
    min_area_px: int = 40,
    max_area_px: int = 6000,
    local_block_size: int = 51,
    local_offset: float = 0.0,
    seed_min_distance: int = 5,
) -> np.ndarray:
    """Detect DAPI-positive object candidates only inside the annotated oocyte.

    Local thresholding and distance-transform watershed split touching objects.
    Validate parameters against manually counted nuclei before interpreting the
    result as a cell/nucleus count.
    """
    image, roi = np.asarray(dapi), np.asarray(oocyte_mask, dtype=bool)
    if image.ndim != 2 or image.shape != roi.shape:
        raise ValueError("DAPI must be 2-D and match the oocyte mask.")
    if not roi.any():
        return np.zeros_like(roi)
    smooth = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1.2)
    block = max(3, int(local_block_size) | 1)
    thresh = threshold_local(smooth, block_size=block, method="gaussian", offset=local_offset)
    binary = binary_opening((smooth > thresh) & roi, disk(1))
    binary = remove_small_objects(binary, min_size=max(2, min_area_px // 3))
    distance = ndi.distance_transform_edt(binary)
    coords = peak_local_max(
        distance, labels=binary.astype(np.uint8),
        min_distance=max(1, seed_min_distance), exclude_border=False,
    )
    markers = np.zeros(binary.shape, dtype=np.int32)
    for idx, (y, x) in enumerate(coords, start=1):
        markers[y, x] = idx
    if not markers.any():
        return np.zeros_like(roi)
    labels = watershed(-distance, markers, mask=binary)
    out = np.zeros_like(roi)
    for region in regionprops(labels):
        if min_area_px <= region.area <= max_area_px:
            out[labels == region.label] = True
    return out


def segment_border_cluster(
    decad: np.ndarray,
    search_mask: np.ndarray,
    egg_chamber_mask: np.ndarray,
    *,
    min_area_px: int = 20,
    local_block_size: int = 51,
    local_offset: float = 0.0,
    closing_radius: int = 2,
) -> np.ndarray:
    """Find DE-cadherin-positive candidates inside a tightly annotated search ROI.

    Returns the largest connected candidate. The search ROI must be manually
    placed around the expected cluster; this is not a border-cell classifier.
    """
    image = np.asarray(decad)
    search, chamber = np.asarray(search_mask, bool), np.asarray(egg_chamber_mask, bool)
    if image.ndim != 2 or image.shape != search.shape or search.shape != chamber.shape:
        raise ValueError("DEcad, search and egg-chamber masks must be matching 2-D arrays.")
    roi = search & chamber
    if not roi.any():
        return np.zeros_like(roi)
    smooth = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1.0)
    block = max(3, int(local_block_size) | 1)
    thresh = threshold_local(smooth, block_size=block, method="gaussian", offset=local_offset)
    candidate = binary_closing((smooth > thresh) & roi, disk(max(1, closing_radius)))
    candidate = remove_small_objects(candidate, min_size=min_area_px)
    labels = label(candidate)
    if labels.max() == 0:
        return np.zeros_like(roi)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(np.argmax(counts))


def build_compartment_masks(
    shape: tuple[int, int],
    annotations: dict,
    dapi: np.ndarray,
    decad: np.ndarray,
    *,
    nuclei_params: dict | None = None,
    border_params: dict | None = None,
) -> "CompartmentMasks":
    """Create nested masks from polygons: egg_chamber, oocyte, border_search."""
    missing = [k for k in ("egg_chamber", "oocyte", "border_search") if k not in annotations]
    if missing:
        raise ValueError(f"Missing annotation polygon(s): {', '.join(missing)}")
    chamber = polygon_mask(shape, annotations["egg_chamber"])
    oocyte = polygon_mask(shape, annotations["oocyte"]) & chamber
    search = polygon_mask(shape, annotations["border_search"]) & chamber
    nuclei = segment_dapi_nuclei(dapi, oocyte, **(nuclei_params or {}))
    border = segment_border_cluster(decad, search, chamber, **(border_params or {}))
    return CompartmentMasks(chamber, oocyte, nuclei, border)


from dataclasses import dataclass


@dataclass
class CompartmentMasks:
    egg_chamber: np.ndarray
    oocyte: np.ndarray
    nuclei: np.ndarray
    border_cluster: np.ndarray


def mask_metrics(masks: CompartmentMasks) -> dict[str, float | int]:
    """Return pixel-based areas, axes and agreed density metrics."""
    chamber_props = regionprops(masks.egg_chamber.astype(np.uint8))
    oocyte_props = regionprops(masks.oocyte.astype(np.uint8))
    chamber_area, oocyte_area = int(masks.egg_chamber.sum()), int(masks.oocyte.sum())
    cluster_area = int(masks.border_cluster.sum())
    nuclei_count = int(label(masks.nuclei).max())
    return {
        "egg_chamber_area_px2": chamber_area,
        "egg_chamber_major_axis_px": float(chamber_props[0].axis_major_length) if chamber_props else 0.0,
        "egg_chamber_minor_axis_px": float(chamber_props[0].axis_minor_length) if chamber_props else 0.0,
        "oocyte_area_px2": oocyte_area,
        "oocyte_major_axis_px": float(oocyte_props[0].axis_major_length) if oocyte_props else 0.0,
        "oocyte_minor_axis_px": float(oocyte_props[0].axis_minor_length) if oocyte_props else 0.0,
        "oocyte_nuclei_count": nuclei_count,
        "oocyte_nuclei_density_per_px2": nuclei_count / oocyte_area if oocyte_area else 0.0,
        "border_cluster_area_px2": cluster_area,
        "border_cluster_fraction_of_chamber": cluster_area / chamber_area if chamber_area else 0.0,
        "border_cluster_percent_of_chamber": 100.0 * cluster_area / chamber_area if chamber_area else 0.0,
    }


def save_segmentation(
    output_dir: str | Path,
    image: np.ndarray,
    masks: CompartmentMasks,
    metrics: dict | None = None,
) -> None:
    """Write binary masks, a colour overlay and optional metrics CSV."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    named = {
        "egg_chamber": masks.egg_chamber, "oocyte": masks.oocyte,
        "oocyte_nuclei": masks.nuclei, "border_cluster": masks.border_cluster,
    }
    for name, mask in named.items():
        cv2.imwrite(str(out / f"{name}_mask.png"), mask.astype(np.uint8) * 255)
    gray = np.asarray(image)
    if gray.ndim != 2:
        raise ValueError("Overlay base image must be a 2-D grayscale image.")
    base = cv2.cvtColor(cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    colour = np.zeros_like(base)
    colour[masks.egg_chamber] = (0, 180, 255)   # orange
    colour[masks.oocyte] = (255, 160, 0)        # blue
    colour[masks.nuclei] = (0, 255, 0)           # green
    colour[masks.border_cluster] = (0, 0, 255)  # red
    cv2.imwrite(str(out / "compartment_overlay.png"), cv2.addWeighted(base, 0.6, colour, 0.4, 0))
    if metrics is not None:
        pd.DataFrame([metrics]).to_csv(out / "biological_metrics.csv", index=False)


def load_annotations(path: str | Path) -> dict:
    """Load polygon annotations from JSON."""
    import json
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
