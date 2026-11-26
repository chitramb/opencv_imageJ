"""Hierarchical, annotation-guided segmentation for Drosophila egg chambers.

The biological compartments (egg chamber, oocyte and border-cell search region)
are supplied as polygons. DAPI nuclei and the DE-cadherin-positive border-cell
cluster are then segmented automatically inside those constrained regions.
This human-in-the-loop design avoids assigning biological identities from a
generic whole-image threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu, threshold_local
from skimage.measure import label, regionprops
from skimage.morphology import (
    disk, binary_closing, binary_opening, remove_small_objects,
)
from skimage.segmentation import watershed
from scipy import ndimage as ndi


@dataclass
class CompartmentMasks:
    egg_chamber: np.ndarray
    oocyte: np.ndarray
    nuclei: np.ndarray
    border_cluster: np.ndarray


def polygon_mask(shape: tuple[int, int], points: list[list[float]]) -> np.ndarray:
    """Rasterise an [x, y] polygon into a boolean mask."""
    if len(points) < 3:
        raise ValueError("A compartment polygon needs at least three [x, y] points.")
    canvas = np.zeros(shape, dtype=np.uint8)
    pts = np.asarray(points, dtype=np.float32).round().astype(np.int32)
    cv2.fillPoly(canvas, [pts], 1)
    return canvas.astype(bool)


def segment_dapi_nuclei(
    dapi: np.ndarray,
    oocyte_mask: np.ndarray,
    *,
    min_area_px: int = 40,
    max_area_px: int = 6000,
    local_block_size: int = 51,
    local_offset: float = 0.0,
) -> np.ndarray:
    """Detect DAPI-positive objects only within the annotated oocyte.

    Uses local thresholding followed by distance-transform watershed. The
    returned objects are candidates, not guaranteed one-to-one cell counts.
    Tune size/threshold parameters against manually counted nuclei.
    """
    image = np.asarray(dapi)
    roi = np.asarray(oocyte_mask, dtype=bool)
    if image.ndim != 2 or image.shape != roi.shape:
        raise ValueError("DAPI must be 2-D and match the oocyte mask.")
    if not roi.any():
        return np.zeros_like(roi)

    smooth = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1.2)
    block = max(3, int(local_block_size) | 1)
    thresh = threshold_local(smooth, block_size=block, method="gaussian", offset=local_offset)
    binary = (smooth > thresh) & roi
    binary = binary_opening(binary, disk(1))
    binary = remove_small_objects(binary, min_size=max(2, min_area_px // 3))

    # Watershed separates touching bright nuclei where a distance peak exists.
    distance = ndi.distance_transform_edt(binary)
    maxima = (distance == ndi.maximum_filter(distance, size= nine()))
    markers, _ = ndi.label(maxima & binary)
    if markers.max() == 0:
        return np.zeros_like(roi)
    labels = watershed(-distance, markers, mask=binary)
    out = np.zeros_like(roi)
    for region in regionprops(labels):
        if min_area_px <= region.area <= max_area_px:
            out[labels == region.label] = True
    return out


def nine() -> int:
    """Small marker-neighbourhood size for watershed seeds."""
    return 9


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
    """Find DE-cadherin-positive cluster candidates in an annotated search ROI.

    The search ROI should tightly enclose the expected border-cell cluster.
    The largest connected candidate is returned; inspect it before use.
    """
    image = np.asarray(decad)
    search = np.asarray(search_mask, dtype=bool)
    chamber = np.asarray(egg_chamber_mask, dtype=bool)
    if image.ndim != 2 or image.shape != search.shape or search.shape != chamber.shape:
        raise ValueError("DEcad, search and egg-chamber masks must be matching 2-D arrays.")
    roi = search & chamber
    if not roi.any():
        return np.zeros_like(roi)

    smooth = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1.0)
    block = max(3, int(local_block_size) | 1)
    thresh = threshold_local(smooth, block_size=block, method="gaussian", offset=local_offset)
    candidate = (smooth > thresh) & roi
    candidate = binary_closing(candidate, disk(max(1, closing_radius)))
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
) -> CompartmentMasks:
    """Create nested compartment masks from polygon annotations.

    Required annotation keys:
      egg_chamber: polygon around one complete egg chamber
      oocyte: polygon around the oocyte, inside the egg chamber
      border_search: tight polygon around the expected border-cell cluster
    """
    required = ("egg_chamber", "oocyte", "border_search")
    missing = [key for key in required if key not in annotations]
    if missing:
        raise ValueError(f"Missing annotation polygon(s): {', '.join(missing)}")
    chamber = polygon_mask(shape, annotations["egg_chamber"])
    oocyte = polygon_mask(shape, annotations["oocyte"]) & chamber
    search = polygon_mask(shape, annotations["border_search"]) & chamber
    nuclei = segment_dapi_nuclei(dapi, oocyte, **(nuclei_params or {}))
    border = segment_border_cluster(decad, search, chamber, **(border_params or {}))
    return CompartmentMasks(chamber, oocyte, nuclei, border)


def mask_metrics(masks: CompartmentMasks) -> dict[str, float | int]:
    """Return pixel-based area, shape and agreed density metrics."""
    chamber_props = regionprops(masks.egg_chamber.astype(np.uint8))
    oocyte_props = regionprops(masks.oocyte.astype(np.uint8))
    chamber_area = int(masks.egg_chamber.sum())
    oocyte_area = int(masks.oocyte.sum())
    cluster_area = int(masks.border_cluster.sum())
    nuclei_labels = label(masks.nuclei)
    nuclei_count = int(nuclei_labels.max())
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
    """Write binary masks, colour overlay and a one-row metrics CSV."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, mask in (
        ("egg_chamber", masks.egg_chamber),
        ("oocyte", masks.oocyte),
        ("oocyte_nuclei", masks.nuclei),
        ("border_cluster", masks.border_cluster),
    ):
        cv2.imwrite(str(out / f"{name}_mask.png"), mask.astype(np.uint8) * 255)

    gray = np.asarray(image)
    if gray.ndim != 2:
        raise ValueError("Overlay base image must be a 2-D grayscale image.")
    base = cv2.cvtColor(cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    overlay[masks.egg_chamber] = (0, 180, 255)       # orange: egg chamber
    overlay[masks.oocyte] = (255, 160, 0)            # blue: oocyte
    overlay[masks.nuclei] = (0, 255, 0)              # green: nuclei
    overlay[masks.border_cluster] = (0, 0, 255)      # red: border cluster
    blended = cv2.addWeighted(base, 0.55, overlay, 0.45, 0)
    cv2.imwrite(str(out / "compartment_overlay.png"), blended)
    if metrics is not None:
        pd.DataFrame([metrics]).to_csv(out / "biological_metrics.csv", index=False)


def load_annotations(path: str | Path) -> dict:
    """Load polygon annotations from a JSON file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
