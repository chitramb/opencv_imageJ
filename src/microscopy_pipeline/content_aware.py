"""Content-aware first-pass compartment proposals for multi-channel egg-chamber images.

AcTub cortical signal proposes chamber envelopes. DAPI spatial distribution proposes
an oocyte/nurse-cell split. DAPI detections are restricted to a strongly eroded
oocyte interior to reduce follicle-cell contamination. DEcad candidates are
restricted to a narrow band around the inferred interface.

These are proposals for visual review, not validated biological segmentations.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d
from skimage.filters import threshold_local
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, remove_small_objects


def propose_egg_chambers(actub: np.ndarray, *, min_component_px: int = 9000) -> list[np.ndarray]:
    """Return filled convex envelopes of AcTub cortical components."""
    image = np.asarray(actub)
    if image.ndim != 2:
        raise ValueError("AcTub input must be a 2-D grayscale image.")
    smooth = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 2.0)
    binary = smooth > threshold_local(smooth, 101, method="gaussian", offset=-2.0)
    binary = remove_small_objects(closing(binary, disk(8)), min_size=250)
    masks = []
    for region in regionprops(label(binary)):
        if region.area < min_component_px:
            continue
        points = region.coords[:, [1, 0]].astype(np.int32)
        hull = cv2.convexHull(points)
        mask = np.zeros(image.shape, np.uint8)
        cv2.fillPoly(mask, [hull], 1)
        mask = mask.astype(bool)
        if mask.sum() > 12000:
            masks.append(mask)
    return masks


def propose_oocyte(chamber: np.ndarray, dapi: np.ndarray) -> np.ndarray:
    """Split the chamber at a DAPI-profile valley; distal low-DAPI side is candidate."""
    chamber, dapi = np.asarray(chamber, bool), np.asarray(dapi)
    if chamber.shape != dapi.shape or not chamber.any():
        raise ValueError("Non-empty chamber and matching DAPI image are required.")
    coords = np.argwhere(chamber).astype(float)
    center = coords.mean(axis=0)
    eigval, eigvec = np.linalg.eigh(np.cov((coords - center).T))
    axis = eigvec[:, np.argmax(eigval)]
    projection = (coords - center) @ axis
    length = projection.max() - projection.min()
    smooth = cv2.GaussianBlur(dapi.astype(np.float32), (0, 0), 5)
    weights = np.maximum(smooth[chamber] - np.percentile(smooth[chamber], 55), 0)
    weighted_center = (coords * weights[:, None]).sum(axis=0) / weights.sum() if weights.sum() else center
    if (weighted_center - center) @ axis > 0:
        axis = -axis
        projection = -projection
    edges = np.linspace(projection.min(), projection.max(), 61)
    signal = np.maximum(smooth[chamber] - np.percentile(smooth[chamber], 55), 0)
    bins = np.clip(np.digitize(projection, edges) - 1, 0, 59)
    profile = np.bincount(bins, weights=signal, minlength=60) / np.maximum(np.bincount(bins, minlength=60), 1)
    profile = gaussian_filter1d(profile, 2)
    centers = (edges[:-1] + edges[1:]) / 2
    valid = np.where((centers >= projection.min() + .38 * length) & (centers <= projection.min() + .78 * length))[0]
    cut = centers[valid[np.argmin(profile[valid])]] if len(valid) else projection.min() + .58 * length
    yy, xx = np.indices(chamber.shape)
    coordinates = np.stack((yy, xx), axis=-1)
    return chamber & (((coordinates - center) @ axis) >= cut)


def propose_oocyte_nuclei(dapi: np.ndarray, oocyte: np.ndarray, *, edge_exclusion_px: int = 45) -> np.ndarray:
    """Detect internal DAPI objects while excluding the peripheral follicle-cell belt."""
    image, roi = np.asarray(dapi), np.asarray(oocyte, bool)
    if image.shape != roi.shape:
        raise ValueError("DAPI and oocyte mask must have matching dimensions.")
    interior = ndi.binary_erosion(roi, iterations=max(1, edge_exclusion_px))
    if not interior.any():
        return np.zeros_like(roi)
    smooth = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1.3)
    values = smooth[interior]
    threshold = max(float(np.percentile(values, 82)), float(np.median(values) + 2 * np.std(values)))
    binary = (smooth > threshold) & interior
    binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    binary = remove_small_objects(binary, min_size=25)
    distance = ndi.distance_transform_edt(binary)
    maxima = (distance == ndi.maximum_filter(distance, size=9)) & binary
    markers, _ = ndi.label(maxima)
    if not markers.any():
        return np.zeros_like(roi)
    watershed_labels = ndi.watershed_ift(np.uint8(255 - np.clip(distance / max(distance.max(), 1) * 255, 0, 255)), markers.astype(np.int32))
    result = np.zeros_like(roi)
    for region in regionprops(watershed_labels):
        if 30 <= region.area <= 5000:
            result[watershed_labels == region.label] = True
    return result


def propose_border_cluster(decad: np.ndarray, chamber: np.ndarray, oocyte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the largest DEcad-positive candidate in a narrow inferred-interface band."""
    image, chamber, oocyte = np.asarray(decad), np.asarray(chamber, bool), np.asarray(oocyte, bool)
    if image.shape != chamber.shape or chamber.shape != oocyte.shape:
        raise ValueError("DEcad and masks must have matching dimensions.")
    band = chamber & ndi.binary_dilation(oocyte, iterations=18) & ~ndi.binary_erosion(oocyte, iterations=18)
    if not band.any():
        return np.zeros_like(chamber), band
    smooth = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1.2)
    threshold = max(float(np.percentile(smooth[band], 75)), float(np.percentile(smooth[chamber], 88)))
    candidate = closing((smooth > threshold) & band, disk(2))
    candidate = remove_small_objects(candidate, min_size=12)
    labels = label(candidate)
    if labels.max() == 0:
        return np.zeros_like(chamber), band
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(np.argmax(counts)), band


def candidate_metrics(chamber: np.ndarray, oocyte: np.ndarray, nuclei: np.ndarray, border: np.ndarray) -> dict:
    """Pixel-based proposal metrics; not calibrated physical measurements."""
    chamber_area, oocyte_area, border_area = int(chamber.sum()), int(oocyte.sum()), int(border.sum())
    return {
        "egg_chamber_area_px2": chamber_area,
        "oocyte_area_px2": oocyte_area,
        "oocyte_area_fraction": oocyte_area / chamber_area if chamber_area else 0.0,
        "oocyte_nuclei_candidate_count": int(label(nuclei).max()),
        "oocyte_nuclei_density_per_px2": int(label(nuclei).max()) / oocyte_area if oocyte_area else 0.0,
        "border_cluster_candidate_area_px2": border_area,
        "border_cluster_percent_chamber": 100 * border_area / chamber_area if chamber_area else 0.0,
    }
