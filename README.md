# Microscopy Image Analysis

A reproducible Python pipeline for Drosophila egg-chamber microscopy. It inventories TIFF acquisitions, performs baseline QC, and includes both annotation-guided segmentation and content-aware automated compartment proposals.

## Scientific safeguards

Automated masks are **candidate segmentations, not validated biological ground truth**. A segmentation algorithm can detect fluorescence structures but cannot guarantee their biological identity. Inspect masks and validate against manual Fiji/ImageJ annotations before reporting biological measurements.

RGB/pseudocoloured composites are not quantitative fluorescence sources. Use original single-channel acquisitions for intensity measurements. Without microscope calibration, lengths and areas remain in pixels and pixels².

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Baseline QC

```bash
microscopy-pipeline inventory /path/to/tiff_folder --output results/inventory.csv
microscopy-pipeline batch /path/to/tiff_folder --output results
```

These commands produce generic foreground masks and connected-component QC only.

## Content-aware automatic proposals

Module: `microscopy_pipeline.content_aware`

The first-pass algorithm uses:
1. AcTub cortical signal to propose egg-chamber envelopes using adaptive thresholding, morphological gap closing and convex envelopes.
2. The spatial DAPI intensity profile along the chamber's major axis to propose a nurse-cell/oocyte split at a low-signal valley.
3. DAPI object detection inside the proposed oocyte, with a 45-pixel peripheral exclusion to reduce follicle-cell nucleus contamination.
4. DEcad signal restricted to a narrow band around the inferred oocyte/chamber interface to propose a border-cell cluster.

Example:

```python
import cv2
from microscopy_pipeline.content_aware import (
    propose_egg_chambers, propose_oocyte, propose_oocyte_nuclei,
    propose_border_cluster, candidate_metrics,
)

actub = cv2.imread("sample_AcTub.tif", cv2.IMREAD_UNCHANGED)
dapi = cv2.imread("sample_DAPI.tif", cv2.IMREAD_UNCHANGED)
decad = cv2.imread("sample_DEcad.tif", cv2.IMREAD_UNCHANGED)

for chamber in propose_egg_chambers(actub):
    oocyte = propose_oocyte(chamber, dapi)
    nuclei = propose_oocyte_nuclei(dapi, oocyte)
    border, interface_band = propose_border_cluster(decad, chamber, oocyte)
    print(candidate_metrics(chamber, oocyte, nuclei, border))
```

### Limitations of this first pass

- Convex envelopes can overfill concavities or merge nearby structures; truncated chambers touching image edges need manual review.
- The DAPI-profile valley is a heuristic for the oocyte/nurse-cell interface, not a learned anatomical classifier.
- The peripheral exclusion distance is in pixels and must be adjusted for image scale and acquisition.
- A DEcad-positive connected component near the inferred interface is only a border-cluster candidate; it may include other membranes or fragmented signal.
- The nuclei count can be zero if no confident internal objects remain after the follicle-cell exclusion. Do not reinterpret that as biological absence without inspecting the source image.

The outputs produced in `active_contour_pass` are exploratory candidates for the supplied images. They are not a validated quantitative comparison between Control, Par-1 RNAi and Sgg RNAi.

## Annotation-guided workflow

For higher-confidence results, use `microscopy_pipeline.hierarchical_segmentation`. It accepts manually drawn egg-chamber, oocyte and border-search polygons, then performs constrained DAPI and DEcad detection.

## Validation plan

1. Review chamber and oocyte boundaries in each image.
2. Compare candidate nuclei against manual DAPI counts, distinguishing follicle cells from internal nuclei.
3. Compare the DEcad cluster proposal against manually identified border cells.
4. Quantify Dice/IoU, count error and area error on held-out images.
5. Only after validation, batch-process the experimental groups.

Keep raw microscopy data outside the repository unless project/data policy explicitly permits publishing it. Commit code and small synthetic tests, not unpublished raw images.
