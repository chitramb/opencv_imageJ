# Microscopy Image Analysis

A reproducible Python pipeline for Drosophila egg-chamber microscopy. It inventories TIFF acquisitions, performs baseline QC, and now includes an annotation-guided hierarchical segmentation workflow for biological compartments.

## Scientific scope and safeguards

The hierarchical workflow is deliberately human-in-the-loop. Egg-chamber, oocyte and border-cell search-region polygons are annotated by a user; DAPI nuclei and DE-cadherin-positive structures are then detected only inside the relevant polygon. This prevents a generic threshold from silently assigning biological identities.

The automated nuclei and border-cluster masks are candidates, not validated biological truth. Inspect overlays and compare against manual Fiji/ImageJ annotations before reporting counts or areas. The border-cluster algorithm returns the largest connected candidate within a tightly drawn search ROI; it does not identify individual border cells.

RGB/pseudocoloured composites are not treated as quantitative fluorescence sources. Use original single-channel acquisitions for intensity measurements.

## Install

Python 3.10+ is recommended.

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

These commands produce generic foreground masks and connected-component QC only. They do not infer egg-chamber or oocyte identities.

## Hierarchical compartment segmentation

### 1. Annotate the biological regions

For each image, create a JSON file with three polygons. Coordinates are pixel coordinates in `[x, y]` order:

```json
{
  "egg_chamber": [[120, 210], [180, 140], [420, 150], [610, 310], [590, 560], [300, 610], [130, 430]],
  "oocyte": [[390, 250], [450, 220], [570, 330], [560, 500], [430, 520], [370, 390]],
  "border_search": [[340, 300], [420, 280], [450, 390], [390, 430]]
}
```

The coordinates above are illustrative placeholders, **not annotations for the supplied images**. Draw the actual polygons in Fiji/ImageJ or another annotation tool. Use one egg chamber per annotation file. The oocyte and border-search polygons are clipped to the egg-chamber mask.

### 2. Run the segmentation functions

```python
import cv2
from microscopy_pipeline.hierarchical_segmentation import (
    build_compartment_masks, mask_metrics, save_segmentation, load_annotations
)

dapi = cv2.imread("sample_DAPI.tif", cv2.IMREAD_UNCHANGED)
decad = cv2.imread("sample_DEcad.tif", cv2.IMREAD_UNCHANGED)
actub = cv2.imread("sample_AcTub.tif", cv2.IMREAD_UNCHANGED)

annotations = load_annotations("sample_annotations.json")
masks = build_compartment_masks(
    dapi.shape, annotations, dapi, decad,
    nuclei_params={"min_area_px": 40, "max_area_px": 6000},
    border_params={"min_area_px": 20},
)
metrics = mask_metrics(masks)
save_segmentation("results/sample", actub, masks, metrics)
```

Use matching fields of view and verify that the DAPI, DEcad and overlay images are registered. The example parameters are starting values and require tuning against manual annotations. If the files are not registered, register them before applying masks across channels.

### 3. Outputs and metric definitions

- Egg chamber: polygon area and major/minor axis length.
- Oocyte: polygon area and major/minor axis length.
- Oocyte nuclei: DAPI-positive object candidates within the oocyte; density = detected object count / oocyte area.
- Border-cell cluster: DE-cadherin-positive candidate area within the border-search ROI; density = cluster area / egg-chamber area × 100%.

Without pixel calibration, results are in pixels and pixels². Do not convert to micrometres without the acquisition calibration.

## Validation plan

1. Annotate egg chamber, oocyte and border-cell search region for representative images from each condition.
2. Have the biological boundary definitions checked by the experimenter.
3. Compare automatic DAPI nuclei detections against manual counts; tune size and watershed parameters.
4. Compare the border-cluster mask against a manual DEcad mask. A single connected-component heuristic may fail when signal is fragmented or other membranes enter the search ROI.
5. Quantify boundary overlap (e.g. Dice/IoU), count error and area error on held-out images.
6. Only after validation, batch-process Control, Par-1 RNAi and Sgg RNAi images.

## Data handling

Keep raw microscopy data outside the repository unless project/data policy explicitly permits publishing it. Commit code and small synthetic tests, not unpublished raw images.
