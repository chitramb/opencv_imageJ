# Microscopy Image Analysis

A reproducible Python starter pipeline for the supplied Drosophila egg-chamber TIFF images. It preserves source pixel data, inventories TIFF metadata, separates grayscale acquisitions from RGB display composites, creates reviewable segmentation masks, and exports morphometry and fluorescence summaries.

## Important scientific scope

This first implementation is a **baseline workflow**, not a validated biological segmentation model. Otsu/adaptive thresholding and connected components are generic image-processing tools; they cannot reliably infer the oocyte boundary, egg-chamber boundary, or individual border cells in every preparation without user review or annotated training data. All automatically generated masks must be inspected before treating measurements as biological results.

RGB/pseudocoloured composites are marked non-quantitative by default. Their RGB values are display colours and must not be treated as original fluorescence intensities. Quantitative fluorescence analysis should use the original grayscale channel acquisitions with matched exposure/gain settings and background correction.

## Install

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pip install -e .
```

## Inventory images

```bash
microscopy-pipeline inventory /path/to/tiff_folder --output results/inventory.csv
```

## Analyse a folder

```bash
microscopy-pipeline batch /path/to/tiff_folder --output results --pixel-size-um 0.25
```

Omit `--pixel-size-um` if calibration is unknown; measurements will remain in pixels and pixels². The example value above is illustrative only and must be replaced with the microscope calibration.

The batch command writes:
- `inventory.csv`: file dimensions, dtype, channels, range and TIFF metadata.
- `measurements.csv`: preliminary image/object measurements.
- `overlays/`: review images showing detected components.
- `masks/`: binary masks for the generic foreground segmentation.

## Single image

```bash
microscopy-pipeline analyse sample.tif --output results/sample --pixel-size-um 0.25
```

## Current analysis behaviour

- Reads TIFF pixel arrays with `tifffile`; never overwrites or rescales source files.
- Records dtype, shape, finite intensity range and TIFF page metadata.
- Treats single-channel grayscale images as candidate quantitative acquisitions.
- Treats RGB/RGBA images as display/composite images and excludes them from quantitative fluorescence calculations by default.
- Creates a conservative Otsu foreground mask and connected-component measurements for QC and exploration.
- Reports area, major/minor axis, equivalent diameter, centroid and bounding box.
- Does **not** automatically assign a component as oocyte, nurse-cell compartment, egg chamber or border-cell cluster.

## Next validation steps

1. Confirm pixel-size calibration and acquisition metadata.
2. Annotate representative egg-chamber and oocyte boundaries.
3. Agree on an operational definition of border-cell cluster and density.
4. Validate masks against manual ImageJ/Fiji measurements.
5. Add channel-specific fluorescence ROIs and background regions.
6. Only then lock parameters and compare Control, Par-1 RNAi and Sgg RNAi groups.

## Data handling

Do not commit identifiable, unpublished or sensitive microscopy data to a public repository unless permitted by the relevant project/data policy. Keep raw data outside the repository; commit code and small synthetic test fixtures only.
