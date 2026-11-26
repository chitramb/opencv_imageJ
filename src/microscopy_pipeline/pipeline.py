"""Orchestration for baseline TIFF analysis."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from .io import discover_tiffs, load_tiff, inventory
from .preprocessing import otsu_mask
from .morphometry import measure_components, mask_area
from .visualization import save_overlay


def analyse_one(path: str | Path, output_dir: str | Path,
                pixel_size_um: float | None = None) -> list[dict]:
    path, out = Path(path), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    image, meta = load_tiff(path)
    # Composite RGB images are for display/QC only; do not infer original channels.
    if image.ndim != 2:
        return [{
            "file": path.name, "analysis_status": "skipped_rgb_composite",
            "reason": "RGB/RGBA composite is not used for quantitative fluorescence or automatic segmentation",
            "dtype": str(image.dtype), "shape": str(image.shape),
        }]
    mask = otsu_mask(image)
    mask_path = out / "masks" / f"{path.stem}_foreground.tif"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    import tifffile
    tifffile.imwrite(mask_path, mask.astype(np.uint8))
    overlay_path = out / "overlays" / f"{path.stem}_overlay.png"
    save_overlay(image, mask, overlay_path)
    components = measure_components(mask, pixel_size_um)
    rows = []
    for _, component in components.iterrows():
        row = {"file": path.name, "analysis_status": "qc_foreground_components",
               "is_biologically_validated": False, **meta, **component.to_dict()}
        rows.append(row)
    if not rows:
        rows.append({"file": path.name, "analysis_status": "no_foreground_detected",
                     "is_biologically_validated": False, **meta, **mask_area(mask, pixel_size_um)})
    return rows


def analyse_folder(folder: str | Path, output_dir: str | Path,
                   pixel_size_um: float | None = None) -> pd.DataFrame:
    folder, out = Path(folder), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inventory(folder)).to_csv(out / "inventory.csv", index=False)
    rows = []
    for path in discover_tiffs(folder):
        try:
            rows.extend(analyse_one(path, out, pixel_size_um))
        except Exception as exc:
            rows.append({"file": path.name, "analysis_status": "error", "error": str(exc)})
    result = pd.DataFrame(rows)
    result.to_csv(out / "measurements.csv", index=False)
    return result
