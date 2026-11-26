"""TIFF loading and metadata inventory. Source pixel arrays are never modified."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import tifffile


def load_tiff(path: str | Path) -> tuple[np.ndarray, dict]:
    path = Path(path)
    with tifffile.TiffFile(path) as tif:
        array = tif.asarray()
        page = tif.pages[0]
        metadata = {
            "file": path.name,
            "shape": "x".join(map(str, array.shape)),
            "dtype": str(array.dtype),
            "axes": getattr(tif.series[0], "axes", "") if tif.series else "",
            "pages": len(tif.pages),
            "photometric": str(page.photometric),
            "compression": str(page.compression),
            "description": (page.description or "")[:1000],
            "is_rgb": bool(array.ndim >= 3 and array.shape[-1] in (3, 4)),
            "is_quantitative_candidate": bool(array.ndim == 2),
        }
    finite = np.asarray(array)[np.isfinite(array)] if np.issubdtype(array.dtype, np.number) else np.array([])
    metadata["min"] = float(finite.min()) if finite.size else None
    metadata["max"] = float(finite.max()) if finite.size else None
    metadata["mean"] = float(finite.mean()) if finite.size else None
    metadata["std"] = float(finite.std()) if finite.size else None
    return array, metadata


def discover_tiffs(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".tif", ".tiff"})


def inventory(folder: str | Path) -> list[dict]:
    records = []
    for path in discover_tiffs(folder):
        try:
            _, record = load_tiff(path)
            record["relative_path"] = str(path.relative_to(folder))
            record["error"] = ""
        except Exception as exc:  # retain a record so one bad file doesn't stop a batch
            record = {"file": path.name, "relative_path": str(path), "error": str(exc)}
        records.append(record)
    return records
