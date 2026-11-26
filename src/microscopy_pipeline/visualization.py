"""QC overlay rendering."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from skimage.segmentation import find_boundaries

from .preprocessing import display_uint8


def save_overlay(image: np.ndarray, mask: np.ndarray, output: str | Path) -> None:
    if image.ndim == 2:
        base = cv2.cvtColor(display_uint8(image), cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[-1] >= 3:
        # RGB display only: OpenCV writes BGR.
        base = cv2.cvtColor(np.asarray(image[..., :3], dtype=np.uint8), cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(f"Unsupported image shape for overlay: {image.shape}")
    boundaries = find_boundaries(mask, mode="outer")
    base[boundaries] = (0, 255, 255)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), base)
