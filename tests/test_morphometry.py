import numpy as np
from microscopy_pipeline.morphometry import measure_components, mask_area


def test_component_area_and_axes():
    mask = np.zeros((40, 50), dtype=bool)
    mask[5:15, 10:30] = True
    result = measure_components(mask)
    assert len(result) == 1
    assert result.iloc[0]["area_px2"] == 200
    assert result.iloc[0]["major_axis_px"] >= result.iloc[0]["minor_axis_px"]


def test_calibrated_area():
    mask = np.ones((10, 10), dtype=bool)
    result = mask_area(mask, pixel_size_um=0.5)
    assert result["foreground_area_px2"] == 100
    assert result["foreground_area_um2"] == 25.0
