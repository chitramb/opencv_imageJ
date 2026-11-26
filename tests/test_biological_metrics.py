import numpy as np
from microscopy_pipeline.biological_metrics import (
    oocyte_count_density, border_cluster_area_fraction,
)


def test_oocyte_count_density_calibrated():
    mask = np.ones((10, 10), dtype=bool)
    result = oocyte_count_density(mask, count_objects=20, pixel_size_um=0.5)
    assert result["oocyte_area_um2"] == 25.0
    assert result["oocyte_count_density_per_um2"] == 0.8


def test_border_cluster_area_fraction():
    chamber = np.zeros((20, 20), dtype=bool)
    chamber[2:18, 2:18] = True
    cluster = np.zeros_like(chamber)
    cluster[3:7, 3:7] = True
    result = border_cluster_area_fraction(cluster, chamber)
    assert result["border_cluster_area_px2"] == 16
    assert result["egg_chamber_area_px2"] == 256
    assert result["border_cluster_percent_of_chamber"] == 6.25
