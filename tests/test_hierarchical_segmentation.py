import numpy as np

from microscopy_pipeline.hierarchical_segmentation import (
    CompartmentMasks,
    mask_metrics,
    polygon_mask,
    segment_dapi_nuclei,
)


def test_polygon_mask_uses_xy_coordinates():
    mask = polygon_mask((20, 20), [[2, 2], [10, 2], [10, 10], [2, 10]])
    assert mask[5, 5]
    assert not mask[15, 15]


def test_nuclei_are_constrained_to_oocyte_roi():
    image = np.zeros((100, 100), dtype=np.uint8)
    yy, xx = np.ogrid[:100, :100]
    image[((xx - 30) ** 2 + (yy - 30) ** 2) < 8 ** 2] = 220
    image[((xx - 75) ** 2 + (yy - 75) ** 2) < 8 ** 2] = 220
    roi = np.zeros_like(image, dtype=bool)
    roi[10:50, 10:50] = True
    detected = segment_dapi_nuclei(
        image, roi, min_area_px=30, max_area_px=1000,
        local_block_size=31, seed_min_distance=5,
    )
    assert detected.any()
    assert not detected[75, 75]
    assert np.all(~detected | roi)


def test_biological_density_metrics():
    chamber = np.zeros((20, 20), dtype=bool)
    chamber[2:18, 2:18] = True
    oocyte = np.zeros_like(chamber)
    oocyte[2:10, 2:10] = True
    nuclei = np.zeros_like(chamber)
    nuclei[4:6, 4:6] = True
    nuclei[7:9, 7:9] = True
    cluster = np.zeros_like(chamber)
    cluster[12:14, 12:16] = True
    metrics = mask_metrics(CompartmentMasks(chamber, oocyte, nuclei, cluster))
    assert metrics["oocyte_nuclei_count"] == 2
    assert metrics["oocyte_nuclei_density_per_px2"] == 2 / 64
    assert metrics["border_cluster_area_px2"] == 8
    assert metrics["border_cluster_percent_of_chamber"] == 100 * 8 / 256
