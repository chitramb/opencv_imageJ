import numpy as np

from microscopy_pipeline.content_aware import (
    candidate_metrics,
    propose_egg_chambers,
    propose_oocyte_nuclei,
)


def test_empty_nuclear_roi_returns_empty_mask():
    image = np.zeros((80, 80), dtype=np.uint8)
    roi = np.zeros((80, 80), dtype=bool)
    assert not propose_oocyte_nuclei(image, roi).any()


def test_candidate_metrics_use_agreed_definitions():
    chamber = np.zeros((40, 40), bool)
    chamber[2:38, 2:38] = True
    oocyte = np.zeros_like(chamber)
    oocyte[10:30, 10:30] = True
    nuclei = np.zeros_like(chamber)
    nuclei[15:17, 15:17] = True
    nuclei[20:22, 20:22] = True
    border = np.zeros_like(chamber)
    border[5:8, 5:9] = True
    result = candidate_metrics(chamber, oocyte, nuclei, border)
    assert result["oocyte_nuclei_candidate_count"] == 2
    assert result["border_cluster_percent_chamber"] == 100 * 12 / 1296
