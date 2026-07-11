import numpy as np

from training.clinical_measurements import compute_clinical_measurements


def _make_synthetic_mask(width: int = 128, height: int = 64) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.int32)
    # synthetic cornea band centered vertically
    y_center = height // 2
    cornea_height = max(2, height // 8)
    cornea_top = y_center - cornea_height // 2
    cornea_bottom = cornea_top + cornea_height
    mask[cornea_top:cornea_bottom, width // 8 : width - width // 8] = 1

    # synthetic iris region below cornea
    iris_top = cornea_bottom
    iris_bottom = min(height, iris_top + max(2, height // 4))
    mask[iris_top:iris_bottom, width // 4 : width - width // 4] = 2
    return mask


def test_compute_clinical_measurements_basic():
    mask = _make_synthetic_mask(width=120, height=80)
    result = compute_clinical_measurements(mask, pixel_spacing_mm=0.05)

    assert result.measurements["central_corneal_thickness_mm"] is not None
    assert result.measurements["corneal_diameter_mm"] is not None
    assert result.measurements["iris_area_mm2"] > 0.0
    assert result.confidences["central_corneal_thickness_mm"] > 0.0
    assert isinstance(result.to_dict(), dict)
    json_text = result.to_json()
    assert "central_corneal_thickness_mm" in json_text
