"""Clinical measurement engine for AS-OCT segmentation masks.

This module computes clinically meaningful biometric measurements from a
segmentation mask without modifying segmentation internals.

Supported outputs:
- JSON
- CSV
- Python dictionary

Supported measurements:
- Central Corneal Thickness (CCT)
- Corneal Thickness Profile
- Anterior Chamber Depth (ACD)
- Corneal Diameter
- Corneal Radius
- Corneal Curvature
- Iris Area
- Iris Thickness
- White-to-White (WTW)
- Pupil Diameter (future-ready)

All measured values are returned with an estimated confidence score in the
range [0.0, 1.0].
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np


Pixels = Union[np.ndarray, List[List[int]], List[int]]


@dataclass(frozen=True)
class ClinicalMeasurementConfig:
    """Configuration for clinical measurement extraction."""

    pixel_spacing_mm: Union[float, Tuple[float, float]] = 1.0
    class_ids: Dict[str, int] = field(default_factory=lambda: {
        "background": 0,
        "cornea": 1,
        "iris": 2,
        "pupil": 3,
    })

    def get_pixel_spacing(self) -> Tuple[float, float]:
        """Return (horizontal, vertical) pixel spacing in millimeters."""
        if isinstance(self.pixel_spacing_mm, tuple):
            if len(self.pixel_spacing_mm) != 2:
                raise ValueError("pixel_spacing_mm tuple must have length 2")
            return float(self.pixel_spacing_mm[0]), float(self.pixel_spacing_mm[1])
        return float(self.pixel_spacing_mm), float(self.pixel_spacing_mm)


@dataclass
class ClinicalMeasurementResult:
    measurements: Dict[str, Optional[float]]
    confidences: Dict[str, float]
    thickness_profile: List[Dict[str, float]]
    calibration: Dict[str, float]
    notes: List[str] = field(default_factory=list)
    raw_pixel_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "measurements": self.measurements,
            "confidences": self.confidences,
            "thickness_profile": self.thickness_profile,
            "calibration": self.calibration,
            "notes": self.notes,
            "raw_pixel_summary": self.raw_pixel_summary,
        }

    def to_json(self, path: Optional[Union[str, Path]] = None) -> str:
        payload = self.to_dict()
        json_text = json.dumps(payload, indent=2)
        if path is not None:
            out_path = Path(path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json_text, encoding="utf-8")
        return json_text

    def to_csv(self, path: Union[str, Path]) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        rows: List[Dict[str, Any]] = []
        for name, value in self.measurements.items():
            rows.append(
                {
                    "metric": name,
                    "value": value if value is None else float(value),
                    "confidence": float(self.confidences.get(name, 0.0)),
                }
            )
        rows.append(
            {
                "metric": "corneal_thickness_profile",
                "value": json.dumps(self.thickness_profile),
                "confidence": float(self.confidences.get("corneal_thickness_profile", 0.0)),
            }
        )
        rows.append(
            {
                "metric": "notes",
                "value": json.dumps(self.notes),
                "confidence": 1.0,
            }
        )

        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["metric", "value", "confidence"])
            writer.writeheader()
            writer.writerows(rows)


@dataclass(frozen=True)
class _VerticalBoundary:
    x: int
    top: int
    bottom: int


def _to_numpy(mask: Pixels) -> np.ndarray:
    if isinstance(mask, np.ndarray):
        array = mask
    else:
        array = np.asarray(mask)

    if array.ndim != 2:
        raise ValueError("Segmentation mask must be a 2D array")
    return array


def _binary_mask(mask: np.ndarray, class_id: int) -> np.ndarray:
    return mask == class_id


def _find_vertical_bounds(binary_mask: np.ndarray) -> List[_VerticalBoundary]:
    height, width = binary_mask.shape
    bounds: List[_VerticalBoundary] = []
    for x in range(width):
        column = binary_mask[:, x]
        indices = np.flatnonzero(column)
        if indices.size == 0:
            continue
        bounds.append(_VerticalBoundary(x=x, top=int(indices[0]), bottom=int(indices[-1])))
    return bounds


def _nearest_boundary(bounds: List[_VerticalBoundary], target_x: int) -> Optional[_VerticalBoundary]:
    if not bounds:
        return None
    nearest = min(bounds, key=lambda bound: abs(bound.x - target_x))
    return nearest


def _compute_thickness_profile(
    bounds: List[_VerticalBoundary],
    mm_y: float,
) -> List[Dict[str, float]]:
    profile: List[Dict[str, float]] = []
    for bound in bounds:
        thickness_mm = float((bound.bottom - bound.top + 1) * mm_y)
        profile.append({"x_mm": float(bound.x * mm_y), "thickness_mm": thickness_mm})
    return profile


def _fit_circle_radius(
    bounds: List[_VerticalBoundary],
    mm_x: float,
    mm_y: float,
) -> Optional[float]:
    if len(bounds) < 6:
        return None

    xs = np.array([float(bound.x) * mm_x for bound in bounds], dtype=np.float64)
    ys = np.array([float(bound.top) * mm_y for bound in bounds], dtype=np.float64)

    if xs.ptp() <= 0 or ys.ptp() <= 0:
        return None

    center = np.array([xs.mean(), ys.mean()])
    u = xs - center[0]
    v = ys - center[1]
    w = u * u + v * v
    A = np.column_stack([u, v, np.ones_like(u)])

    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, w, rcond=None)
    except np.linalg.LinAlgError:
        return None

    a, b, c = coeffs
    center_x = a / 2.0 + center[0]
    center_y = b / 2.0 + center[1]
    radius_sq = center_x**2 + center_y**2 + c
    if radius_sq <= 0.0:
        return None
    return float(np.sqrt(radius_sq))


def _confidence_for_center_measurement(
    found_at_center: bool,
    num_columns: int,
) -> float:
    if num_columns <= 0:
        return 0.0
    confidence = 0.9 if found_at_center else 0.7
    if num_columns < 20:
        confidence *= 0.8
    return float(max(0.0, min(1.0, confidence)))


def _confidence_for_width_measurement(num_columns: int) -> float:
    if num_columns <= 0:
        return 0.0
    if num_columns < 10:
        return 0.5
    if num_columns < 30:
        return 0.75
    return 0.9


def _confidence_for_radius(radius_mm: Optional[float], num_points: int) -> float:
    if radius_mm is None or num_points < 6:
        return 0.0
    if 5.0 <= radius_mm <= 9.0:
        base = 0.95
    elif 3.0 <= radius_mm <= 12.0:
        base = 0.75
    else:
        base = 0.45
    if num_points < 15:
        base *= 0.8
    return float(max(0.0, min(1.0, base)))


def _compute_measurements(
    mask: np.ndarray,
    config: ClinicalMeasurementConfig,
    pupil_mask: Optional[np.ndarray] = None,
) -> ClinicalMeasurementResult:
    mm_x, mm_y = config.get_pixel_spacing()
    mm_per_pixel_area = float(mm_x * mm_y)

    cornea_mask = _binary_mask(mask, config.class_ids["cornea"])
    iris_mask = _binary_mask(mask, config.class_ids["iris"])
    pupil_mask_arr = None
    if pupil_mask is not None:
        pupil_mask_arr = _binary_mask(_to_numpy(pupil_mask), 1)
    elif config.class_ids.get("pupil") is not None:
        pupil_mask_arr = _binary_mask(mask, config.class_ids["pupil"])

    cornea_bounds = _find_vertical_bounds(cornea_mask)
    iris_bounds = _find_vertical_bounds(iris_mask)

    height, width = mask.shape
    center_x = width // 2

    cornea_center_bound = _nearest_boundary(cornea_bounds, center_x)
    found_center_cornea = cornea_center_bound is not None and cornea_center_bound.x == center_x
    central_corneal_thickness_mm = (
        float((cornea_center_bound.bottom - cornea_center_bound.top + 1) * mm_y)
        if cornea_center_bound is not None
        else None
    )
    central_corneal_thickness_confidence = _confidence_for_center_measurement(
        found_center_cornea, len(cornea_bounds)
    )

    thickness_profile = _compute_thickness_profile(cornea_bounds, mm_y)
    thickness_profile_confidence = float(
        max(0.0, min(1.0, 0.4 + len(thickness_profile) * 0.01))
    )

    if cornea_center_bound is not None and iris_bounds:
        iris_center_bound = _nearest_boundary(iris_bounds, cornea_center_bound.x)
        if iris_center_bound is not None:
            acd_pixel = iris_center_bound.top - cornea_center_bound.bottom
            if acd_pixel > 0:
                anterior_chamber_depth_mm = float(acd_pixel * mm_y)
            else:
                anterior_chamber_depth_mm = None
        else:
            anterior_chamber_depth_mm = None
    else:
        anterior_chamber_depth_mm = None
    anterior_chamber_depth_confidence = 0.0
    if anterior_chamber_depth_mm is not None:
        anterior_chamber_depth_confidence = float(
            0.95 if cornea_center_bound is not None and iris_center_bound is not None else 0.7
        )

    cornea_columns = [bound.x for bound in cornea_bounds]
    corneal_diameter_mm = (
        float((max(cornea_columns) - min(cornea_columns) + 1) * mm_x)
        if cornea_columns
        else None
    )
    corneal_diameter_confidence = _confidence_for_width_measurement(len(cornea_columns))

    corneal_radius_mm = _fit_circle_radius(cornea_bounds, mm_x, mm_y)
    corneal_radius_confidence = _confidence_for_radius(corneal_radius_mm, len(cornea_bounds))
    corneal_curvature_diopters = (
        float(337.5 / corneal_radius_mm) if corneal_radius_mm and corneal_radius_mm > 0.0 else None
    )
    corneal_curvature_dim_inverse = (
        float(1.0 / corneal_radius_mm) if corneal_radius_mm and corneal_radius_mm > 0.0 else None
    )

    iris_area_mm2 = float(np.count_nonzero(iris_mask) * mm_per_pixel_area)
    iris_area_confidence = 0.9 if np.count_nonzero(iris_mask) > 0 else 0.0

    iris_center_bound = _nearest_boundary(iris_bounds, center_x)
    iris_thickness_mm = (
        float((iris_center_bound.bottom - iris_center_bound.top + 1) * mm_y)
        if iris_center_bound is not None
        else None
    )
    iris_thickness_confidence = _confidence_for_center_measurement(
        iris_center_bound is not None and iris_center_bound.x == center_x,
        len(iris_bounds),
    )

    white_to_white_mm = None
    white_to_white_confidence = 0.0
    if iris_bounds:
        iris_cols = [bound.x for bound in iris_bounds]
        white_to_white_mm = float((max(iris_cols) - min(iris_cols) + 1) * mm_x)
        white_to_white_confidence = _confidence_for_width_measurement(len(iris_cols))
    elif cornea_columns:
        white_to_white_mm = corneal_diameter_mm
        white_to_white_confidence = float(corneal_diameter_confidence * 0.6)

    pupil_diameter_mm = None
    pupil_diameter_confidence = 0.0
    if pupil_mask_arr is not None and np.count_nonzero(pupil_mask_arr) > 0:
        pupil_bounds = _find_vertical_bounds(pupil_mask_arr)
        pupil_center_bound = _nearest_boundary(pupil_bounds, center_x)
        if pupil_center_bound is not None:
            pupil_diameter_mm = float((pupil_center_bound.bottom - pupil_center_bound.top + 1) * mm_y)
            pupil_diameter_confidence = _confidence_for_center_measurement(
                pupil_center_bound.x == center_x,
                len(pupil_bounds),
            )
        else:
            pupil_diameter_mm = None
            pupil_diameter_confidence = 0.0

    notes: List[str] = []
    if np.count_nonzero(cornea_mask) == 0:
        notes.append("Cornea segmentation is absent or empty.")
    if np.count_nonzero(iris_mask) == 0:
        notes.append("Iris segmentation is absent or empty.")
    if pupil_mask is None and config.class_ids.get("pupil") is None:
        notes.append(
            "Pupil diameter calculation is future-ready: provide a pupil mask or map class 'pupil'."
        )

    measurements = {
        "central_corneal_thickness_mm": central_corneal_thickness_mm,
        "anterior_chamber_depth_mm": anterior_chamber_depth_mm,
        "corneal_diameter_mm": corneal_diameter_mm,
        "corneal_radius_mm": corneal_radius_mm,
        "corneal_curvature_diopters": corneal_curvature_diopters,
        "corneal_curvature_mm_inv": corneal_curvature_dim_inverse,
        "iris_area_mm2": iris_area_mm2,
        "iris_thickness_mm": iris_thickness_mm,
        "white_to_white_mm": white_to_white_mm,
        "pupil_diameter_mm": pupil_diameter_mm,
    }

    confidences = {
        "central_corneal_thickness_mm": central_corneal_thickness_confidence,
        "corneal_thickness_profile": thickness_profile_confidence,
        "anterior_chamber_depth_mm": anterior_chamber_depth_confidence,
        "corneal_diameter_mm": corneal_diameter_confidence,
        "corneal_radius_mm": corneal_radius_confidence,
        "corneal_curvature_diopters": corneal_radius_confidence,
        "corneal_curvature_mm_inv": corneal_radius_confidence,
        "iris_area_mm2": iris_area_confidence,
        "iris_thickness_mm": iris_thickness_confidence,
        "white_to_white_mm": white_to_white_confidence,
        "pupil_diameter_mm": pupil_diameter_confidence,
    }

    raw_pixel_summary = {
        "cornea_pixel_count": int(np.count_nonzero(cornea_mask)),
        "iris_pixel_count": int(np.count_nonzero(iris_mask)),
        "pupil_pixel_count": int(np.count_nonzero(pupil_mask_arr)) if pupil_mask_arr is not None else 0,
        "cornea_columns": len(cornea_bounds),
        "iris_columns": len(iris_bounds),
    }

    calibration = {
        "pixel_spacing_x_mm": mm_x,
        "pixel_spacing_y_mm": mm_y,
        "pixel_area_mm2": mm_per_pixel_area,
    }

    return ClinicalMeasurementResult(
        measurements=measurements,
        confidences=confidences,
        thickness_profile=thickness_profile,
        calibration=calibration,
        notes=notes,
        raw_pixel_summary=raw_pixel_summary,
    )


def compute_clinical_measurements(
    mask: Pixels,
    *,
    pixel_spacing_mm: Union[float, Tuple[float, float]] = 1.0,
    class_ids: Optional[Dict[str, int]] = None,
    pupil_mask: Optional[Pixels] = None,
) -> ClinicalMeasurementResult:
    """Compute clinical biometric measurements from a segmentation mask.

    Parameters:
        mask: 2D segmentation mask containing at least cornea and iris classes.
        pixel_spacing_mm: millimeters per pixel in x/y direction. A single float
            applies isotropically; a tuple allows distinct horizontal and vertical
            resolutions.
        class_ids: Optional mapping from semantic class names to integer labels.
            Defaults to {background:0, cornea:1, iris:2, pupil:3}.
        pupil_mask: Optional standalone pupil mask for future-ready pupil diameter
            computation.

    Returns:
        ClinicalMeasurementResult with measurements, confidences, calibration,
        and optional notes.
    """
    if class_ids is None:
        class_ids = {
            "background": 0,
            "cornea": 1,
            "iris": 2,
            "pupil": 3,
        }
    config = ClinicalMeasurementConfig(
        pixel_spacing_mm=pixel_spacing_mm,
        class_ids=class_ids,
    )
    mask_arr = _to_numpy(mask)
    pupil_arr = _to_numpy(pupil_mask) if pupil_mask is not None else None
    return _compute_measurements(mask_arr, config, pupil_arr)


__all__ = [
    "ClinicalMeasurementConfig",
    "ClinicalMeasurementResult",
    "compute_clinical_measurements",
]
