"""Hospital-grade OCT report generator.

This module produces patient-facing PDF reports with:
- Patient information
- Original OCT image
- Predicted semantic mask
- Overlay image
- Measurements table
- AI confidence metrics
- Model version and timestamp
- Summary and doctor notes

The report is generated as a self-contained PDF using Matplotlib's PDF
backend and PIL image helpers.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from PIL import Image
import numpy as np


@dataclass(frozen=True)
class PatientInfo:
    patient_id: str
    name: str
    date_of_birth: Optional[str] = None
    sex: Optional[str] = None
    study_id: Optional[str] = None
    exam_date: Optional[str] = None
    referring_physician: Optional[str] = None
    operator: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True)
class ReportLayoutConfig:
    hospital_name: str = "MedicalAI Eye Center"
    institution_name: str = "Hospital Imaging Department"
    author: str = "MedicalAI Clinical Reporting"
    page_size: Tuple[float, float] = (11.0, 8.5)
    title_fontsize: int = 24
    heading_fontsize: int = 16
    text_fontsize: int = 10
    table_fontsize: int = 9
    label_colors: Dict[int, Tuple[int, int, int]] = field(default_factory=lambda: {
        0: (0, 0, 0),
        1: (0, 102, 204),
        2: (0, 153, 0),
        3: (255, 153, 0),
    })
    class_names: Dict[int, str] = field(default_factory=lambda: {
        0: "Background",
        1: "Cornea",
        2: "Iris",
        3: "Pupil",
    })


@dataclass
class ReportPayload:
    patient_info: PatientInfo
    original_image: np.ndarray
    predicted_mask: np.ndarray
    overlay_image: np.ndarray
    measurements: Dict[str, Optional[float]]
    confidences: Dict[str, float]
    model_version: str
    timestamp: str
    summary: str
    doctor_notes: str
    report_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "patient_info": self.patient_info.to_dict(),
            "measurements": self.measurements,
            "confidences": self.confidences,
            "model_version": self.model_version,
            "timestamp": self.timestamp,
            "summary": self.summary,
            "doctor_notes": self.doctor_notes,
            "report_id": self.report_id,
            "extra": self.extra,
        }
        return payload

    def save_json(self, path: Union[str, Path]) -> str:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        json_text = json.dumps(payload, indent=2)
        out_path.write_text(json_text, encoding="utf-8")
        return json_text


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    rgb = _ensure_rgb(image)
    if rgb.shape[2] == 3:
        return rgb[..., ::-1]
    return rgb


def _create_class_colormap(layout: ReportLayoutConfig) -> ListedColormap:
    colors = [tuple(c / 255.0 for c in layout.label_colors.get(idx, (0, 0, 0))) for idx in sorted(layout.label_colors)]
    return ListedColormap(colors)


def _render_title_page(pdf: PdfPages, payload: ReportPayload, layout: ReportLayoutConfig) -> None:
    fig = plt.figure(figsize=layout.page_size)
    fig.patch.set_facecolor("white")
    fig.text(0.05, 0.92, layout.hospital_name, fontsize=layout.title_fontsize, weight="bold")
    fig.text(0.05, 0.88, layout.institution_name, fontsize=layout.heading_fontsize)
    fig.text(0.05, 0.82, "OCT Clinical Report", fontsize=layout.heading_fontsize + 2, weight="bold")

    fig.text(0.05, 0.75, f"Patient Name: {payload.patient_info.name}", fontsize=layout.text_fontsize)
    fig.text(0.05, 0.73, f"Patient ID: {payload.patient_info.patient_id}", fontsize=layout.text_fontsize)
    if payload.patient_info.date_of_birth:
        fig.text(0.05, 0.71, f"Date of Birth: {payload.patient_info.date_of_birth}", fontsize=layout.text_fontsize)
    if payload.patient_info.sex:
        fig.text(0.05, 0.69, f"Sex: {payload.patient_info.sex}", fontsize=layout.text_fontsize)
    if payload.patient_info.study_id:
        fig.text(0.05, 0.67, f"Study ID: {payload.patient_info.study_id}", fontsize=layout.text_fontsize)
    if payload.patient_info.exam_date:
        fig.text(0.05, 0.65, f"Exam Date: {payload.patient_info.exam_date}", fontsize=layout.text_fontsize)
    if payload.patient_info.referring_physician:
        fig.text(0.05, 0.63, f"Referring Physician: {payload.patient_info.referring_physician}", fontsize=layout.text_fontsize)
    if payload.patient_info.operator:
        fig.text(0.05, 0.61, f"Operator: {payload.patient_info.operator}", fontsize=layout.text_fontsize)

    fig.text(0.05, 0.55, f"Model Version: {payload.model_version}", fontsize=layout.text_fontsize)
    fig.text(0.05, 0.53, f"Report Timestamp: {payload.timestamp}", fontsize=layout.text_fontsize)
    if payload.report_id:
        fig.text(0.05, 0.51, f"Report ID: {payload.report_id}", fontsize=layout.text_fontsize)

    fig.text(0.05, 0.45, "Summary:", fontsize=layout.heading_fontsize)
    fig.text(0.05, 0.42, payload.summary or "No summary provided.", fontsize=layout.text_fontsize, wrap=True)

    fig.text(0.05, 0.30, "Doctor Notes:", fontsize=layout.heading_fontsize)
    notes = payload.doctor_notes or "No doctor notes provided."
    fig.text(0.05, 0.27, notes, fontsize=layout.text_fontsize, wrap=True)

    fig.text(0.05, 0.12, "Generated by MedicalAI Clinical Reporting Engine.", fontsize=layout.text_fontsize - 1, color="#666666")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _render_image_page(pdf: PdfPages, payload: ReportPayload, layout: ReportLayoutConfig) -> None:
    fig, axes = plt.subplots(2, 2, figsize=layout.page_size)
    fig.patch.set_facecolor("white")
    axes = axes.flatten()

    original = _bgr_to_rgb(payload.original_image)
    axes[0].imshow(original)
    axes[0].set_title("Original OCT", fontsize=layout.heading_fontsize)
    axes[0].axis("off")

    mask_cmap = _create_class_colormap(layout)
    axes[1].imshow(payload.predicted_mask, cmap=mask_cmap, vmin=0, vmax=len(layout.label_colors) - 1)
    axes[1].set_title("Predicted Semantic Mask", fontsize=layout.heading_fontsize)
    axes[1].axis("off")

    overlay = _bgr_to_rgb(payload.overlay_image)
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=layout.heading_fontsize)
    axes[2].axis("off")

    legend_rows = [f"{layout.class_names[k]}" for k in sorted(layout.label_colors.keys())]
    legend_colors = [np.array(layout.label_colors[k], dtype=np.uint8) / 255.0 for k in sorted(layout.label_colors.keys())]
    axes[3].axis("off")
    axes[3].text(0.0, 1.0, "Class Legend", fontsize=layout.heading_fontsize)
    for idx, (label, color) in enumerate(zip(legend_rows, legend_colors), start=1):
        axes[3].text(0.05, 1.0 - idx * 0.18, "■ ", color=color, fontsize=18, transform=axes[3].transAxes)
        axes[3].text(0.10, 1.0 - idx * 0.18, label, fontsize=layout.text_fontsize, transform=axes[3].transAxes)

    fig.suptitle("OCT Imaging Overview", fontsize=layout.title_fontsize)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _render_measurements_page(pdf: PdfPages, payload: ReportPayload, layout: ReportLayoutConfig) -> None:
    fig = plt.figure(figsize=layout.page_size)
    fig.patch.set_facecolor("white")
    fig.suptitle("Clinical Measurements", fontsize=layout.title_fontsize)

    measurement_items = [
        (label.replace("_", " ").title(), value)
        for label, value in payload.measurements.items()
    ]
    captions = [
        (name, f"{value:.3f}" if isinstance(value, float) else str(value))
        for name, value in measurement_items
    ]

    table_data = [(name, value, f"{payload.confidences.get(name.replace(' ', '_').lower(), 0.0):.2f}") for name, value in captions]
    table_data.insert(0, ("Measurement", "Value", "Confidence"))

    ax = fig.add_subplot(111)
    ax.axis("off")
    table = ax.table(
        cellText=table_data,
        colLabels=None,
        cellLoc="left",
        loc="center",
        colColours=["#F1F1F1"] * 3,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(layout.table_fontsize)
    table.scale(1, 1.8)

    meta_text = (
        f"Model Version: {payload.model_version}\n"
        f"Generated: {payload.timestamp}\n"
        f"Patient ID: {payload.patient_info.patient_id}\n"
        f"Study ID: {payload.patient_info.study_id or 'N/A'}"
    )
    fig.text(0.60, 0.85, "Report Metadata", fontsize=layout.heading_fontsize)
    fig.text(0.60, 0.80, meta_text, fontsize=layout.text_fontsize)

    confidences_text = "AI Confidence Summary:\n"
    for key, value in payload.confidences.items():
        confidences_text += f"{key.replace('_', ' ').title()}: {value:.2f}\n"

    fig.text(0.60, 0.55, confidences_text, fontsize=layout.text_fontsize)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def generate_pdf_report(
    output_path: Union[str, Path],
    payload: ReportPayload,
    layout: Optional[ReportLayoutConfig] = None,
    save_json_payload: bool = True,
) -> Path:
    """Generate a hospital-grade PDF report for OCT inference results."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if layout is None:
        layout = ReportLayoutConfig()

    with PdfPages(out_path) as pdf:
        _render_title_page(pdf, payload, layout)
        _render_image_page(pdf, payload, layout)
        _render_measurements_page(pdf, payload, layout)

    if save_json_payload:
        json_path = out_path.with_suffix(".json")
        payload.save_json(json_path)

    return out_path


def build_payload(
    *,
    patient_info: PatientInfo,
    original_image: np.ndarray,
    predicted_mask: np.ndarray,
    overlay_image: np.ndarray,
    measurements: Dict[str, Optional[float]],
    confidences: Dict[str, float],
    model_version: str,
    summary: str,
    doctor_notes: str,
    report_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> ReportPayload:
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    return ReportPayload(
        patient_info=patient_info,
        original_image=_ensure_rgb(original_image),
        predicted_mask=predicted_mask.astype(np.int32),
        overlay_image=_ensure_rgb(overlay_image),
        measurements=measurements,
        confidences=confidences,
        model_version=model_version,
        timestamp=timestamp,
        summary=summary,
        doctor_notes=doctor_notes,
        report_id=report_id,
        extra=extra or {},
    )


__all__ = [
    "PatientInfo",
    "ReportLayoutConfig",
    "ReportPayload",
    "build_payload",
    "generate_pdf_report",
]
