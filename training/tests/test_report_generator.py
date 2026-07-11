import numpy as np

from training.report_generator import (
    PatientInfo,
    build_payload,
    generate_pdf_report,
)


def test_generate_pdf_report(tmp_path):
    patient_info = PatientInfo(
        patient_id="P123",
        name="Jane Doe",
        date_of_birth="1980-01-01",
        sex="F",
        study_id="S456",
        exam_date="2026-07-11",
        referring_physician="Dr. Smith",
        operator="Tech A",
        notes="Routine exam.",
    )

    original = np.zeros((512, 512, 3), dtype=np.uint8)
    predicted_mask = np.zeros((512, 512), dtype=np.uint8)
    overlay = np.zeros((512, 512, 3), dtype=np.uint8)
    measurements = {
        "central_corneal_thickness_mm": 0.57,
        "anterior_chamber_depth_mm": 3.12,
    }
    confidences = {
        "central_corneal_thickness_mm": 0.92,
        "anterior_chamber_depth_mm": 0.86,
    }

    payload = build_payload(
        patient_info=patient_info,
        original_image=original,
        predicted_mask=predicted_mask,
        overlay_image=overlay,
        measurements=measurements,
        confidences=confidences,
        model_version="v1.0.0",
        summary="Normal cornea with successful segmentation.",
        doctor_notes="Recommend follow-up in 6 months.",
    )

    pdf_path = tmp_path / "report.pdf"
    generated = generate_pdf_report(pdf_path, payload)
    assert generated.exists()
    assert generated.suffix == ".pdf"
    json_path = generated.with_suffix(".json")
    assert json_path.exists()
