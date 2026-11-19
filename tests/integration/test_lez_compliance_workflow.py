from collections.abc import Sequence

import pytest

from plate_vision.data.models import BoundingBox
from plate_vision.data.synthetic import encode_grayscale_png
from plate_vision.detection.models import DetectionPrediction
from plate_vision.pipeline.runner import PlateVisionPipeline
from plate_vision.recognition.decoding import DecodedPlate
from plate_vision.video.lez import (
    EmissionClass,
    LowEmissionZoneConfig,
    LowEmissionZoneInspector,
    VehicleEmissionProfile,
)


class _LezDetector:
    def detect(
        self,
        image_bytes: bytes,
        image_width: int,
        image_height: int,
    ) -> list[DetectionPrediction]:
        _ = (image_bytes, image_width, image_height)
        return [
            DetectionPrediction(
                bbox=BoundingBox(x_min=8.0, y_min=8.0, x_max=104.0, y_max=36.0),
                confidence=0.96,
                class_id=0,
            )
        ]


class _LezRecognizer:
    def __init__(self, texts: Sequence[str]) -> None:
        self._texts = list(texts)
        self._idx = 0

    def recognize(
        self,
        crop_pixels: Sequence[Sequence[int]],
        *,
        use_beam_search: bool = False,
    ) -> DecodedPlate:
        _ = (crop_pixels, use_beam_search)
        text = self._texts[self._idx]
        self._idx += 1
        return DecodedPlate(text=text, confidence=0.94, token_indices=(1,))


def test_alpr_pipeline_integrated_lez_compliance_workflow() -> None:
    inspector = LowEmissionZoneInspector(
        config=LowEmissionZoneConfig(
            zone_id="BOSPHORUS_CORE",
            allowed_classes=(EmissionClass.ZERO_EMISSION, EmissionClass.EURO_6),
            enforce_odd_even_days=True,
            non_compliant_fine_amount=1500.0,
            unregistered_fine_amount=2000.0,
            odd_even_fine_amount=800.0,
            currency="TRY",
        ),
        registry=[
            VehicleEmissionProfile(
                plate_text="34ABC124",
                country_code="TR",
                emission_class=EmissionClass.EURO_6,
            ),
            VehicleEmissionProfile(
                plate_text="34OLD999",
                country_code="TR",
                emission_class=EmissionClass.EURO_3_OR_LOWER,
            ),
        ],
    )
    pipeline = PlateVisionPipeline(
        detector=_LezDetector(),
        recognizer=_LezRecognizer(["34 A8C 124", "34 OLD 999", "06 UNR 555"]),
    )
    png_bytes = encode_grayscale_png([[140 for _ in range(120)] for _ in range(48)])

    p1 = pipeline.predict_bytes(png_bytes).plates[0]
    res1 = inspector.inspect(p1.plate_text, country_code=p1.country_code, day_of_month=10)
    assert res1.plate_text == "34ABC124"
    assert res1.is_compliant is True
    assert res1.fine_amount == 0.0

    p2 = pipeline.predict_bytes(png_bytes).plates[0]
    res2 = inspector.inspect(p2.plate_text, country_code=p2.country_code, day_of_month=10)
    assert res2.is_compliant is False
    assert res2.violation_reason == "restricted_emission_class"
    assert res2.fine_amount == pytest.approx(1500.0)

    p3 = pipeline.predict_bytes(png_bytes).plates[0]
    res3 = inspector.inspect(p3.plate_text, country_code=p3.country_code, day_of_month=10)
    assert res3.is_compliant is False
    assert res3.violation_reason == "unregistered_vehicle"
    assert res3.fine_amount == pytest.approx(2000.0)

    summary = inspector.summarize()
    assert summary.total_inspections == 3
    assert summary.compliant_count == 1
    assert summary.violation_count == 2
    assert summary.total_fine_amount == pytest.approx(3500.0)
