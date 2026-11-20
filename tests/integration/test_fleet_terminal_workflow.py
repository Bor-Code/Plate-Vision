from collections.abc import Sequence

import pytest

from plate_vision.data.models import BoundingBox
from plate_vision.data.synthetic import encode_grayscale_png
from plate_vision.detection.models import DetectionPrediction
from plate_vision.pipeline.runner import PlateVisionPipeline
from plate_vision.recognition.decoding import DecodedPlate
from plate_vision.video.fleet import (
    FleetSlaConfig,
    FleetTerminalController,
    TerminalZoneType,
)


class _FleetDetector:
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


class _FleetRecognizer:
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
        return DecodedPlate(text=text, confidence=0.95, token_indices=(1,))


def test_alpr_pipeline_integrated_fleet_terminal_workflow() -> None:
    controller = FleetTerminalController(
        config=FleetSlaConfig(
            facility_id="PORT_TERMINAL_1",
            max_turnaround_minutes=45.0,
            demurrage_rate_per_hour=400.0,
            currency="TRY",
        )
    )
    pipeline = PlateVisionPipeline(
        detector=_FleetDetector(),
        recognizer=_FleetRecognizer(["34 A8C 123", "06 TRK 99", "34 ABC 123", "06 TRK 99"]),
    )
    png_bytes = encode_grayscale_png([[140 for _ in range(120)] for _ in range(48)])

    t0 = 1_700_000_000_000
    in_1 = pipeline.predict_bytes(png_bytes).plates[0]
    controller.check_in(
        in_1.plate_text,
        country_code=in_1.country_code,
        carrier_name="EURO_TRUCKING",
        assigned_zone=TerminalZoneType.LOADING_DOCK,
        timestamp_ms=t0,
    )

    in_2 = pipeline.predict_bytes(png_bytes).plates[0]
    controller.check_in(
        in_2.plate_text,
        country_code=in_2.country_code,
        carrier_name="BALKAN_EXPRESS",
        assigned_zone=TerminalZoneType.STAGING,
        timestamp_ms=t0 + 60_000,
    )

    out_1 = pipeline.predict_bytes(png_bytes).plates[0]
    rec_1 = controller.check_out(
        out_1.plate_text,
        country_code=out_1.country_code,
        timestamp_ms=t0 + (30 * 60_000),
    )
    assert rec_1.sla_breached is False
    assert rec_1.demurrage_fee == 0.0

    out_2 = pipeline.predict_bytes(png_bytes).plates[0]
    rec_2 = controller.check_out(
        out_2.plate_text,
        country_code=out_2.country_code,
        timestamp_ms=t0 + (76 * 60_000),
    )
    assert rec_2.sla_breached is True
    assert rec_2.overstay_minutes == pytest.approx(30.0)
    assert rec_2.demurrage_fee == pytest.approx(200.0)

    summary = controller.summarize()
    assert summary.completed_visits == 2
    assert summary.sla_breach_count == 1
    assert summary.total_demurrage_fee == pytest.approx(200.0)
