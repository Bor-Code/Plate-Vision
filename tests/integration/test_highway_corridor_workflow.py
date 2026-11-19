from collections.abc import Sequence

import pytest

from plate_vision.data.models import BoundingBox
from plate_vision.data.synthetic import encode_grayscale_png
from plate_vision.detection.models import DetectionPrediction
from plate_vision.pipeline.runner import PlateVisionPipeline
from plate_vision.recognition.decoding import DecodedPlate
from plate_vision.video.corridor import CorridorConfig, HighwayCorridorMonitor


class _CorridorDetector:
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
                confidence=0.97,
                class_id=0,
            )
        ]


class _CorridorRecognizer:
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


def test_alpr_pipeline_integrated_highway_corridor_workflow() -> None:
    monitor = HighwayCorridorMonitor(
        config=CorridorConfig(
            corridor_id="NORTH_MARMARA",
            distance_km=15.0,
            speed_limit_kmh=120.0,
            tolerance_percent=5.0,
            toll_rate_per_km=4.0,
            speeding_fine_amount=1800.0,
            currency="TRY",
        )
    )
    pipeline = PlateVisionPipeline(
        detector=_CorridorDetector(),
        recognizer=_CorridorRecognizer(["34 A8C 123", "06 ZZZ 888", "34 ABC 123", "06 ZZZ 888"]),
    )
    png_bytes = encode_grayscale_png([[140 for _ in range(120)] for _ in range(48)])

    t0 = 1_700_000_000_000
    in_1 = pipeline.predict_bytes(png_bytes).plates[0]
    monitor.record_entry(
        in_1.plate_text,
        country_code=in_1.country_code,
        checkpoint_id="KM_00_GATE",
        timestamp_ms=t0,
    )

    in_2 = pipeline.predict_bytes(png_bytes).plates[0]
    monitor.record_entry(
        in_2.plate_text,
        country_code=in_2.country_code,
        checkpoint_id="KM_00_GATE",
        timestamp_ms=t0 + 10_000,
    )

    out_1 = pipeline.predict_bytes(png_bytes).plates[0]
    rec_1 = monitor.record_exit(
        out_1.plate_text,
        country_code=out_1.country_code,
        checkpoint_id="KM_15_GATE",
        timestamp_ms=t0 + 450_000,
    )
    assert rec_1.average_speed_kmh == pytest.approx(120.0)
    assert rec_1.is_speed_violation is False
    assert rec_1.total_charge == pytest.approx(60.0)

    out_2 = pipeline.predict_bytes(png_bytes).plates[0]
    rec_2 = monitor.record_exit(
        out_2.plate_text,
        country_code=out_2.country_code,
        checkpoint_id="KM_15_GATE",
        timestamp_ms=t0 + 370_000,
    )
    assert rec_2.average_speed_kmh == pytest.approx(150.0)
    assert rec_2.is_speed_violation is True
    assert rec_2.total_charge == pytest.approx(1860.0)

    summary = monitor.summarize()
    assert summary.total_passages == 2
    assert summary.active_in_corridor == 0
    assert summary.violation_count == 1
    assert summary.total_toll_revenue == pytest.approx(120.0)
    assert summary.total_fine_revenue == pytest.approx(1800.0)
