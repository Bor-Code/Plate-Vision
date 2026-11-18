from collections.abc import Sequence

from plate_vision.data.models import BoundingBox
from plate_vision.data.synthetic import encode_grayscale_png
from plate_vision.detection.models import DetectionPrediction
from plate_vision.pipeline.runner import PlateVisionPipeline
from plate_vision.pipeline.watchlist import (
    WatchlistCategory,
    WatchlistEntry,
    WatchlistMatcher,
)
from plate_vision.recognition.decoding import DecodedPlate
from plate_vision.video.parking import ParkingLotManager, ParkingTariffConfig


class _GateDetector:
    def detect(
        self,
        image_bytes: bytes,
        image_width: int,
        image_height: int,
    ) -> list[DetectionPrediction]:
        _ = (image_bytes, image_width, image_height)
        return [
            DetectionPrediction(
                bbox=BoundingBox(x_min=10.0, y_min=8.0, x_max=96.0, y_max=36.0),
                confidence=0.96,
                class_id=0,
            )
        ]


class _GateRecognizer:
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


def test_alpr_pipeline_integrated_parking_gate_and_billing() -> None:
    matcher = WatchlistMatcher(
        entries=[
            WatchlistEntry(plate_text="34VIP999", country_code="TR", category=WatchlistCategory.VIP)
        ]
    )
    manager = ParkingLotManager(
        tariff=ParkingTariffConfig(
            grace_period_minutes=15.0,
            hourly_rate=50.0,
            daily_max_rate=300.0,
            vip_free_parking=True,
        ),
        watchlist=matcher,
        capacity=50,
    )
    pipeline = PlateVisionPipeline(
        detector=_GateDetector(),
        recognizer=_GateRecognizer(["34 A8C 123", "34 VIP 999", "34 ABC 123", "34 VIP 999"]),
    )
    png_bytes = encode_grayscale_png([[135 for _ in range(120)] for _ in range(48)])

    t0 = 1_700_000_000_000
    r_in_1 = pipeline.predict_bytes(png_bytes).plates[0]
    d_in_1 = manager.record_entry(
        r_in_1.plate_text, country_code=r_in_1.country_code, timestamp_ms=t0
    )
    assert d_in_1.open_barrier is True

    r_in_2 = pipeline.predict_bytes(png_bytes).plates[0]
    d_in_2 = manager.record_entry(
        r_in_2.plate_text, country_code=r_in_2.country_code, timestamp_ms=t0 + 60_000
    )
    assert d_in_2.open_barrier is True
    assert manager.occupancy_count == 2

    r_out_1 = pipeline.predict_bytes(png_bytes).plates[0]
    d_out_1 = manager.record_exit(
        r_out_1.plate_text,
        country_code=r_out_1.country_code,
        timestamp_ms=t0 + 90 * 60_000,
    )
    assert d_out_1.open_barrier is True
    assert d_out_1.session is not None
    assert d_out_1.session.fee_amount == 100.0

    r_out_2 = pipeline.predict_bytes(png_bytes).plates[0]
    d_out_2 = manager.record_exit(
        r_out_2.plate_text,
        country_code=r_out_2.country_code,
        timestamp_ms=t0 + 180 * 60_000,
    )
    assert d_out_2.open_barrier is True
    assert d_out_2.session is not None
    assert d_out_2.session.fee_amount == 0.0
    assert manager.occupancy_count == 0
    assert manager.total_revenue() == 100.0
