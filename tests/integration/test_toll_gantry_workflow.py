from collections.abc import Sequence

from plate_vision.data.models import BoundingBox
from plate_vision.data.synthetic import encode_grayscale_png
from plate_vision.detection.models import DetectionPrediction
from plate_vision.pipeline.runner import PlateVisionPipeline
from plate_vision.recognition.decoding import DecodedPlate
from plate_vision.video import (
    TollAccount,
    TollGantryEngine,
    TollTariffConfig,
    TollTransactionStatus,
    VehicleTollClass,
    VideoFrameInput,
    VideoStreamProcessor,
)


class _TollDetector:
    def detect(
        self,
        image_bytes: bytes,
        image_width: int,
        image_height: int,
    ) -> list[DetectionPrediction]:
        _ = (image_bytes, image_width, image_height)
        return [
            DetectionPrediction(
                bbox=BoundingBox(x_min=12.0, y_min=8.0, x_max=100.0, y_max=36.0),
                confidence=0.96,
                class_id=0,
            )
        ]


class _TollRecognizer:
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


def test_video_stream_integrated_electronic_toll_gantry_workflow() -> None:
    pipeline = PlateVisionPipeline(
        detector=_TollDetector(),
        recognizer=_TollRecognizer(["34 A8C 123", "34 ABC 123", "06 UNR 555"]),
    )
    processor = VideoStreamProcessor(pipeline=pipeline)
    engine = TollGantryEngine(
        tariff=TollTariffConfig(
            gantry_id="highway_m1_north",
            min_pass_interval_seconds=30.0,
            violation_surcharge=150.0,
        )
    )
    engine.register_account(
        TollAccount(
            plate_text="34ABC123",
            country_code="TR",
            vehicle_class=VehicleTollClass.PASSENGER,
            balance=200.0,
        )
    )

    png_bytes = encode_grayscale_png([[135 for _ in range(120)] for _ in range(48)])
    base_ts = 11 * 3_600_000
    frames = [
        VideoFrameInput(frame_index=0, timestamp_ms=base_ts, image_bytes=png_bytes),
        VideoFrameInput(frame_index=1, timestamp_ms=base_ts + 2_000, image_bytes=png_bytes),
        VideoFrameInput(frame_index=2, timestamp_ms=base_ts + 45_000, image_bytes=png_bytes),
    ]

    results = [processor.process_frame(frame) for frame in frames]
    tx_list = []
    for frame, frame_res in zip(frames, results, strict=True):
        for track in frame_res.tracked_detections:
            tx_list.append(
                engine.process_passage(
                    track.prediction.plate_text,
                    timestamp_ms=int(frame.timestamp_ms),
                    country_code=track.prediction.country_code,
                )
            )

    assert len(tx_list) == 3
    assert tx_list[0].status == TollTransactionStatus.PAID_PREPAID
    assert tx_list[0].plate_text == "34ABC123"
    assert tx_list[1].status == TollTransactionStatus.DUPLICATE_IGNORED
    assert tx_list[2].status == TollTransactionStatus.UNPAID_VIOLATION
    assert tx_list[2].plate_text == "06UNR555"

    summary = engine.summary()
    assert summary.total_passages == 3
    assert summary.paid_prepaid_count == 1
    assert summary.duplicate_ignored_count == 1
    assert summary.violation_count == 1
    assert summary.total_collected_revenue == 35.0
    assert summary.total_unpaid_amount == 185.0
