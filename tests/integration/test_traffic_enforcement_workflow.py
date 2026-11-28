from collections.abc import Sequence

from plate_vision.data.models import BoundingBox
from plate_vision.data.synthetic import encode_grayscale_png
from plate_vision.detection.models import DetectionPrediction
from plate_vision.pipeline.runner import PlateVisionPipeline
from plate_vision.recognition.decoding import DecodedPlate
from plate_vision.video import (
    EnforcementZoneRule,
    SignalPhase,
    TrafficEnforcementEngine,
    VideoFrameInput,
    VideoStreamProcessor,
    ViolationObservation,
    ViolationType,
)


class _EnforcementDetector:
    def detect(
        self,
        image_bytes: bytes,
        image_width: int,
        image_height: int,
    ) -> list[DetectionPrediction]:
        _ = (image_bytes, image_width, image_height)
        return [
            DetectionPrediction(
                bbox=BoundingBox(x_min=14.0, y_min=10.0, x_max=104.0, y_max=36.0),
                confidence=0.96,
                class_id=0,
            )
        ]


class _EnforcementRecognizer:
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


def test_video_stream_integrated_traffic_enforcement_workflow() -> None:
    pipeline = PlateVisionPipeline(
        detector=_EnforcementDetector(),
        recognizer=_EnforcementRecognizer(["34 RLD 901", "34 BUS 001", "06 VIO 888"]),
    )
    processor = VideoStreamProcessor(pipeline=pipeline)
    engine = TrafficEnforcementEngine(
        rules=[
            EnforcementZoneRule(
                zone_id="red_intersection_01",
                violation_type=ViolationType.RED_LIGHT,
                fine_amount=1500.0,
                red_light_grace_ms=400,
            ),
            EnforcementZoneRule(
                zone_id="bus_corridor_01",
                violation_type=ViolationType.BUS_LANE,
                fine_amount=850.0,
            ),
        ]
    )
    engine.add_exempt_plate("34BUS001", country_code="TR", zone_id="bus_corridor_01")

    png_bytes = encode_grayscale_png([[135 for _ in range(120)] for _ in range(48)])
    base_ts = 1_700_000_100_000
    frames = [
        VideoFrameInput(frame_index=0, timestamp_ms=base_ts, image_bytes=png_bytes),
        VideoFrameInput(frame_index=1, timestamp_ms=base_ts + 2500, image_bytes=png_bytes),
        VideoFrameInput(frame_index=2, timestamp_ms=base_ts + 5000, image_bytes=png_bytes),
    ]
    results = [processor.process_frame(frame) for frame in frames]

    det0 = results[0].tracked_detections[0]
    t0 = engine.evaluate(
        ViolationObservation(
            zone_id="red_intersection_01",
            plate_text=det0.prediction.plate_text,
            country_code=det0.prediction.country_code,
            timestamp_ms=int(frames[0].timestamp_ms),
            confidence=det0.prediction.combined_confidence,
            frame_indices=(0,),
            signal_phase=SignalPhase.RED,
            signal_elapsed_ms=1200,
        )
    )
    assert t0 is not None
    assert t0.plate_text == "34RLD901"

    det1 = results[1].tracked_detections[0]
    t1 = engine.evaluate(
        ViolationObservation(
            zone_id="bus_corridor_01",
            plate_text=det1.prediction.plate_text,
            country_code=det1.prediction.country_code,
            timestamp_ms=int(frames[1].timestamp_ms),
            confidence=det1.prediction.combined_confidence,
            frame_indices=(1,),
        )
    )
    assert t1 is None

    det2 = results[2].tracked_detections[0]
    t2 = engine.evaluate(
        ViolationObservation(
            zone_id="bus_corridor_01",
            plate_text=det2.prediction.plate_text,
            country_code=det2.prediction.country_code,
            timestamp_ms=int(frames[2].timestamp_ms),
            confidence=det2.prediction.combined_confidence,
            frame_indices=(2,),
        )
    )
    assert t2 is not None
    assert t2.plate_text == "06VIO888"

    summary = engine.summary()
    assert summary.total_observations == 3
    assert summary.tickets_issued == 2
    assert summary.exempt_skipped == 1
    assert summary.total_fine_amount == 2350.0
    assert summary.by_violation_type == {"red_light": 1, "bus_lane": 1}
