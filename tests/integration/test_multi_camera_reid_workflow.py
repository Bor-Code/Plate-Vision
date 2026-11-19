from collections.abc import Sequence

from plate_vision.data.models import BoundingBox
from plate_vision.data.synthetic import encode_grayscale_png
from plate_vision.detection.models import DetectionPrediction
from plate_vision.pipeline.runner import PlateVisionPipeline
from plate_vision.recognition.decoding import DecodedPlate
from plate_vision.video.journey import (
    CameraNode,
    CameraSighting,
    MultiCameraJourneyTracker,
)


class _JourneyDetector:
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


class _JourneyRecognizer:
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


def test_alpr_pipeline_integrated_multi_camera_reid_workflow() -> None:
    tracker = MultiCameraJourneyTracker(
        cameras=(
            CameraNode(camera_id="BRIDGE_W", latitude=41.045, longitude=29.034, zone_name="EUROPE"),
            CameraNode(camera_id="BRIDGE_E", latitude=41.040, longitude=29.045, zone_name="ASIA"),
            CameraNode(camera_id="HIGHWAY_K", latitude=40.765, longitude=29.940, zone_name="EAST"),
        ),
        cloned_plate_speed_threshold_kmh=240.0,
    )
    pipeline = PlateVisionPipeline(
        detector=_JourneyDetector(),
        recognizer=_JourneyRecognizer(["34 A8C 123", "34 ABC 123", "06 ZZZ 99", "06 ZZZ 99"]),
    )
    png_bytes = encode_grayscale_png([[140 for _ in range(120)] for _ in range(48)])

    t0 = 1_700_000_000_000
    d1 = pipeline.predict_bytes(png_bytes).plates[0]
    tracker.record_sighting(
        CameraSighting(
            camera_id="BRIDGE_W",
            plate_text=d1.plate_text,
            country_code=d1.country_code,
            timestamp_ms=t0,
            confidence=d1.combined_confidence,
        )
    )
    d2 = pipeline.predict_bytes(png_bytes).plates[0]
    tracker.record_sighting(
        CameraSighting(
            camera_id="BRIDGE_E",
            plate_text=d2.plate_text,
            country_code=d2.country_code,
            timestamp_ms=t0 + 60_000,
            confidence=d2.combined_confidence,
        )
    )

    c1 = pipeline.predict_bytes(png_bytes).plates[0]
    tracker.record_sighting(
        CameraSighting(
            camera_id="BRIDGE_W",
            plate_text=c1.plate_text,
            country_code=c1.country_code,
            timestamp_ms=t0,
            confidence=c1.combined_confidence,
        )
    )
    c2 = pipeline.predict_bytes(png_bytes).plates[0]
    tracker.record_sighting(
        CameraSighting(
            camera_id="HIGHWAY_K",
            plate_text=c2.plate_text,
            country_code=c2.country_code,
            timestamp_ms=t0 + 300_000,
            confidence=c2.combined_confidence,
        )
    )

    normal = tracker.build_journey("34ABC123", country_code="TR")
    assert normal.visited_zones == ("EUROPE", "ASIA")
    assert normal.has_cloned_plate_anomaly is False

    cloned = tracker.build_journey("06ZZZ99", country_code="TR")
    assert cloned.has_cloned_plate_anomaly is True
    assert tracker.anomalous_vehicles() == (("TR", "06ZZZ99"),)
