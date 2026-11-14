from collections.abc import Sequence

import pytest

from plate_vision.data.models import BoundingBox
from plate_vision.data.synthetic import encode_grayscale_png
from plate_vision.detection.models import DetectionPrediction
from plate_vision.pipeline.rectification import Point2D
from plate_vision.pipeline.runner import PlateVisionPipeline
from plate_vision.recognition.decoding import DecodedPlate
from plate_vision.utils.exceptions import (
    CorruptedImageError,
    EmptyDetectionError,
    InferenceTimeoutError,
    InvalidInputError,
)


class StubDetector:
    def __init__(self, predictions: list[DetectionPrediction]) -> None:
        self._predictions = predictions

    def detect(
        self,
        image_bytes: bytes,
        image_width: int,
        image_height: int,
    ) -> list[DetectionPrediction]:
        _ = (image_bytes, image_width, image_height)
        return list(self._predictions)


class StubRecognizer:
    def __init__(self, outputs: list[DecodedPlate]) -> None:
        self._outputs = outputs
        self._index = 0

    def recognize(
        self,
        crop_pixels: Sequence[Sequence[int]],
        *,
        use_beam_search: bool = False,
    ) -> DecodedPlate:
        _ = (crop_pixels, use_beam_search)
        out = self._outputs[self._index % len(self._outputs)]
        self._index += 1
        return out


def _make_sample_png(width: int = 160, height: int = 80) -> bytes:
    grid = [[(x + y * 2) % 200 + 20 for x in range(width)] for y in range(height)]
    return encode_grayscale_png(grid)


def test_pipeline_handles_multiple_plates_skewed_and_confusion_correction() -> None:
    box1 = BoundingBox(x_min=10.0, y_min=10.0, x_max=90.0, y_max=40.0)
    box2 = BoundingBox(x_min=20.0, y_min=45.0, x_max=110.0, y_max=72.0)
    tiny_box = BoundingBox(x_min=0.0, y_min=0.0, x_max=5.0, y_max=4.0)

    detector = StubDetector(
        [
            DetectionPrediction(bbox=box1, confidence=0.95, class_id=0),
            DetectionPrediction(bbox=box2, confidence=0.88, class_id=0),
            DetectionPrediction(bbox=tiny_box, confidence=0.99, class_id=0),
        ]
    )
    recognizer = StubRecognizer(
        [
            DecodedPlate(text="34A80123", confidence=0.90, token_indices=(1, 2)),
            DecodedPlate(text="O6I2S456", confidence=0.85, token_indices=(3, 4)),
        ]
    )
    pipeline = PlateVisionPipeline(detector=detector, recognizer=recognizer)

    corners = {
        0: (
            Point2D(x=10.0, y=12.0),
            Point2D(x=88.0, y=8.0),
            Point2D(x=90.0, y=38.0),
            Point2D(x=12.0, y=40.0),
        )
    }
    result = pipeline.predict_bytes(_make_sample_png(), corners_override=corners)
    assert len(result.plates) == 2
    assert result.plates[0].plate_text == "34AB0123"
    assert result.plates[0].is_valid_format is True
    assert result.plates[0].was_corrected is True
    assert result.plates[1].plate_text == "06IZS456"
    assert result.plates[1].is_valid_format is True
    assert result.plates[1].was_corrected is True


def test_pipeline_empty_detection_modes() -> None:
    detector = StubDetector([])
    recognizer = StubRecognizer([DecodedPlate(text="34ABC123", confidence=0.9, token_indices=(1,))])

    lenient_pipeline = PlateVisionPipeline(
        detector=detector,
        recognizer=recognizer,
        require_detection=False,
    )
    empty_res = lenient_pipeline.predict_bytes(_make_sample_png())
    assert len(empty_res.plates) == 0

    strict_pipeline = PlateVisionPipeline(
        detector=detector,
        recognizer=recognizer,
        require_detection=True,
    )
    with pytest.raises(EmptyDetectionError):
        strict_pipeline.predict_bytes(_make_sample_png())


def test_pipeline_enforces_timeout_and_input_validation() -> None:
    ticks = iter([0.0, 0.05, 1.50, 2.0, 2.1, 2.2])

    detector = StubDetector(
        [
            DetectionPrediction(
                bbox=BoundingBox(x_min=10.0, y_min=10.0, x_max=80.0, y_max=40.0),
                confidence=0.9,
                class_id=0,
            )
        ]
    )
    recognizer = StubRecognizer([DecodedPlate(text="34ABC123", confidence=0.9, token_indices=(1,))])

    pipeline = PlateVisionPipeline(
        detector=detector,
        recognizer=recognizer,
        timeout_seconds=0.5,
        clock=lambda: next(ticks),
    )
    with pytest.raises(InferenceTimeoutError):
        pipeline.predict_bytes(_make_sample_png())

    with pytest.raises(InvalidInputError):
        pipeline.predict_bytes(b"non_image_upload_payload")

    with pytest.raises(CorruptedImageError):
        pipeline.predict_bytes(b"\x89PNG\r\n\x1a\ncorrupted_png_header")

    small_limit_pipeline = PlateVisionPipeline(
        detector=detector,
        recognizer=recognizer,
        max_payload_bytes=64,
    )
    with pytest.raises(InvalidInputError):
        small_limit_pipeline.predict_bytes(_make_sample_png())
