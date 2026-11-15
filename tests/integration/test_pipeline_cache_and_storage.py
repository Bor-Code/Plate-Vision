from collections.abc import Sequence
from pathlib import Path

from plate_vision.data.models import BoundingBox
from plate_vision.data.synthetic import encode_grayscale_png
from plate_vision.detection.models import DetectionPrediction
from plate_vision.pipeline.runner import PipelineInferenceResult, PlateVisionPipeline
from plate_vision.recognition.decoding import DecodedPlate
from plate_vision.utils.storage import (
    LocalJsonlPredictionStore,
    PredictionCache,
)


class _CountingDetector:
    def __init__(self) -> None:
        self.call_count = 0

    def detect(
        self,
        image_bytes: bytes,
        image_width: int,
        image_height: int,
    ) -> list[DetectionPrediction]:
        _ = (image_bytes, image_width, image_height)
        self.call_count += 1
        return [
            DetectionPrediction(
                bbox=BoundingBox(x_min=10.0, y_min=8.0, x_max=90.0, y_max=36.0),
                confidence=0.95,
                class_id=0,
            )
        ]


class _CountingRecognizer:
    def __init__(self) -> None:
        self.call_count = 0

    def recognize(
        self,
        crop_pixels: Sequence[Sequence[int]],
        *,
        use_beam_search: bool = False,
    ) -> DecodedPlate:
        _ = (crop_pixels, use_beam_search)
        self.call_count += 1
        return DecodedPlate(text="06 DEF 456", confidence=0.92, token_indices=(1, 2))


def test_pipeline_uses_cache_and_writes_audit_records(tmp_path: Path) -> None:
    detector = _CountingDetector()
    recognizer = _CountingRecognizer()
    cache = PredictionCache[PipelineInferenceResult](max_entries=16)
    audit_store = LocalJsonlPredictionStore(tmp_path / "audit.jsonl")

    pipeline = PlateVisionPipeline(
        detector=detector,
        recognizer=recognizer,
        cache=cache,
        storage=audit_store,
    )

    grid = [[135 for _ in range(120)] for _ in range(48)]
    png_bytes = encode_grayscale_png(grid)

    first = pipeline.predict_bytes(png_bytes)
    second = pipeline.predict_bytes(png_bytes)

    assert first.plates[0].plate_text == "06DEF456"
    assert second.plates[0].plate_text == "06DEF456"
    assert detector.call_count == 1
    assert recognizer.call_count == 1
    assert cache.hits == 1
    assert cache.misses == 1

    records = audit_store.list_records()
    assert len(records) == 2
    assert records[0].cached is False
    assert records[1].cached is True
    assert records[1].plate_texts == ("06DEF456",)
