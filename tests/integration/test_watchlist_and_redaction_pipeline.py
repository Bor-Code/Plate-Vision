from collections.abc import Sequence

from plate_vision.data.models import BoundingBox
from plate_vision.data.synthetic import encode_grayscale_png
from plate_vision.detection.models import DetectionPrediction
from plate_vision.pipeline.privacy import (
    RedactionConfig,
    RedactionMode,
    redact_regions,
)
from plate_vision.pipeline.rectification import decode_image_to_grayscale
from plate_vision.pipeline.runner import PlateVisionPipeline
from plate_vision.pipeline.watchlist import (
    WatchlistCategory,
    WatchlistEntry,
    WatchlistMatcher,
)
from plate_vision.recognition.decoding import DecodedPlate


class _TwoPlateDetector:
    def detect(
        self,
        image_bytes: bytes,
        image_width: int,
        image_height: int,
    ) -> list[DetectionPrediction]:
        _ = (image_bytes, image_width, image_height)
        return [
            DetectionPrediction(
                bbox=BoundingBox(x_min=8.0, y_min=6.0, x_max=58.0, y_max=22.0),
                confidence=0.95,
                class_id=0,
            ),
            DetectionPrediction(
                bbox=BoundingBox(x_min=62.0, y_min=26.0, x_max=114.0, y_max=42.0),
                confidence=0.92,
                class_id=0,
            ),
        ]


class _TwoPlateRecognizer:
    def __init__(self) -> None:
        self._call = 0

    def recognize(
        self,
        crop_pixels: Sequence[Sequence[int]],
        *,
        use_beam_search: bool = False,
    ) -> DecodedPlate:
        _ = (crop_pixels, use_beam_search)
        self._call += 1
        if self._call == 1:
            return DecodedPlate(text="34 ABC 123", confidence=0.94, token_indices=(1,))
        return DecodedPlate(text="06 XYZ 789", confidence=0.91, token_indices=(2,))


def test_end_to_end_pipeline_watchlist_and_privacy_redaction() -> None:
    pipeline = PlateVisionPipeline(
        detector=_TwoPlateDetector(),
        recognizer=_TwoPlateRecognizer(),
    )
    grid = [[180 for _ in range(128)] for _ in range(48)]
    png_bytes = encode_grayscale_png(grid)

    infer_res = pipeline.predict_bytes(png_bytes)
    assert len(infer_res.plates) == 2

    matcher = WatchlistMatcher(
        entries=[
            WatchlistEntry(
                plate_text="34ABC123",
                country_code="TR",
                category=WatchlistCategory.ALLOWLIST,
                note="authorized fleet",
            )
        ],
        max_edit_distance=0,
    )
    first_matches = matcher.check(infer_res.plates[0].plate_text, infer_res.plates[0].country_code)
    second_matches = matcher.check(infer_res.plates[1].plate_text, infer_res.plates[1].country_code)
    assert len(first_matches) == 1
    assert first_matches[0].entry.category == WatchlistCategory.ALLOWLIST
    assert second_matches == ()

    decoded_grid = decode_image_to_grayscale(png_bytes)
    redacted = redact_regions(
        decoded_grid,
        [p.bbox for p in infer_res.plates],
        RedactionConfig(mode=RedactionMode.SOLID_FILL, fill_value=0),
    )
    assert redacted[0][0] == 180
    assert redacted[10][20] == 0
    assert redacted[32][80] == 0
