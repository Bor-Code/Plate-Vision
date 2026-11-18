from collections.abc import Sequence

from plate_vision.data.models import BoundingBox
from plate_vision.data.synthetic import encode_grayscale_png
from plate_vision.detection.models import DetectionPrediction
from plate_vision.evaluation.shadow import ShadowPipelineRunner
from plate_vision.pipeline.runner import PlateVisionPipeline
from plate_vision.recognition.decoding import DecodedPlate


class _StubDetector:
    def detect(
        self,
        image_bytes: bytes,
        image_width: int,
        image_height: int,
    ) -> list[DetectionPrediction]:
        _ = (image_bytes, image_width, image_height)
        return [
            DetectionPrediction(
                bbox=BoundingBox(x_min=10.0, y_min=8.0, x_max=90.0, y_max=36.0),
                confidence=0.95,
                class_id=0,
            )
        ]


class _SequenceRecognizer:
    def __init__(self, texts: Sequence[str], confidence: float) -> None:
        self._texts = list(texts)
        self._confidence = confidence
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
        return DecodedPlate(text=text, confidence=self._confidence, token_indices=(1,))


def test_shadow_ab_comparison_blocks_promotion_on_disagreement() -> None:
    primary = PlateVisionPipeline(
        detector=_StubDetector(),
        recognizer=_SequenceRecognizer(["34ABC123", "06XYZ789", "35DEF456"], confidence=0.90),
    )
    candidate = PlateVisionPipeline(
        detector=_StubDetector(),
        recognizer=_SequenceRecognizer(["34ABC123", "06XYZ780", "35DEF456"], confidence=0.93),
    )
    runner = ShadowPipelineRunner(
        primary,
        candidate,
        min_agreement_rate=0.90,
        max_latency_regression_ms=50.0,
    )

    png_bytes = encode_grayscale_png([[135 for _ in range(120)] for _ in range(48)])
    for idx in range(3):
        primary_res, _ = runner.predict_and_compare(f"frame_{idx}", png_bytes)
        assert len(primary_res.plates) == 1

    summary = runner.summarize()
    assert summary.total_samples == 3
    assert summary.agreement_rate == round(2 / 3, 6)
    assert summary.disagreed_sample_ids == ("frame_1",)
    assert summary.mean_confidence_delta > 0.0
    assert summary.promote_recommended is False
