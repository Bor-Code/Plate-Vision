from collections.abc import Sequence

import pytest

from plate_vision.data.models import BoundingBox
from plate_vision.data.synthetic import encode_grayscale_png
from plate_vision.detection.models import DetectionPrediction
from plate_vision.evaluation.shadow import ShadowPipelineRunner
from plate_vision.pipeline.runner import PlateVisionPipeline
from plate_vision.recognition.decoding import DecodedPlate
from plate_vision.utils.exceptions import ConfigurationError, InvalidInputError


class _DummyDetector:
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


class _FixedRecognizer:
    def __init__(self, text: str, conf: float) -> None:
        self.text = text
        self.conf = conf

    def recognize(
        self,
        crop_pixels: Sequence[Sequence[int]],
        *,
        use_beam_search: bool = False,
    ) -> DecodedPlate:
        _ = (crop_pixels, use_beam_search)
        return DecodedPlate(text=self.text, confidence=self.conf, token_indices=(1,))


def test_shadow_pipeline_runner_validation_and_empty_summary() -> None:
    p1 = PlateVisionPipeline(
        detector=_DummyDetector(), recognizer=_FixedRecognizer("34ABC123", 0.9)
    )
    p2 = PlateVisionPipeline(
        detector=_DummyDetector(), recognizer=_FixedRecognizer("34ABC123", 0.92)
    )

    with pytest.raises(ConfigurationError):
        ShadowPipelineRunner(p1, p2, min_agreement_rate=1.5)
    with pytest.raises(ConfigurationError):
        ShadowPipelineRunner(p1, p2, max_latency_regression_ms=-1.0)

    runner = ShadowPipelineRunner(p1, p2)
    empty = runner.summarize()
    assert empty.total_samples == 0
    assert empty.promote_recommended is False

    png_bytes = encode_grayscale_png([[130 for _ in range(120)] for _ in range(48)])
    with pytest.raises(InvalidInputError):
        runner.predict_and_compare("   ", png_bytes)

    primary_res, record = runner.predict_and_compare("s1", png_bytes)
    assert primary_res.plates[0].plate_text == "34ABC123"
    assert record.exact_agreement is True
    assert len(runner.records()) == 1
    summary = runner.summarize()
    assert summary.promote_recommended is True

    runner.clear()
    assert runner.summarize().total_samples == 0
