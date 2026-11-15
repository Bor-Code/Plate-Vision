import json
from collections.abc import Sequence
from pathlib import Path

from plate_vision.data.active_learning import (
    ActiveLearningConfig,
    ActiveLearningSampler,
    HardExampleCandidate,
)
from plate_vision.data.models import BoundingBox
from plate_vision.data.synthetic import encode_grayscale_png
from plate_vision.detection.models import DetectionPrediction
from plate_vision.pipeline.runner import PlateVisionPipeline
from plate_vision.recognition.decoding import DecodedPlate


class _VariableDetector:
    def detect(
        self,
        image_bytes: bytes,
        image_width: int,
        image_height: int,
    ) -> list[DetectionPrediction]:
        _ = (image_bytes, image_width, image_height)
        return [
            DetectionPrediction(
                bbox=BoundingBox(x_min=8.0, y_min=8.0, x_max=88.0, y_max=34.0),
                confidence=0.68,
                class_id=0,
            ),
            DetectionPrediction(
                bbox=BoundingBox(x_min=10.0, y_min=36.0, x_max=96.0, y_max=60.0),
                confidence=0.95,
                class_id=0,
            ),
        ]


class _VariableRecognizer:
    def __init__(self) -> None:
        self._idx = 0

    def recognize(
        self,
        crop_pixels: Sequence[Sequence[int]],
        *,
        use_beam_search: bool = False,
    ) -> DecodedPlate:
        _ = (crop_pixels, use_beam_search)
        self._idx += 1
        if self._idx == 1:
            return DecodedPlate(text="06ABC123", confidence=0.96, token_indices=(1, 2))
        return DecodedPlate(text="34 A8C 12O", confidence=0.64, token_indices=(1, 2))


def test_active_learning_end_to_end_pipeline_manifest_export(tmp_path: Path) -> None:
    pipeline = PlateVisionPipeline(
        detector=_VariableDetector(),
        recognizer=_VariableRecognizer(),
    )
    grid = [[125 for _ in range(120)] for _ in range(72)]
    png_bytes = encode_grayscale_png(grid)
    infer_res = pipeline.predict_bytes(png_bytes)

    sampler = ActiveLearningSampler(
        ActiveLearningConfig(
            recognition_confidence_threshold=0.80,
            detection_confidence_threshold=0.75,
            max_candidates=10,
        )
    )
    candidates: list[HardExampleCandidate] = []
    for idx, plate in enumerate(infer_res.plates):
        cand = sampler.evaluate_prediction(
            sample_id=f"frame_0_plate_{idx}",
            bbox=plate.bbox,
            raw_text=plate.raw_text,
            plate_text=plate.plate_text,
            country_code=plate.country_code,
            detection_confidence=plate.detection_confidence,
            recognition_confidence=plate.recognition_confidence,
            is_valid_format=plate.is_valid_format,
            was_corrected=plate.was_corrected,
        )
        if cand is not None:
            candidates.append(cand)

    top = sampler.select_top_candidates(candidates)
    assert len(top) == 1
    assert top[0].plate_text == "34ABC120"
    assert top[0].was_corrected is True

    manifest_path = tmp_path / "active_learning" / "hard_examples.jsonl"
    sampler.export_manifest(top, manifest_path)
    assert manifest_path.is_file()
    lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["plate_text"] == "34ABC120"
