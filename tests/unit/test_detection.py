from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from plate_vision.config.schema import DetectorConfig
from plate_vision.data.models import BoundingBox
from plate_vision.detection import (
    DetectionPrediction,
    DetectorCheckpoint,
    DetectorMetrics,
    DetectorTrainer,
    UltralyticsYoloBackend,
    YoloPlateDetector,
    apply_nms,
    compute_ap_at_iou,
    compute_detector_lr,
    compute_iou,
    evaluate_detector_predictions,
    load_detector_checkpoint,
    save_detector_checkpoint,
)
from plate_vision.utils.exceptions import DataIOError, DatasetValidationError, ModelLoadError
from tests.unit.test_data_pipeline import make_png_bytes


class StagnantBackend:
    def __init__(self) -> None:
        self._state: dict[str, Any] = {"step": 0}

    def train_epoch(
        self,
        epoch: int,
        learning_rate: float,
        dataset_yaml: Path,
        config: DetectorConfig,
    ) -> DetectorMetrics:
        _ = (epoch, learning_rate, dataset_yaml, config)
        self._state["step"] = int(self._state["step"]) + 1
        return DetectorMetrics(precision=0.5, recall=0.5, map50=0.5, map50_95=0.4)

    def evaluate(
        self,
        dataset_yaml: Path,
        split: str,
        config: DetectorConfig,
    ) -> DetectorMetrics:
        _ = (dataset_yaml, split, config)
        return DetectorMetrics(precision=0.5, recall=0.5, map50=0.5, map50_95=0.4)

    def get_state(self) -> dict[str, Any]:
        return dict(self._state)

    def load_state(self, state: dict[str, Any]) -> None:
        self._state = dict(state)


def test_iou_nms_and_map_calculations() -> None:
    box1 = BoundingBox(x_min=10.0, y_min=10.0, x_max=50.0, y_max=30.0)
    box2 = BoundingBox(x_min=10.0, y_min=10.0, x_max=50.0, y_max=30.0)
    box_disjoint = BoundingBox(x_min=100.0, y_min=100.0, x_max=120.0, y_max=120.0)

    assert compute_iou(box1, box2) == 1.0
    assert compute_iou(box1, box_disjoint) == 0.0

    preds = [
        DetectionPrediction(bbox=box1, confidence=0.95),
        DetectionPrediction(bbox=box2, confidence=0.80),
        DetectionPrediction(bbox=box_disjoint, confidence=0.20),
    ]
    kept = apply_nms(preds, confidence_threshold=0.25, iou_threshold=0.45, max_detections=5)
    assert len(kept) == 1
    assert kept[0].confidence == 0.95

    assert compute_ap_at_iou({}, {}, 0.5) == (1.0, 1.0, 1.0)
    assert compute_ap_at_iou({"img1": kept}, {}, 0.5) == (0.0, 0.0, 0.0)
    assert compute_ap_at_iou({}, {"img1": [box1]}, 0.5) == (0.0, 0.0, 0.0)

    metrics = evaluate_detector_predictions(
        predictions_by_image={"img1": [DetectionPrediction(bbox=box1, confidence=0.9)]},
        ground_truths_by_image={"img1": [box1]},
    )
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.map50 == 1.0
    assert metrics.map50_95 == 1.0
    assert metrics.fitness == 1.0


def test_lr_schedule_and_checkpoint_io(tmp_path: Path) -> None:
    config = DetectorConfig(epochs=10, warmup_epochs=2, learning_rate=0.01)
    with pytest.raises(ValueError, match="epoch must be >= 1"):
        compute_detector_lr(0, config)

    lr_e1 = compute_detector_lr(1, config)
    lr_e2 = compute_detector_lr(2, config)
    lr_e10 = compute_detector_lr(10, config)
    assert lr_e1 < lr_e2 == 0.01
    assert lr_e10 < lr_e2

    ckpt = DetectorCheckpoint(
        epoch=1,
        best_fitness=0.8,
        metrics=DetectorMetrics(precision=0.8, recall=0.8, map50=0.8, map50_95=0.8),
        model_architecture="yolo11n.pt",
        input_size=640,
        state={"anchor_box": [0.1, 0.2, 0.8, 0.9]},
    )
    ckpt_path = tmp_path / "model.ckpt.json"
    save_detector_checkpoint(ckpt, ckpt_path)
    loaded = load_detector_checkpoint(ckpt_path)
    assert loaded == ckpt

    with pytest.raises(ModelLoadError, match="does not exist"):
        load_detector_checkpoint(tmp_path / "missing.json")

    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{corrupted", encoding="utf-8")
    with pytest.raises(ModelLoadError, match="Corrupted or invalid"):
        load_detector_checkpoint(bad_path)

    with (
        patch.object(Path, "write_text", side_effect=OSError("disk error")),
        pytest.raises(DataIOError, match="Failed to save detector checkpoint"),
    ):
        save_detector_checkpoint(ckpt, tmp_path / "fail.json")


def test_early_stopping_and_yolo_backend_errors(tmp_path: Path) -> None:
    config = DetectorConfig(epochs=10, early_stopping_patience=2)
    trainer = DetectorTrainer(config=config, backend=StagnantBackend())
    summary = trainer.train(dataset_yaml=tmp_path / "dummy.yaml", output_dir=tmp_path / "out")
    assert summary.stopped_early
    assert summary.epochs_completed == 3

    backend = UltralyticsYoloBackend()
    with pytest.raises(DataIOError, match="YOLO dataset configuration not found"):
        backend.evaluate(tmp_path / "nonexistent.yaml", "val", config)

    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("- list_item\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="Invalid YOLO dataset YAML"):
        backend.evaluate(invalid_yaml, "val", config)

    detector = YoloPlateDetector(config=config, checkpoint_path=summary.best_checkpoint_path)
    preds = detector.detect(make_png_bytes(128, 64), image_width=128, image_height=64)
    assert len(preds) == 1
    assert preds[0].confidence >= config.confidence_threshold
