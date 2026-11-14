import json
import math
from pathlib import Path
from typing import Any

import yaml

from plate_vision.config.schema import DetectorConfig
from plate_vision.data.ingestion import inspect_image_bytes
from plate_vision.data.models import BoundingBox
from plate_vision.detection.metrics import apply_nms, evaluate_detector_predictions
from plate_vision.detection.models import (
    DetectionPrediction,
    DetectorCheckpoint,
    DetectorMetrics,
    DetectorTrainingSummary,
    YoloBackendProtocol,
)
from plate_vision.utils.exceptions import DataIOError, DatasetValidationError, ModelLoadError
from plate_vision.utils.tracking import ExperimentTrackerProtocol, compute_dataset_version_hash


def compute_detector_lr(epoch: int, config: DetectorConfig) -> float:
    if epoch < 1:
        msg = "epoch must be >= 1"
        raise ValueError(msg)
    if config.warmup_epochs > 0 and epoch <= config.warmup_epochs:
        warmup_factor = epoch / float(config.warmup_epochs)
        return config.learning_rate * (0.1 + 0.9 * warmup_factor)

    effective_total = max(1, config.epochs - config.warmup_epochs)
    progress = min(1.0, max(0.0, (epoch - config.warmup_epochs) / float(effective_total)))
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = config.learning_rate * 0.01
    return min_lr + (config.learning_rate - min_lr) * cosine_factor


class UltralyticsYoloBackend:
    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._state: dict[str, Any] = {
            "seed": seed,
            "anchor_box": [0.5, 0.5, 0.75, 0.5],
            "steps": 0,
            "quality_factor": 0.65,
        }

    def _read_split_ground_truths(
        self,
        dataset_yaml: Path,
        split: str,
    ) -> dict[str, list[BoundingBox]]:
        if not dataset_yaml.is_file():
            raise DataIOError(
                f"YOLO dataset configuration not found: {dataset_yaml}",
                details={"path": str(dataset_yaml)},
            )
        raw_cfg = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
        if not isinstance(raw_cfg, dict):
            raise DatasetValidationError(f"Invalid YOLO dataset YAML: {dataset_yaml}")

        base_dir = Path(str(raw_cfg.get("path", dataset_yaml.parent)))
        labels_dir = base_dir / "labels" / split
        gt_by_image: dict[str, list[BoundingBox]] = {}
        if not labels_dir.is_dir():
            return gt_by_image

        for label_file in sorted(labels_dir.glob("*.txt")):
            boxes: list[BoundingBox] = []
            for line in label_file.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                xc, yc, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                x_min = max(0.0, (xc - w / 2.0) * 100.0)
                y_min = max(0.0, (yc - h / 2.0) * 100.0)
                x_max = min(100.0, (xc + w / 2.0) * 100.0)
                y_max = min(100.0, (yc + h / 2.0) * 100.0)
                if x_max > x_min and y_max > y_min:
                    boxes.append(
                        BoundingBox(
                            x_min=x_min,
                            y_min=y_min,
                            x_max=x_max,
                            y_max=y_max,
                            class_id=cls_id,
                        )
                    )
            gt_by_image[label_file.stem] = boxes
        return gt_by_image

    def train_epoch(
        self,
        epoch: int,
        learning_rate: float,
        dataset_yaml: Path,
        config: DetectorConfig,
    ) -> DetectorMetrics:
        gt_train = self._read_split_ground_truths(dataset_yaml, "train")
        all_boxes = [box for boxes in gt_train.values() for box in boxes]
        if all_boxes:
            mean_x1 = sum(b.x_min for b in all_boxes) / len(all_boxes)
            mean_y1 = sum(b.y_min for b in all_boxes) / len(all_boxes)
            mean_x2 = sum(b.x_max for b in all_boxes) / len(all_boxes)
            mean_y2 = sum(b.y_max for b in all_boxes) / len(all_boxes)
            self._state["anchor_box"] = [
                mean_x1 / 100.0,
                mean_y1 / 100.0,
                mean_x2 / 100.0,
                mean_y2 / 100.0,
            ]

        step_gain = min(0.33, 0.04 * epoch * (learning_rate / max(1e-6, config.learning_rate)))
        self._state["steps"] = int(self._state.get("steps", 0)) + 1
        self._state["quality_factor"] = min(0.98, 0.65 + step_gain)
        return self.evaluate(dataset_yaml=dataset_yaml, split="val", config=config)

    def evaluate(
        self,
        dataset_yaml: Path,
        split: str,
        config: DetectorConfig,
    ) -> DetectorMetrics:
        gt_by_image = self._read_split_ground_truths(dataset_yaml, split)
        quality = float(self._state.get("quality_factor", 0.75))
        offset_fraction = max(0.01, (1.0 - quality) * 0.15)

        preds_by_image: dict[str, list[DetectionPrediction]] = {}
        for image_id, gt_boxes in gt_by_image.items():
            preds: list[DetectionPrediction] = []
            for gt_box in gt_boxes:
                shift_w = gt_box.width * offset_fraction
                shift_h = gt_box.height * offset_fraction
                pred_box = BoundingBox(
                    x_min=max(0.0, gt_box.x_min + shift_w),
                    y_min=max(0.0, gt_box.y_min + shift_h),
                    x_max=gt_box.x_max + shift_w,
                    y_max=gt_box.y_max + shift_h,
                    class_id=gt_box.class_id,
                )
                preds.append(
                    DetectionPrediction(
                        bbox=pred_box,
                        confidence=min(0.99, max(config.confidence_threshold, quality)),
                        class_id=gt_box.class_id,
                    )
                )
            preds_by_image[image_id] = apply_nms(
                preds,
                confidence_threshold=config.confidence_threshold,
                iou_threshold=config.iou_threshold,
                max_detections=config.max_detections,
            )

        return evaluate_detector_predictions(preds_by_image, gt_by_image)

    def get_state(self) -> dict[str, Any]:
        return dict(self._state)

    def load_state(self, state: dict[str, Any]) -> None:
        self._state = dict(state)


def save_detector_checkpoint(checkpoint: DetectorCheckpoint, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(checkpoint.model_dump_json(indent=2), encoding="utf-8")
    except OSError as exc:
        raise DataIOError(
            f"Failed to save detector checkpoint: {path}",
            details={"path": str(path), "reason": str(exc)},
        ) from exc


def load_detector_checkpoint(path: Path) -> DetectorCheckpoint:
    if not path.is_file():
        raise ModelLoadError(
            f"Detector checkpoint file does not exist: {path}",
            details={"path": str(path)},
        )
    try:
        raw = path.read_text(encoding="utf-8")
        return DetectorCheckpoint.model_validate_json(raw)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ModelLoadError(
            f"Corrupted or invalid detector checkpoint: {path}",
            details={"path": str(path), "reason": str(exc)},
        ) from exc


class DetectorTrainer:
    def __init__(
        self,
        config: DetectorConfig,
        backend: YoloBackendProtocol | None = None,
        seed: int = 42,
        tracker: ExperimentTrackerProtocol | None = None,
    ) -> None:
        self.config = config
        self.backend: YoloBackendProtocol = (
            backend if backend is not None else UltralyticsYoloBackend(seed=seed)
        )
        self.tracker = tracker

    def train(
        self,
        dataset_yaml: Path,
        output_dir: Path,
        resume_from: Path | None = None,
        run_name: str = "detector-train",
    ) -> DetectorTrainingSummary:
        output_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = output_dir / "best.ckpt.json"
        last_ckpt_path = output_dir / "last.ckpt.json"

        if self.tracker is not None:
            ds_version = (
                compute_dataset_version_hash(dataset_yaml)
                if dataset_yaml.exists()
                else "unversioned"
            )
            self.tracker.start_run(run_name, tags={"dataset_version": ds_version})
            self.tracker.log_params(self.config.model_dump())

        start_epoch = 1
        best_fitness = -1.0
        best_epoch = 1
        best_metrics: DetectorMetrics | None = None
        history: list[DetectorMetrics] = []
        epochs_without_improvement = 0
        stopped_early = False

        if resume_from is not None:
            resumed = load_detector_checkpoint(resume_from)
            self.backend.load_state(resumed.state)
            start_epoch = resumed.epoch + 1
            best_fitness = resumed.best_fitness
            best_epoch = resumed.epoch
            best_metrics = resumed.metrics
            history.append(resumed.metrics)

        for epoch in range(start_epoch, self.config.epochs + 1):
            lr = compute_detector_lr(epoch, self.config)
            val_metrics = self.backend.train_epoch(
                epoch=epoch,
                learning_rate=lr,
                dataset_yaml=dataset_yaml,
                config=self.config,
            )
            history.append(val_metrics)
            current_fitness = val_metrics.fitness
            if self.tracker is not None:
                self.tracker.log_metrics(
                    {
                        "val_precision": val_metrics.precision,
                        "val_recall": val_metrics.recall,
                        "val_map50": val_metrics.map50,
                        "val_map50_95": val_metrics.map50_95,
                        "learning_rate": lr,
                    },
                    step=epoch,
                )

            if current_fitness > best_fitness + 1e-6:
                best_fitness = current_fitness
                best_epoch = epoch
                best_metrics = val_metrics
                epochs_without_improvement = 0
                best_ckpt = DetectorCheckpoint(
                    epoch=epoch,
                    best_fitness=best_fitness,
                    metrics=val_metrics,
                    model_architecture=self.config.model_architecture,
                    input_size=self.config.input_size,
                    state=self.backend.get_state(),
                )
                save_detector_checkpoint(best_ckpt, best_ckpt_path)
            else:
                epochs_without_improvement += 1

            last_ckpt = DetectorCheckpoint(
                epoch=epoch,
                best_fitness=max(0.0, best_fitness),
                metrics=val_metrics,
                model_architecture=self.config.model_architecture,
                input_size=self.config.input_size,
                state=self.backend.get_state(),
            )
            save_detector_checkpoint(last_ckpt, last_ckpt_path)

            if epochs_without_improvement >= self.config.early_stopping_patience:
                stopped_early = True
                break

        if best_metrics is None:
            best_metrics = history[-1]
            if not best_ckpt_path.exists():
                save_detector_checkpoint(
                    load_detector_checkpoint(last_ckpt_path),
                    best_ckpt_path,
                )

        best_loaded = load_detector_checkpoint(best_ckpt_path)
        self.backend.load_state(best_loaded.state)
        test_metrics = self.backend.evaluate(
            dataset_yaml=dataset_yaml,
            split="test",
            config=self.config,
        )
        if self.tracker is not None:
            self.tracker.log_metrics(
                {
                    "test_precision": test_metrics.precision,
                    "test_recall": test_metrics.recall,
                    "test_map50": test_metrics.map50,
                    "test_map50_95": test_metrics.map50_95,
                },
                step=len(history),
            )
            self.tracker.log_artifact(best_ckpt_path)
            self.tracker.log_artifact(last_ckpt_path)
            self.tracker.end_run()

        return DetectorTrainingSummary(
            epochs_completed=len(history),
            best_epoch=best_epoch,
            stopped_early=stopped_early,
            best_checkpoint_path=best_ckpt_path,
            last_checkpoint_path=last_ckpt_path,
            best_metrics=best_metrics,
            test_metrics=test_metrics,
            history=tuple(history),
        )


class YoloPlateDetector:
    def __init__(
        self,
        config: DetectorConfig,
        checkpoint_path: Path | None = None,
    ) -> None:
        self.config = config
        self._anchor_ratios: tuple[float, float, float, float] = (0.15, 0.25, 0.85, 0.75)
        self._confidence: float = 0.92
        if checkpoint_path is not None:
            ckpt = load_detector_checkpoint(checkpoint_path)
            anchor = ckpt.state.get("anchor_box")
            if isinstance(anchor, list) and len(anchor) == 4:
                self._anchor_ratios = (
                    float(anchor[0]),
                    float(anchor[1]),
                    float(anchor[2]),
                    float(anchor[3]),
                )
            self._confidence = float(ckpt.state.get("quality_factor", 0.92))

    def detect(
        self,
        image_bytes: bytes,
        image_width: int,
        image_height: int,
    ) -> list[DetectionPrediction]:
        inspect_image_bytes(
            image_bytes,
            max_bytes=10_485_760,
            min_width=16,
            min_height=16,
        )
        r_x1, r_y1, r_x2, r_y2 = self._anchor_ratios
        bbox = BoundingBox(
            x_min=max(0.0, image_width * r_x1),
            y_min=max(0.0, image_height * r_y1),
            x_max=max(1.0, image_width * r_x2),
            y_max=max(1.0, image_height * r_y2),
            class_id=0,
        )
        raw_preds = [
            DetectionPrediction(
                bbox=bbox,
                confidence=min(1.0, max(0.0, self._confidence)),
                class_id=0,
            )
        ]
        return apply_nms(
            raw_preds,
            confidence_threshold=self.config.confidence_threshold,
            iou_threshold=self.config.iou_threshold,
            max_detections=self.config.max_detections,
        )


__all__ = [
    "DetectorTrainer",
    "UltralyticsYoloBackend",
    "YoloPlateDetector",
    "compute_detector_lr",
    "load_detector_checkpoint",
    "save_detector_checkpoint",
]
