from plate_vision.detection.cli import build_parser, main, run_train_detector
from plate_vision.detection.metrics import (
    apply_nms,
    compute_ap_at_iou,
    compute_iou,
    evaluate_detector_predictions,
)
from plate_vision.detection.models import (
    DetectionPrediction,
    DetectorCheckpoint,
    DetectorMetrics,
    DetectorTrainingSummary,
    PlateDetectorProtocol,
    YoloBackendProtocol,
)
from plate_vision.detection.trainer import (
    DetectorTrainer,
    UltralyticsYoloBackend,
    YoloPlateDetector,
    compute_detector_lr,
    load_detector_checkpoint,
    save_detector_checkpoint,
)

__all__ = [
    "DetectionPrediction",
    "DetectorCheckpoint",
    "DetectorMetrics",
    "DetectorTrainer",
    "DetectorTrainingSummary",
    "PlateDetectorProtocol",
    "UltralyticsYoloBackend",
    "YoloBackendProtocol",
    "YoloPlateDetector",
    "apply_nms",
    "build_parser",
    "compute_ap_at_iou",
    "compute_detector_lr",
    "compute_iou",
    "evaluate_detector_predictions",
    "load_detector_checkpoint",
    "main",
    "run_train_detector",
    "save_detector_checkpoint",
]
