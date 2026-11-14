# License Plate Detection Pipeline

## Architecture

Stage 1 of `plate-vision` detects license plate bounding boxes using the `PlateDetectorProtocol` and `YoloBackendProtocol` abstractions (`plate_vision.detection`):

- **`YoloPlateDetector`**: Executes bounding-box detection followed by confidence filtering and greedy IoU Non-Maximum Suppression (`apply_nms`).
- **`DetectorTrainer`**: Orchestrates multi-epoch detector training with linear warmup and cosine learning-rate decay (`compute_detector_lr`), mixed-precision configuration, early stopping based on composite fitness (`0.1 * mAP@0.5 + 0.9 * mAP@0.5:0.95`), `best.ckpt.json` and `last.ckpt.json` persistence, and checkpoint resume support.

## Evaluation Metrics

Detector predictions are evaluated via 101-point interpolated Average Precision (`plate_vision.detection.metrics`):

- `precision`: Precision at IoU threshold `0.50`.
- `recall`: Recall at IoU threshold `0.50`.
- `map50`: Mean Average Precision at IoU threshold `0.50` (`mAP@0.5`).
- `map50_95`: Mean Average Precision averaged across IoU thresholds `0.50` to `0.95` in `0.05` increments (`mAP@0.5:0.95`).

## CLI Usage

Train and evaluate the detector from a YAML configuration and exported YOLO dataset:

```bash
uv run train_detector \
  --config configs/detection/default.yaml \
  --dataset-yaml data/processed/yolo/dataset.yaml \
  --output-dir checkpoints/detector \
  --seed 42
```
