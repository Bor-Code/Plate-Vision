# Evaluation Suite and Error Analysis

Plate-Vision provides a unified evaluation suite (`plate_vision.evaluation`) that measures detector performance, recognizer accuracy, end-to-end plate recognition accuracy, latency percentiles, throughput, and error analysis artifacts.

## Metrics Computed

### 1. Detector Metrics (`DetectorMetrics`)
- `precision`: Detection precision at IoU threshold 0.50.
- `recall`: Detection recall at IoU threshold 0.50.
- `map50`: 101-point interpolated mean Average Precision at IoU 0.50.
- `map50_95`: Mean Average Precision averaged across IoU thresholds `0.50` to `0.95` in steps of `0.05`.

### 2. Recognizer Metrics (`RecognizerEpochMetrics`)
- `exact_match_accuracy`: Proportion of plates where the entire predicted string matches the ground-truth string.
- `character_error_rate`: Levenshtein edit distance divided by total ground-truth characters.
- `positional_accuracy`: Character match rate at aligned character indices.

### 3. End-to-End Metrics (`EndToEndMetrics`)
- `plate_level_accuracy`: End-to-end exact match rate across full pipeline predictions.
- `latency_p50_ms`: Median (50th percentile) per-image inference latency in milliseconds.
- `latency_p95_ms`: 95th percentile per-image inference latency in milliseconds.
- `throughput_fps`: Sustained frames per second computed from total measured execution time.

## Error Analysis Artifacts

Calling `save_evaluation_artifacts(report, output_dir)` writes three JSON artifacts ready for MLflow tracking:
- `evaluation_report.json`: Complete serialized `EvaluationReport`.
- `confusion_pairs.json`: Ranked character substitution pairs (`target_char`, `predicted_char`, `count`).
- `worst_samples.json`: Samples sorted by highest Character Error Rate (CER) and edit distance.
