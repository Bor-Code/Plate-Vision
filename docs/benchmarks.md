# Benchmark Results

All benchmark values reported in this document were measured empirically by executing the Plate-Vision training, ONNX export, parity verification, and end-to-end evaluation pipelines (`plate_vision.evaluation`) on Python `3.11.18` (`win32`, `AMD64`, CPU execution, `seed=42`).

## 1. Detector Evaluation (`DetectorTrainer` + `OnnxDetectorRuntime`)

Measured on the synthetic Turkish (`TR`) evaluation split (`input_size=640`, `iou_threshold=0.50` to `0.95`):

| Metric | Measured Value |
| --- | --- |
| `precision` (`IoU=0.50`) | `1.000000` |
| `recall` (`IoU=0.50`) | `1.000000` |
| `mAP@0.50` | `1.000000` |
| `mAP@0.50:0.95` | `0.700000` |

## 2. Recognizer Evaluation (`CRNNModel` + `ctc_beam_search_decode`)

Measured across two reproducible evaluation regimes (`image_height=32`, `image_width=128`, `beam_width=5`, `seed=42`):

| Evaluation Split Regime | Exact-Match Plate Accuracy | Character Error Rate (`CER`) | Positional Accuracy | Validation CTC Loss |
| --- | --- | --- | --- | --- |
| Closed-Set Noisy Fleet (`noise_std=1.5`) | `1.000000` | `0.000000` | `1.000000` | `0.001655` |
| Unseen Random Synthetic (`perspective=0.06`, `noise=4.0`, `blur=0.25`) | `0.000000` | `0.965517` | `0.096774` | `8.973916` |

## 3. ONNX Numerical Parity and Model Size

Measured using `verify_detector_parity` and `verify_recognizer_parity` (`opset_version=17`):

| Artifact | Precision | File Size (Bytes) | `max_abs_diff` | `mean_abs_diff` | Parity Passed |
| --- | --- | --- | --- | --- | --- |
| `detector.onnx` | `fp32` | `212` | `0.00000000` | `0.00000000` | `True` |
| `recognizer.onnx` | `fp32` | `10965` | `0.00000000` | `0.00000000` | `True` |
| `recognizer_int8.onnx` | `int8` (dynamic) | `9561` | `0.00003090` | `0.00000515` | `True` (`tol=0.05`) |

## 4. End-to-End Pipeline Latency and Throughput

Measured on full end-to-end `PlateVisionPipeline.predict_bytes` execution (PNG decode + detection + NMS + crop extraction + low-light/blur normalization + ONNX CRNN recognition + `TR` post-processing):

| Workload | `plate_level_accuracy` | `latency_p50_ms` | `latency_p95_ms` | `throughput_fps` |
| --- | --- | --- | --- | --- |
| Closed-Set Noisy Fleet (`128x32` crops) | `1.000000` | `11.8443` | `11.9528` | `84.3701` |
| Full Pipeline with Synthetic Degradations | `0.000000` | `16.9195` | `18.2297` | `58.2282` |
