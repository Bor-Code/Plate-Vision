# Plate-Vision

Plate-Vision is a production-grade, end-to-end Automatic License Plate Recognition (ALPR/ANPR) system built with Python 3.11, typed configuration (`pydantic-settings` + YAML), structured JSON logging (`structlog`), MLflow experiment tracking, DVC dataset versioning, ONNX Runtime serving, and FastAPI.

## Key Capabilities

- **Two-Stage ALPR Architecture**:
  - **Stage 1 (Plate Detector)**: YOLO-based license plate bounding box detector with Non-Maximum Suppression (`apply_nms`) and 101-point interpolated `mAP@0.5` / `mAP@0.5:0.95` evaluation.
  - **Intermediate Geometric & Photometric Normalization**: 4-point perspective homography rectification (`rectify_perspective_crop`), low-light contrast stretching, and horizontal motion-blur sharpening (`normalize_plate_crop`).
  - **Stage 2 (Plate Recognizer)**: Custom CRNN (CNN feature extractor + Bidirectional LSTM + Connectionist Temporal Classification loss) supporting both greedy and prefix beam search decoding (`ctc_beam_search_decode`).
  - **Multi-Country Profile Post-Processing**: Pluggable `CountryProfileProtocol` registry with 8 built-in profiles (`TR`, `DE`, `GB`, `AZ`, `FR`, `IT`, `ES`, `NL`) and automatic multi-jurisdiction resolution (`MultiCountryPostProcessor`) with positional optical confusion correction (`0/O`, `1/I`, `8/B`, `5/S`, `2/Z`).
- **Privacy, Security, and Operational Intelligence**:
  - **Privacy Redaction (`redact_regions`)**: Anonymizes detected license plate bounding boxes using `BLUR`, `PIXELATE`, or `SOLID_FILL` modes.
  - **Watchlist Alerting (`WatchlistMatcher`)**: Matches normalized plates against `ALLOWLIST`, `BLOCKLIST`, `VIP`, and `STOLEN` lists with exact and bounded Levenshtein matching.
  - **Active Learning (`ActiveLearningSampler`)**: Ranks low-confidence, format-invalid, or confusion-corrected predictions and exports deduplicated JSONL manifests for labeling.
  - **Calibration & Drift Monitoring**: Post-hoc `TemperatureScaler`, Expected Calibration Error (`compute_expected_calibration_error`), and Population Stability Index (`DriftDetector`) monitoring.
- **Reproducible Data, Training & CLI Tooling**:
  - Six registered CLI entrypoints: `prepare_dataset`, `train_detector`, `train_recognizer`, `export_models`, `evaluate_pipeline`, and `predict_plate`.
  - Ingestion validation, SHA-256 deduplication, deterministic stratified splitting, synthetic plate generation, and configurable augmentation pipelines.
- **Production Serving, Audit Storage & Video Analytics**:
  - ONNX export, numerical parity verification, and dynamic `int8` weight quantization (`plate_vision.export`).
  - FastAPI service (`plate_vision.serving`) exposing `/v1/predict`, `/v1/predict/upload`, `/v1/predict/batch`, `/v1/predict/video`, `/v1/audit/recent`, `/v1/watchlist/check`, `/v1/monitoring/drift`, `/healthz`, `/readyz`, and `/metrics`, protected by optional `X-API-Key` verification, `SlidingWindowRateLimiter`, and `X-Request-ID` tracing.
  - Content-addressable `PredictionCache` and pluggable `PredictionStorageProtocol` audit stores (`InMemoryPredictionStore`, `LocalJsonlPredictionStore`, `SqlitePredictionStore`).
  - Multi-object IoU tracking (`PlateTracker`), confidence-weighted temporal voting (`TemporalVoter`), and vehicle speed / dwell-time analytics (`VideoTrafficAnalyzer`).

## Repository Layout

```text
src/plate_vision/
  config/       Typed Pydantic settings and YAML configuration loader
  data/         Ingestion, deduplication, splits, exporters, synthetic generator, country profiles, active learning, CLI
  detection/    Detector training loop, NMS, mAP evaluation, CLI
  recognition/  CRNN model, charset encoder, CTC greedy/beam decoding, trainer, CLI
  pipeline/     Two-stage inference runner, rectification, post-processing, privacy redaction, watchlist, CLI
  evaluation/   Evaluation suite, error analysis artifacts, ECE calibration, PSI drift detection, CLI
  export/       ONNX exporter, ONNX Runtime wrappers, parity verification, dynamic int8 quantization, CLI
  serving/      FastAPI endpoints, API key auth, sliding-window rate limiter, Prometheus metrics
  video/        Multi-object plate tracker, temporal voting, and vehicle speed / dwell-time analytics
  utils/        Structured logging, domain exception hierarchy, MLflow tracking, prediction cache/stores, release helpers
configs/        Default YAML configurations for data, detection, recognition, and serving
docker/         Multi-stage non-root Dockerfile and docker-compose.yml
docs/           Architecture, datasets, model card, benchmarks, and component guides
tests/          Unit and integration test suites (>= 90% branch and statement coverage)
```

## Quickstart

### 1. Environment Setup

```bash
uv sync --locked --dev
```

### 2. Quality Gates

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest
```

### 3. CLI Workflows

```bash
uv run prepare_dataset --output-dir data/prepared --country-code TR --synthetic-count 30 --seed 42
uv run train_detector --config configs/detection/default.yaml --dataset-yaml data/prepared/yolo/dataset.yaml --output-dir checkpoints/detector
uv run train_recognizer --config configs/recognition/default.yaml --train-manifest data/prepared/recognizer/recognizer_train.jsonl --val-manifest data/prepared/recognizer/recognizer_val.jsonl --test-manifest data/prepared/recognizer/recognizer_test.jsonl --output-dir checkpoints/recognizer
uv run export_models --output-dir artifacts/models --quantize
uv run evaluate_pipeline --detector-onnx artifacts/models/detector.onnx --recognizer-onnx artifacts/models/recognizer.onnx --test-manifest data/prepared/recognizer/recognizer_test.jsonl --output-dir artifacts/eval
uv run predict_plate --detector-onnx artifacts/models/detector.onnx --recognizer-onnx artifacts/models/recognizer.onnx --image sample.png --output-json predictions.json
```

### 4. Containerized Inference Service

```bash
docker compose -f docker/docker-compose.yml up --build -d
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
```

## Documentation Index

- [Architecture Overview](docs/architecture.md)
- [Model Card](docs/model_card.md)
- [Datasets and DVC Pipeline](docs/datasets.md)
- [Benchmark Results](docs/benchmarks.md)
- [Configuration System](docs/configuration.md)
- [Detector Training and Evaluation](docs/detection.md)
- [Recognizer CRNN and Decoding](docs/recognition.md)
- [Experiment Tracking](docs/experiment_tracking.md)
- [Evaluation Suite and Error Analysis](docs/evaluation.md)
- [Two-Stage Inference Pipeline](docs/pipeline.md)
- [ONNX Export and Quantization](docs/export.md)
- [FastAPI Serving](docs/serving.md)
- [Container Deployment](docs/deployment.md)
- [Video Tracking and Temporal Voting](docs/video.md)
- [Contributing Guide](CONTRIBUTING.md)

## License

Released under the MIT License. See [LICENSE](LICENSE).
