# Plate-Vision

Plate-Vision is a production-grade, end-to-end Automatic License Plate Recognition (ALPR/ANPR) system built with Python 3.11, typed configuration (`pydantic-settings` + YAML), structured JSON logging (`structlog`), MLflow experiment tracking, DVC dataset versioning, ONNX Runtime serving, and FastAPI.

## Key Capabilities

- **Two-Stage ALPR Architecture**:
  - **Stage 1 (Plate Detector)**: YOLO-based license plate bounding box detector with Non-Maximum Suppression (`apply_nms`) and 101-point interpolated `mAP@0.5` / `mAP@0.5:0.95` evaluation.
  - **Intermediate Geometric & Photometric Normalization**: 4-point perspective homography rectification (`rectify_perspective_crop`), low-light contrast stretching, and horizontal motion-blur sharpening (`normalize_plate_crop`).
  - **Stage 2 (Plate Recognizer)**: Custom CRNN (CNN feature extractor + Bidirectional LSTM + Connectionist Temporal Classification loss) supporting both greedy and prefix beam search decoding (`ctc_beam_search_decode`).
  - **Country Profile Post-Processing**: Pluggable `CountryProfileProtocol` registry with built-in Turkey (`TR`) license plate validation (province codes `01`-`81`, letter groups excluding `Q`, `W`, `X`, and positional optical confusion correction for `0/O`, `1/I`, `8/B`, `5/S`, `2/Z`).
- **Reproducible Data & Training Pipelines**:
  - Ingestion validation, SHA-256 deduplication, deterministic stratified `train`/`val`/`test` splitting, YOLO and recognizer manifest exporters, synthetic plate generation, and configurable augmentation pipelines.
  - Config-driven CLI entry points (`train_detector`, `train_recognizer`) with mixed-precision gradient scaling (`AmpGradScaler`), cosine learning rate warmup/decay, early stopping, resumable checkpointing, and MLflow experiment tracking.
- **Production Serving & Video Stream Processing**:
  - ONNX export, numerical parity verification, and dynamic `int8` weight quantization (`plate_vision.export`).
  - FastAPI service (`plate_vision.serving`) exposing single (`/v1/predict`, `/v1/predict/upload`) and batch (`/v1/predict/batch`) endpoints, `/healthz` and `/readyz` probes, `/metrics` Prometheus exposition, and structured error responses.
  - Multi-object IoU tracking (`PlateTracker`) and confidence-weighted temporal voting (`TemporalVoter`, `VideoStreamProcessor`) for video streams.

## Repository Layout

```text
src/plate_vision/
  config/       Typed Pydantic settings and YAML configuration loader
  data/         Ingestion, deduplication, stratified splits, exporters, synthetic generator, country profiles
  detection/    Detector training loop, NMS, mAP evaluation, CLI
  recognition/  CRNN model, charset encoder, CTC greedy/beam decoding, trainer, CLI
  pipeline/     Two-stage inference runner, perspective rectification, post-processing
  evaluation/   Detector, recognizer, and end-to-end evaluation suite and error analysis artifacts
  export/       ONNX exporter, ONNX Runtime wrappers, parity verification, dynamic int8 quantization
  serving/      FastAPI endpoints, health/readiness probes, Prometheus metrics
  video/        Multi-object plate tracker and confidence-weighted temporal voting
  utils/        Structured logging, domain exception hierarchy, MLflow tracking, release utilities
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

### 3. Training

```bash
uv run train_detector --config configs/detection/default.yaml --dataset-yaml data/processed/yolo/dataset.yaml --output-dir checkpoints/detector
uv run train_recognizer --config configs/recognition/default.yaml --train-manifest data/processed/recognizer/train.jsonl --val-manifest data/processed/recognizer/val.jsonl --test-manifest data/processed/recognizer/test.jsonl --output-dir checkpoints/recognizer
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
