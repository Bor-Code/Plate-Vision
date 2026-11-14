# Development Guide

## Project Structure

The repository follows the standard Python `src` layout with the top-level package `plate_vision`:

- `src/plate_vision/config`: Typed configuration loading and validation.
- `src/plate_vision/data`: Dataset ingestion, validation, splitting, synthetic generation, and augmentation.
- `src/plate_vision/detection`: License plate detection training and inference abstractions.
- `src/plate_vision/recognition`: License plate character recognition (CRNN + CTC) models and decoders.
- `src/plate_vision/pipeline`: End-to-end two-stage ALPR pipeline, rectification, and post-processing.
- `src/plate_vision/evaluation`: Metric computation and error analysis for detection, recognition, and end-to-end flows.
- `src/plate_vision/export`: ONNX export, parity verification, and quantization utilities.
- `src/plate_vision/serving`: FastAPI inference service, middleware, and telemetry.
- `src/plate_vision/video`: Multi-object tracking and temporal voting across video streams.
- `src/plate_vision/utils`: Shared logging, exceptions, and helper utilities.

## Environment Setup

Install Python 3.11 and synchronize dependencies using `uv`:

```bash
uv sync --dev
uv run pre-commit install
```

## Quality Gates

Every change must pass all static analysis, formatting, type checking, and test suites before submission:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest
```
