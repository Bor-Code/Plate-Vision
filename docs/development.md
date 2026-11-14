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

## Continuous Integration

The GitHub Actions workflow in `.github/workflows/ci.yml` runs on every push to `main` and every pull request targeting `main`. It provisions Python 3.11 via `astral-sh/setup-uv`, caches the `uv` environment keyed on `uv.lock`, installs locked development dependencies, executes `ruff check`, `ruff format --check`, `mypy src tests`, and `pytest` with coverage enforcement, and uploads `coverage.xml` as a build artifact.
