# Configuration, Logging, and Error Handling

## Overview

`plate_vision` enforces strict runtime validation and observability across training, evaluation, and inference pipelines through three foundational modules:

1. **Typed Configuration (`plate_vision.config`)**: Pydantic v2 and `pydantic-settings` schemas backed by YAML files under `configs/` and environment variable overrides.
2. **Structured Logging (`plate_vision.utils.logging`)**: `structlog` integration producing machine-parseable JSON or formatted key-value logs with contextual metadata and zero `print` statements.
3. **Domain Exception Hierarchy (`plate_vision.utils.exceptions`)**: Deterministic error classes with standardized error codes and serializable context payloads.

## Configuration Loading

Default YAML configurations are organized by subsystem:

- `configs/data/default.yaml`: Seed, country profile, split ratios, and image constraints.
- `configs/detection/default.yaml`: Detector architecture, input resolution, thresholds, and training hyperparameters.
- `configs/recognition/default.yaml`: CRNN dimensions, sequence modeling parameters, beam width, and optimizer settings.
- `configs/serving/default.yaml`: FastAPI bind address, payload limits, timeouts, and ONNX artifact paths.

Environment variables use the `PLATE_VISION_` prefix and `__` nested delimiter (for example, `PLATE_VISION_SERVING__PORT=8080`).

## Exception Hierarchy

All custom exceptions inherit from `PlateVisionError` and expose `error_code`, `message`, `details`, and `to_dict()`:

- `ConfigurationError` (`CONFIGURATION_ERROR`)
- `DataIOError` (`DATA_IO_ERROR`)
- `CorruptedImageError` (`CORRUPTED_IMAGE_ERROR`)
- `EmptyDetectionError` (`EMPTY_DETECTION_ERROR`)
- `ModelLoadError` (`MODEL_LOAD_ERROR`)
- `InvalidInputError` (`INVALID_INPUT_ERROR`)
- `InferenceTimeoutError` (`INFERENCE_TIMEOUT_ERROR`)
- `DatasetValidationError` (`DATASET_VALIDATION_ERROR`)
