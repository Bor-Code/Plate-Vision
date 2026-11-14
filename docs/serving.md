# FastAPI Inference Service

The `plate_vision.serving` package exposes the ONNX-backed ALPR pipeline over HTTP via FastAPI.

## Endpoints

- `GET /healthz`: Liveness probe returning service health and package version.
- `GET /readyz`: Readiness probe checking whether the ONNX detector and recognizer models are loaded or present on disk (`200 OK` when ready, `503 Service Unavailable` otherwise).
- `GET /metrics`: Prometheus text exposition endpoint reporting request totals by route and status code, domain error counts, total detected plates, and latency histogram buckets (`5ms` to `1000ms`).
- `POST /v1/predict`: Single-image inference from JSON base64 payload (`SinglePredictRequest`).
- `POST /v1/predict/upload`: Single-image inference from `multipart/form-data` binary upload.
- `POST /v1/predict/batch`: Batch inference (`BatchPredictRequest`) enforcing `max_batch_size` and `max_payload_bytes`.

## Structured Error Mapping

All domain exceptions inheriting from `PlateVisionError` return a `StructuredErrorResponse` JSON body (`error_code`, `message`, `details`):
- `InvalidInputError` / `CorruptedImageError` -> `422 Unprocessable Entity`
- `EmptyDetectionError` -> `404 Not Found`
- `ModelLoadError` -> `503 Service Unavailable`
- `InferenceTimeoutError` -> `504 Gateway Timeout`
