# FastAPI Inference Service

The `plate_vision.serving` package exposes the ONNX-backed ALPR pipeline, video stream analytics, audit log querying, watchlist matching, and drift monitoring over HTTP via FastAPI (`create_app`).

## Endpoints

- `GET /healthz`: Liveness probe returning service health and package version.
- `GET /readyz`: Readiness probe checking whether the ONNX detector and recognizer models are loaded or present on disk (`200 OK` when ready, `503 Service Unavailable` otherwise).
- `GET /metrics`: Prometheus text exposition endpoint reporting request totals by route and status code, domain error counts, total detected plates, and latency histogram buckets (`5ms` to `1000ms`).
- `POST /v1/predict`: Single-image inference from JSON base64 payload (`SinglePredictRequest`), with optional per-request `country_code` override.
- `POST /v1/predict/upload`: Single-image inference from binary upload (`Content-Type` validated against `allowed_mime_types`).
- `POST /v1/predict/batch`: Batch inference (`BatchPredictRequest`) enforcing `max_batch_size`, `max_payload_bytes`, and optional per-item `country_code`.
- `POST /v1/predict/video`: Video sequence inference (`VideoPredictRequest`) executing `VideoStreamProcessor` for multi-object tracking and confidence-weighted temporal voting (`VideoStreamSummary`).
- `GET /v1/audit/recent`: Returns recent `PredictionAuditRecord` entries (`AuditLogResponse`, bounded by `limit`) from the configured `PredictionStorageProtocol` backend (`InMemoryPredictionStore`, `LocalJsonlPredictionStore`, or `SqlitePredictionStore`).
- `POST /v1/watchlist/check`: Evaluates a plate string against `WatchlistMatcher` (`WatchlistCheckRequest` -> `WatchlistCheckResponse`).
- `POST /v1/monitoring/drift`: Evaluates live brightness, contrast, and prediction confidence distributions against a configured `DriftDetector` baseline (`DriftEvaluationRequest` -> `DriftReport`). Returns `503 Service Unavailable` when no baseline is configured.

## Security, Rate Limiting, and Request Correlation

- **API Key Authentication**: When `ServingConfig.api_keys` is non-empty, `SecurityAndAuditMiddleware` enforces constant-time verification of the `X-API-Key` header via `verify_api_key` on all non-probe routes (returning `401 Unauthorized` with `error_code: "UNAUTHORIZED"`).
- **Rate Limiting**: `SlidingWindowRateLimiter` tracks per-client request timestamps over a 60-second sliding window (`ServingConfig.rate_limit_requests_per_minute`). Exceeded clients receive `429 Too Many Requests` (`RATE_LIMIT_EXCEEDED`), `Retry-After`, and `X-RateLimit-Remaining: 0`.
- **Correlation Tracing**: Every HTTP response includes an `X-Request-ID` header (propagating an incoming `X-Request-ID` or generating a UUID4 hex identifier).

## Structured Error Mapping

All domain exceptions inheriting from `PlateVisionError` return a `StructuredErrorResponse` JSON body (`error_code`, `message`, `details`):
- `InvalidInputError` / `CorruptedImageError` -> `422 Unprocessable Entity`
- `EmptyDetectionError` -> `404 Not Found`
- `ModelLoadError` -> `503 Service Unavailable`
- `InferenceTimeoutError` -> `504 Gateway Timeout`
