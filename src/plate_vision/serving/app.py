import base64
import binascii
import json
import time
import uuid
from collections.abc import Mapping, MutableMapping
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from plate_vision import __version__
from plate_vision.config.schema import ServingConfig
from plate_vision.data.profiles import get_country_profile
from plate_vision.evaluation.monitoring import DriftDetector, DriftReport
from plate_vision.export.runtime import OnnxDetectorRuntime, OnnxRecognizerRuntime
from plate_vision.pipeline.runner import PipelineInferenceResult, PlateVisionPipeline
from plate_vision.pipeline.watchlist import WatchlistMatcher
from plate_vision.serving.metrics import PrometheusMetricsRegistry
from plate_vision.serving.schemas import (
    AuditLogResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    DriftEvaluationRequest,
    HealthStatusResponse,
    ReadinessStatusResponse,
    SinglePredictRequest,
    StructuredErrorResponse,
    VideoPredictRequest,
    WatchlistCheckRequest,
    WatchlistCheckResponse,
)
from plate_vision.serving.security import SlidingWindowRateLimiter, verify_api_key
from plate_vision.utils.exceptions import (
    CorruptedImageError,
    EmptyDetectionError,
    InferenceTimeoutError,
    InvalidInputError,
    ModelLoadError,
    PlateVisionError,
)
from plate_vision.utils.storage import (
    InMemoryPredictionStore,
    PredictionStorageProtocol,
    build_audit_record,
)
from plate_vision.video.processor import (
    VideoFrameInput,
    VideoStreamProcessor,
    VideoStreamSummary,
)


def map_error_to_status_code(exc: PlateVisionError) -> int:
    if isinstance(exc, InvalidInputError | CorruptedImageError):
        return 422
    if isinstance(exc, EmptyDetectionError):
        return 404
    if isinstance(exc, InferenceTimeoutError):
        return 504
    if isinstance(exc, ModelLoadError):
        return 503
    return 500


def _decode_base64_payload(
    image_base64: str,
    mime_type: str,
    config: ServingConfig,
) -> bytes:
    if mime_type not in config.allowed_mime_types:
        raise InvalidInputError(
            f"Unsupported MIME type: {mime_type}",
            details={
                "mime_type": mime_type,
                "allowed_mime_types": list(config.allowed_mime_types),
            },
        )
    try:
        raw = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidInputError(
            "Invalid base64 image encoding",
            details={"reason": str(exc)},
        ) from exc
    if not raw:
        raise InvalidInputError("Decoded image payload is empty")
    if len(raw) > config.max_payload_bytes:
        raise InvalidInputError(
            f"Payload size {len(raw)} exceeds limit {config.max_payload_bytes}",
            details={
                "payload_bytes": len(raw),
                "max_payload_bytes": config.max_payload_bytes,
            },
        )
    return raw


def load_pipeline_from_config(config: ServingConfig) -> PlateVisionPipeline:
    detector = OnnxDetectorRuntime(config.detector_onnx_path)
    recognizer = OnnxRecognizerRuntime(config.recognizer_onnx_path)
    return PlateVisionPipeline(
        detector=detector,
        recognizer=recognizer,
        max_payload_bytes=config.max_payload_bytes,
        timeout_seconds=config.timeout_seconds,
    )


def _predict_bytes_with_optional_country(
    runner: PlateVisionPipeline,
    raw_bytes: bytes,
    country_code: str | None,
) -> tuple[PipelineInferenceResult, bool]:
    hits_before = runner.cache.hits if runner.cache is not None else 0
    if country_code is None:
        result = runner.predict_bytes(raw_bytes)
    else:
        profile = get_country_profile(country_code)
        scoped_runner = PlateVisionPipeline(
            detector=runner.detector,
            recognizer=runner.recognizer,
            profile=profile,
            max_payload_bytes=runner.max_payload_bytes,
            timeout_seconds=runner.timeout_seconds,
            min_plate_width=runner.min_plate_width,
            min_plate_height=runner.min_plate_height,
            target_crop_width=runner.target_crop_width,
            target_crop_height=runner.target_crop_height,
            confidence_threshold=runner.confidence_threshold,
            iou_threshold=runner.iou_threshold,
            max_detections=runner.max_detections,
            require_detection=runner.require_detection,
            use_beam_search=runner.use_beam_search,
            clock=runner.clock,
            cache=runner.cache,
            storage=runner.storage,
        )
        result = scoped_runner.predict_bytes(raw_bytes)
    hits_after = runner.cache.hits if runner.cache is not None else 0
    return result, hits_after > hits_before


def _persist_audit_if_needed(
    *,
    runner: PlateVisionPipeline,
    active_storage: PredictionStorageProtocol,
    raw_bytes: bytes,
    result: PipelineInferenceResult,
    country_code: str | None,
    cached: bool,
) -> None:
    if runner.storage is active_storage:
        return
    resolved_country = (
        country_code.strip().upper() if country_code is not None else runner.country_code
    )
    record = build_audit_record(
        raw_bytes,
        country_code=resolved_country,
        plate_texts=tuple(p.plate_text for p in result.plates),
        confidences=tuple(p.combined_confidence for p in result.plates),
        latency_ms=result.latency_ms,
        cached=cached,
    )
    active_storage.save(record)


class SecurityAndAuditMiddleware:
    def __init__(
        self,
        app: Any,
        config: ServingConfig,
        metrics: PrometheusMetricsRegistry,
        rate_limiter: SlidingWindowRateLimiter,
    ) -> None:
        self.app = app
        self.config = config
        self.metrics = metrics
        self.rate_limiter = rate_limiter

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        raw_headers: list[tuple[bytes, bytes]] = list(scope.get("headers", []))
        header_map: dict[str, str] = {
            bytes(k).decode("latin-1").lower(): bytes(v).decode("latin-1") for k, v in raw_headers
        }
        request_id = header_map.get("x-request-id", "").strip() or uuid.uuid4().hex
        path = str(scope.get("path", "/"))
        client_tuple = scope.get("client")
        client_host = (
            str(client_tuple[0])
            if isinstance(client_tuple, tuple | list) and client_tuple
            else "unknown"
        )
        provided_api_key = header_map.get("x-api-key")
        rate_remaining: int | None = None

        if path not in ("/healthz", "/readyz"):
            if not verify_api_key(provided_api_key, self.config.api_keys):
                self.metrics.record_error("UNAUTHORIZED")
                self.metrics.record_request(route=path, status_code=401, latency_ms=0.0)
                err_body = StructuredErrorResponse(
                    error_code="UNAUTHORIZED",
                    message="Missing or invalid API key",
                    details={"request_id": request_id},
                )
                await self._send_json(
                    send,
                    status_code=401,
                    payload=err_body.model_dump(),
                    extra_headers=[
                        (b"x-request-id", request_id.encode("latin-1")),
                    ],
                )
                return

            client_key = provided_api_key if provided_api_key else client_host
            decision = self.rate_limiter.check(client_key)
            rate_remaining = decision.remaining
            if not decision.allowed:
                self.metrics.record_error("RATE_LIMIT_EXCEEDED")
                self.metrics.record_request(route=path, status_code=429, latency_ms=0.0)
                err_body = StructuredErrorResponse(
                    error_code="RATE_LIMIT_EXCEEDED",
                    message="Rate limit exceeded",
                    details={
                        "request_id": request_id,
                        "reset_after_seconds": decision.reset_after_seconds,
                    },
                )
                await self._send_json(
                    send,
                    status_code=429,
                    payload=err_body.model_dump(),
                    extra_headers=[
                        (b"x-request-id", request_id.encode("latin-1")),
                        (b"x-ratelimit-remaining", b"0"),
                        (
                            b"retry-after",
                            str(max(1, int(decision.reset_after_seconds))).encode("ascii"),
                        ),
                    ],
                )
                return

        async def send_with_headers(message: MutableMapping[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                headers: list[tuple[bytes, bytes]] = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("latin-1")))
                if rate_remaining is not None:
                    headers.append(
                        (
                            b"x-ratelimit-remaining",
                            str(rate_remaining).encode("ascii"),
                        )
                    )
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)

    @staticmethod
    async def _send_json(
        send: Any,
        status_code: int,
        payload: dict[str, Any],
        extra_headers: list[tuple[bytes, bytes]],
    ) -> None:
        raw = json.dumps(payload).encode("utf-8")
        headers: list[tuple[bytes, bytes]] = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(raw)).encode("ascii")),
            *extra_headers,
        ]
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": headers,
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": raw,
                "more_body": False,
            }
        )


def create_app(
    config: ServingConfig | None = None,
    pipeline: PlateVisionPipeline | None = None,
    metrics: PrometheusMetricsRegistry | None = None,
    rate_limiter: SlidingWindowRateLimiter | None = None,
    storage: PredictionStorageProtocol | None = None,
    watchlist: WatchlistMatcher | None = None,
    drift_detector: DriftDetector | None = None,
) -> FastAPI:
    active_config = config if config is not None else ServingConfig()
    active_metrics = metrics if metrics is not None else PrometheusMetricsRegistry()
    active_limiter = (
        rate_limiter
        if rate_limiter is not None
        else SlidingWindowRateLimiter(
            max_requests_per_window=active_config.rate_limit_requests_per_minute,
            window_seconds=60.0,
        )
    )
    active_storage: PredictionStorageProtocol = (
        storage
        if storage is not None
        else (
            pipeline.storage
            if pipeline is not None and pipeline.storage is not None
            else InMemoryPredictionStore()
        )
    )
    active_watchlist = watchlist if watchlist is not None else WatchlistMatcher()
    active_pipeline = pipeline

    app = FastAPI(
        title="Plate-Vision Inference Service",
        version=__version__,
    )
    app.add_middleware(
        SecurityAndAuditMiddleware,
        config=active_config,
        metrics=active_metrics,
        rate_limiter=active_limiter,
    )

    @app.exception_handler(PlateVisionError)
    async def handle_domain_error(
        request: Request,
        exc: PlateVisionError,
    ) -> JSONResponse:
        status_code = map_error_to_status_code(exc)
        active_metrics.record_error(exc.error_code)
        active_metrics.record_request(
            route=request.url.path,
            status_code=status_code,
            latency_ms=0.0,
        )
        body = StructuredErrorResponse(
            error_code=exc.error_code,
            message=exc.message,
            details=dict(exc.details),
        )
        return JSONResponse(status_code=status_code, content=body.model_dump())

    def _resolve_pipeline() -> PlateVisionPipeline:
        nonlocal active_pipeline
        if active_pipeline is not None:
            return active_pipeline
        active_pipeline = load_pipeline_from_config(active_config)
        return active_pipeline

    @app.get("/healthz", response_model=HealthStatusResponse)
    async def healthz() -> HealthStatusResponse:
        active_metrics.record_request("/healthz", 200, 0.1)
        return HealthStatusResponse(
            status="ok",
            service="plate-vision",
            version=__version__,
        )

    @app.get("/readyz", response_model=ReadinessStatusResponse)
    async def readyz(response: Response) -> ReadinessStatusResponse:
        if active_pipeline is not None:
            active_metrics.record_request("/readyz", 200, 0.1)
            return ReadinessStatusResponse(
                ready=True,
                detector_loaded=True,
                recognizer_loaded=True,
            )
        det_ok = active_config.detector_onnx_path.is_file()
        rec_ok = active_config.recognizer_onnx_path.is_file()
        ready = det_ok and rec_ok
        status_code = 200 if ready else 503
        response.status_code = status_code
        active_metrics.record_request("/readyz", status_code, 0.1)
        return ReadinessStatusResponse(
            ready=ready,
            detector_loaded=det_ok,
            recognizer_loaded=rec_ok,
        )

    @app.get("/metrics")
    async def prometheus_metrics() -> PlainTextResponse:
        if not active_config.enable_metrics:
            return PlainTextResponse("metrics_disabled\n", status_code=404)
        return PlainTextResponse(
            active_metrics.render_prometheus_text(),
            media_type="text/plain; version=0.0.4",
        )

    @app.post("/v1/predict", response_model=PipelineInferenceResult)
    async def predict_single_json(
        payload: SinglePredictRequest,
    ) -> PipelineInferenceResult:
        raw_bytes = _decode_base64_payload(
            payload.image_base64,
            payload.mime_type,
            active_config,
        )
        runner = _resolve_pipeline()
        result, cached = _predict_bytes_with_optional_country(
            runner,
            raw_bytes,
            payload.country_code,
        )
        _persist_audit_if_needed(
            runner=runner,
            active_storage=active_storage,
            raw_bytes=raw_bytes,
            result=result,
            country_code=payload.country_code,
            cached=cached,
        )
        active_metrics.record_request(
            route="/v1/predict",
            status_code=200,
            latency_ms=result.latency_ms,
            plates_detected=len(result.plates),
        )
        return result

    @app.post("/v1/predict/upload", response_model=PipelineInferenceResult)
    async def predict_single_upload(
        request: Request,
    ) -> PipelineInferenceResult:
        raw_ct = request.headers.get("content-type", "application/octet-stream")
        content_type = raw_ct.split(";")[0].strip().lower()
        if content_type not in active_config.allowed_mime_types:
            raise InvalidInputError(
                f"Unsupported upload MIME type: {content_type}",
                details={
                    "mime_type": content_type,
                    "allowed_mime_types": list(active_config.allowed_mime_types),
                },
            )
        raw_bytes = await request.body()
        if not raw_bytes:
            raise InvalidInputError("Uploaded file is empty")
        if len(raw_bytes) > active_config.max_payload_bytes:
            raise InvalidInputError(
                f"Uploaded file size {len(raw_bytes)} exceeds limit "
                f"{active_config.max_payload_bytes}",
                details={
                    "payload_bytes": len(raw_bytes),
                    "max_payload_bytes": active_config.max_payload_bytes,
                },
            )
        runner = _resolve_pipeline()
        result, cached = _predict_bytes_with_optional_country(
            runner,
            raw_bytes,
            None,
        )
        _persist_audit_if_needed(
            runner=runner,
            active_storage=active_storage,
            raw_bytes=raw_bytes,
            result=result,
            country_code=None,
            cached=cached,
        )
        active_metrics.record_request(
            route="/v1/predict/upload",
            status_code=200,
            latency_ms=result.latency_ms,
            plates_detected=len(result.plates),
        )
        return result

    @app.post("/v1/predict/batch", response_model=BatchPredictResponse)
    async def predict_batch(
        payload: BatchPredictRequest,
    ) -> BatchPredictResponse:
        if len(payload.items) > active_config.max_batch_size:
            raise InvalidInputError(
                f"Batch size {len(payload.items)} exceeds limit {active_config.max_batch_size}",
                details={
                    "batch_size": len(payload.items),
                    "max_batch_size": active_config.max_batch_size,
                },
            )
        start = time.perf_counter()
        runner = _resolve_pipeline()
        results: list[PipelineInferenceResult] = []
        total_plates = 0

        for item in payload.items:
            raw_bytes = _decode_base64_payload(
                item.image_base64,
                item.mime_type,
                active_config,
            )
            res, cached = _predict_bytes_with_optional_country(
                runner,
                raw_bytes,
                item.country_code,
            )
            _persist_audit_if_needed(
                runner=runner,
                active_storage=active_storage,
                raw_bytes=raw_bytes,
                result=res,
                country_code=item.country_code,
                cached=cached,
            )
            results.append(res)
            total_plates += len(res.plates)

        elapsed_ms = max(0.0, (time.perf_counter() - start) * 1000.0)
        active_metrics.record_request(
            route="/v1/predict/batch",
            status_code=200,
            latency_ms=elapsed_ms,
            plates_detected=total_plates,
        )
        return BatchPredictResponse(
            results=tuple(results),
            total_plates=total_plates,
            batch_latency_ms=round(elapsed_ms, 4),
        )

    @app.post("/v1/predict/video", response_model=VideoStreamSummary)
    async def predict_video(
        payload: VideoPredictRequest,
    ) -> VideoStreamSummary:
        if len(payload.frames) > active_config.max_batch_size:
            raise InvalidInputError(
                f"Video frame count {len(payload.frames)} exceeds limit "
                f"{active_config.max_batch_size}",
                details={
                    "frame_count": len(payload.frames),
                    "max_batch_size": active_config.max_batch_size,
                },
            )
        start = time.perf_counter()
        runner = _resolve_pipeline()
        frame_inputs: list[VideoFrameInput] = []
        for idx, frame_req in enumerate(payload.frames):
            raw_bytes = _decode_base64_payload(
                frame_req.image_base64,
                frame_req.mime_type,
                active_config,
            )
            frame_inputs.append(
                VideoFrameInput(
                    frame_index=idx,
                    timestamp_ms=round((idx * 1000.0) / payload.fps, 4),
                    image_bytes=raw_bytes,
                )
            )
        processor = VideoStreamProcessor(pipeline=runner)
        summary = processor.process_stream(frame_inputs)
        elapsed_ms = max(0.0, (time.perf_counter() - start) * 1000.0)
        active_metrics.record_request(
            route="/v1/predict/video",
            status_code=200,
            latency_ms=elapsed_ms,
            plates_detected=summary.unique_tracks,
        )
        return summary

    @app.get("/v1/audit/recent", response_model=AuditLogResponse)
    async def audit_recent(limit: int = 50) -> AuditLogResponse:
        if limit < 1:
            raise InvalidInputError(
                "limit must be >= 1",
                details={"limit": limit},
            )
        records = active_storage.list_records(limit=limit)
        active_metrics.record_request("/v1/audit/recent", 200, 0.1)
        return AuditLogResponse(records=records, count=len(records))

    @app.post("/v1/watchlist/check", response_model=WatchlistCheckResponse)
    async def check_watchlist(
        payload: WatchlistCheckRequest,
    ) -> WatchlistCheckResponse:
        matches = active_watchlist.check(
            payload.plate_text,
            country_code=payload.country_code,
        )
        active_metrics.record_request("/v1/watchlist/check", 200, 0.1)
        return WatchlistCheckResponse(matches=matches, matched=bool(matches))

    @app.post("/v1/monitoring/drift", response_model=DriftReport)
    async def evaluate_drift(
        payload: DriftEvaluationRequest,
    ) -> DriftReport:
        if drift_detector is None:
            raise ModelLoadError("Drift detector baseline is not configured")
        report = drift_detector.evaluate(
            current_brightness=payload.brightness,
            current_contrast=payload.contrast,
            current_confidences=payload.confidences,
        )
        active_metrics.record_request("/v1/monitoring/drift", 200, 0.1)
        return report

    return app


class AsgiResponse:
    def __init__(self, status_code: int, headers: dict[str, str], body: bytes) -> None:
        self.status_code = status_code
        self.headers = headers
        self.body = body

    @property
    def text(self) -> str:
        return self.body.decode("utf-8")

    def json(self) -> Any:
        return json.loads(self.text)


class AsgiTestClient:
    def __init__(self, app: FastAPI) -> None:
        self.app = app

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        raw_body: bytes = b"",
        content_type: str | None = None,
        headers: Mapping[str, str] | None = None,
        client_host: str = "127.0.0.1",
    ) -> AsgiResponse:
        parsed = urlsplit(path)
        headers_list: list[tuple[bytes, bytes]] = [(b"host", b"testserver")]
        if json_body is not None:
            payload_bytes = json.dumps(json_body).encode("utf-8")
            headers_list.append((b"content-type", b"application/json"))
        else:
            payload_bytes = raw_body
            if content_type is not None:
                headers_list.append((b"content-type", content_type.encode("latin-1")))
        headers_list.append((b"content-length", str(len(payload_bytes)).encode("ascii")))
        if headers is not None:
            for k, v in headers.items():
                headers_list.append((k.lower().encode("latin-1"), v.encode("latin-1")))

        scope: dict[str, Any] = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": method.upper(),
            "scheme": "http",
            "path": parsed.path,
            "raw_path": parsed.path.encode("ascii"),
            "query_string": parsed.query.encode("ascii"),
            "headers": headers_list,
            "client": (client_host, 50000),
            "server": ("testserver", 80),
        }

        sent_request = False
        status_code = 500
        resp_headers: dict[str, str] = {}
        body_chunks: list[bytes] = []

        async def receive() -> dict[str, Any]:
            nonlocal sent_request
            if not sent_request:
                sent_request = True
                return {
                    "type": "http.request",
                    "body": payload_bytes,
                    "more_body": False,
                }
            return {"type": "http.disconnect"}

        async def send(message: MutableMapping[str, Any]) -> None:
            nonlocal status_code

            msg_type = message.get("type")
            if msg_type == "http.response.start":
                status_code = int(message.get("status", 200))
                for k, v in message.get("headers", []):
                    resp_headers[bytes(k).decode("latin-1").lower()] = bytes(v).decode("latin-1")
            elif msg_type == "http.response.body":
                chunk = message.get("body", b"")
                if chunk:
                    body_chunks.append(bytes(chunk))

        coro = self.app(scope, receive, send)
        try:
            while True:
                coro.send(None)
        except StopIteration:
            pass

        return AsgiResponse(
            status_code=status_code,
            headers=resp_headers,
            body=b"".join(body_chunks),
        )

    def get(
        self,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> AsgiResponse:
        return self.request("GET", path, headers=headers)

    def post(
        self,
        path: str,
        *,
        json: Any = None,
        raw_body: bytes = b"",
        content_type: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> AsgiResponse:
        return self.request(
            "POST",
            path,
            json_body=json,
            raw_body=raw_body,
            content_type=content_type,
            headers=headers,
        )


__all__ = [
    "AsgiResponse",
    "AsgiTestClient",
    "SecurityAndAuditMiddleware",
    "create_app",
    "load_pipeline_from_config",
    "map_error_to_status_code",
]
