import base64
import binascii
import json
import time
from collections.abc import MutableMapping
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from plate_vision import __version__
from plate_vision.config.schema import ServingConfig
from plate_vision.export.runtime import OnnxDetectorRuntime, OnnxRecognizerRuntime
from plate_vision.pipeline.runner import PipelineInferenceResult, PlateVisionPipeline
from plate_vision.serving.metrics import PrometheusMetricsRegistry
from plate_vision.serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthStatusResponse,
    ReadinessStatusResponse,
    SinglePredictRequest,
    StructuredErrorResponse,
)
from plate_vision.utils.exceptions import (
    CorruptedImageError,
    EmptyDetectionError,
    InferenceTimeoutError,
    InvalidInputError,
    ModelLoadError,
    PlateVisionError,
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


def create_app(
    config: ServingConfig | None = None,
    pipeline: PlateVisionPipeline | None = None,
    metrics: PrometheusMetricsRegistry | None = None,
) -> FastAPI:
    active_config = config if config is not None else ServingConfig()
    active_metrics = metrics if metrics is not None else PrometheusMetricsRegistry()
    active_pipeline = pipeline

    app = FastAPI(
        title="Plate-Vision Inference Service",
        version=__version__,
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
        result = runner.predict_bytes(raw_bytes)
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
        result = runner.predict_bytes(raw_bytes)
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
            res = runner.predict_bytes(raw_bytes)
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
            "client": ("127.0.0.1", 50000),
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

    def get(self, path: str) -> AsgiResponse:
        return self.request("GET", path)

    def post(
        self,
        path: str,
        *,
        json: Any = None,
        raw_body: bytes = b"",
        content_type: str | None = None,
    ) -> AsgiResponse:
        return self.request(
            "POST",
            path,
            json_body=json,
            raw_body=raw_body,
            content_type=content_type,
        )


__all__ = [
    "AsgiResponse",
    "AsgiTestClient",
    "create_app",
    "load_pipeline_from_config",
    "map_error_to_status_code",
]
