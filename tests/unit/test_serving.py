from plate_vision.serving.app import map_error_to_status_code
from plate_vision.serving.metrics import PrometheusMetricsRegistry
from plate_vision.utils.exceptions import (
    ConfigurationError,
    CorruptedImageError,
    EmptyDetectionError,
    InferenceTimeoutError,
    InvalidInputError,
    ModelLoadError,
)


def test_prometheus_metrics_registry_records_and_renders() -> None:
    registry = PrometheusMetricsRegistry()
    registry.record_request("/v1/predict", 200, latency_ms=14.2, plates_detected=2)
    registry.record_request("/v1/predict", 422, latency_ms=1.1, plates_detected=0)
    registry.record_error("PV_INVALID_INPUT")

    snap = registry.snapshot()
    assert snap.requests_total == 2
    assert snap.errors_total == 1
    assert snap.plates_detected_total == 2
    assert snap.latency_count == 2

    text = registry.render_prometheus_text()
    assert 'plate_vision_http_requests_total{route="/v1/predict",status="200"} 1' in text
    assert 'plate_vision_errors_total{error_code="PV_INVALID_INPUT"} 1' in text
    assert "plate_vision_plates_detected_total 2" in text


def test_map_error_to_status_code_covers_all_domain_exceptions() -> None:
    assert map_error_to_status_code(InvalidInputError("bad")) == 422
    assert map_error_to_status_code(CorruptedImageError("corrupt")) == 422
    assert map_error_to_status_code(EmptyDetectionError("none")) == 404
    assert map_error_to_status_code(InferenceTimeoutError("slow")) == 504
    assert map_error_to_status_code(ModelLoadError("missing")) == 503
    assert map_error_to_status_code(ConfigurationError("cfg")) == 500
