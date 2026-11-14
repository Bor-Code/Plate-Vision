import io
import json

import pytest

from plate_vision.utils.logging import (
    bind_context,
    clear_context,
    collect_environment_info,
    configure_logging,
    get_logger,
    log_environment_info,
)


def test_configure_logging_json_output_and_context() -> None:
    buffer = io.StringIO()
    configure_logging(
        level="INFO",
        json_format=True,
        service_name="plate-vision-test",
        include_timestamp=True,
        stream=buffer,
    )
    bind_context(request_id="req-123")
    logger = get_logger("plate_vision.test")
    logger.info("request_processed", latency_ms=12.5)

    output = buffer.getvalue().strip()
    payload = json.loads(output)
    assert payload["event"] == "request_processed"
    assert payload["level"] == "info"
    assert payload["service"] == "plate-vision-test"
    assert payload["request_id"] == "req-123"
    assert payload["latency_ms"] == 12.5
    assert "timestamp" in payload

    clear_context()


def test_configure_logging_console_renderer_without_timestamp() -> None:
    buffer = io.StringIO()
    configure_logging(
        level="DEBUG",
        json_format=False,
        service_name="console-service",
        include_timestamp=False,
        stream=buffer,
    )
    logger = get_logger("plate_vision.console")
    logger.debug("debug_event", step="init")
    output = buffer.getvalue()
    assert "debug_event" in output
    assert "step=init" in output


def test_configure_logging_rejects_invalid_level() -> None:
    with pytest.raises(ValueError, match="Unsupported log level: INVALID_LEVEL"):
        configure_logging(level="INVALID_LEVEL")


def test_log_environment_info_emits_structured_metadata() -> None:
    buffer = io.StringIO()
    configure_logging(level="INFO", json_format=True, stream=buffer)
    logger = get_logger("plate_vision.env")

    metadata = log_environment_info(logger, seed=1337)
    assert metadata["seed"] == 1337
    assert metadata == collect_environment_info(seed=1337)

    logged = json.loads(buffer.getvalue().strip())
    assert logged["event"] == "environment_initialized"
    assert logged["seed"] == 1337
    assert "python_version" in logged
