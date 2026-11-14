import io
import json
from pathlib import Path

import pytest

from plate_vision.config import EnvironmentType, LogLevel, load_settings
from plate_vision.utils import configure_logging, get_logger, log_environment_info

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIGS_DIR = ROOT_DIR / "configs"


def test_load_repository_default_configs_and_initialize_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PLATE_VISION_ENVIRONMENT", "staging")
    monkeypatch.setenv("PLATE_VISION_SEED", "123")
    monkeypatch.setenv("PLATE_VISION_LOGGING__LEVEL", "DEBUG")

    settings = load_settings(
        config_dir=CONFIGS_DIR,
        overrides={"serving": {"port": 9090}},
    )
    assert settings.environment == EnvironmentType.STAGING
    assert settings.seed == 123
    assert settings.logging.level == LogLevel.DEBUG
    assert settings.detection.input_size == 640
    assert settings.recognition.image_width == 128
    assert settings.serving.port == 9090

    stream = io.StringIO()
    configure_logging(
        level=settings.logging.level.value,
        json_format=settings.logging.json_format,
        service_name=settings.logging.service_name,
        include_timestamp=settings.logging.include_timestamp,
        stream=stream,
    )
    logger = get_logger("plate_vision.integration")
    env_info = log_environment_info(logger, seed=settings.seed)
    assert env_info["seed"] == 123

    record = json.loads(stream.getvalue().strip())
    assert record["event"] == "environment_initialized"
    assert record["service"] == "plate-vision"
    assert record["seed"] == 123
