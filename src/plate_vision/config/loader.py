from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from plate_vision.config.schema import (
    AppSettings,
    DataConfig,
    DetectorConfig,
    RecognizerConfig,
    ServingConfig,
)
from plate_vision.utils.exceptions import ConfigurationError, DataIOError

TModel = TypeVar("TModel", bound=BaseModel)


def load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DataIOError(
            f"Configuration file does not exist: {path}",
            details={"path": str(path)},
        )
    if not path.is_file():
        raise DataIOError(
            f"Configuration path is not a file: {path}",
            details={"path": str(path)},
        )
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DataIOError(
            f"Failed to read configuration file: {path}",
            details={"path": str(path), "reason": str(exc)},
        ) from exc

    try:
        parsed = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Invalid YAML syntax in configuration file: {path}",
            details={"path": str(path), "reason": str(exc)},
        ) from exc

    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ConfigurationError(
            f"Configuration root must be a mapping in file: {path}",
            details={"path": str(path), "actual_type": type(parsed).__name__},
        )
    return dict(parsed)


def load_component_config(path: Path, model_cls: type[TModel]) -> TModel:
    payload = load_yaml_file(path)
    try:
        return model_cls.model_validate(payload)
    except ValidationError as exc:
        raise ConfigurationError(
            f"Configuration validation failed for {model_cls.__name__}",
            details={"path": str(path), "errors": exc.errors()},
        ) from exc


def deep_merge_dicts(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge_dicts(existing, value)
        else:
            merged[key] = value
    return merged


def load_settings(
    config_dir: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> AppSettings:
    payload: dict[str, Any] = {}
    if config_dir is not None:
        section_files: tuple[tuple[str, Path, type[BaseModel]], ...] = (
            ("data", config_dir / "data" / "default.yaml", DataConfig),
            ("detection", config_dir / "detection" / "default.yaml", DetectorConfig),
            ("recognition", config_dir / "recognition" / "default.yaml", RecognizerConfig),
            ("serving", config_dir / "serving" / "default.yaml", ServingConfig),
        )
        for section_name, section_path, _ in section_files:
            if section_path.exists():
                payload[section_name] = load_yaml_file(section_path)

    if overrides is not None:
        payload = deep_merge_dicts(payload, overrides)

    try:
        return AppSettings(**payload)
    except ValidationError as exc:
        raise ConfigurationError(
            "Application settings validation failed",
            details={"errors": exc.errors()},
        ) from exc


__all__ = [
    "deep_merge_dicts",
    "load_component_config",
    "load_settings",
    "load_yaml_file",
]
