from pathlib import Path
from unittest.mock import patch

import pytest

from plate_vision.config import (
    AppSettings,
    DataConfig,
    DatasetSplitRatios,
    DetectorConfig,
    EnvironmentType,
    deep_merge_dicts,
    load_component_config,
    load_settings,
    load_yaml_file,
)
from plate_vision.utils.exceptions import ConfigurationError, DataIOError


def test_split_ratios_validation_rejects_invalid_sum() -> None:
    with pytest.raises(ValueError, match=r"Split ratios must sum to 1\.0"):
        DatasetSplitRatios(train=0.60, val=0.20, test=0.10)


def test_detector_config_rejects_non_multiple_of_32_input_size() -> None:
    with pytest.raises(ValueError, match="Detector input_size must be divisible by 32"):
        DetectorConfig(input_size=650)


def test_load_yaml_file_errors(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"
    with pytest.raises(DataIOError, match="Configuration file does not exist"):
        load_yaml_file(missing_path)

    with pytest.raises(DataIOError, match="Configuration path is not a file"):
        load_yaml_file(tmp_path)

    unreadable_file = tmp_path / "unreadable.yaml"
    unreadable_file.write_text("seed: 42\n", encoding="utf-8")
    with (
        patch.object(Path, "read_text", side_effect=OSError("Permission denied")),
        pytest.raises(DataIOError, match="Failed to read configuration file"),
    ):
        load_yaml_file(unreadable_file)

    malformed_file = tmp_path / "malformed.yaml"
    malformed_file.write_text("seed: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Invalid YAML syntax"):
        load_yaml_file(malformed_file)

    list_root_file = tmp_path / "list_root.yaml"
    list_root_file.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Configuration root must be a mapping"):
        load_yaml_file(list_root_file)

    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("", encoding="utf-8")
    assert load_yaml_file(empty_file) == {}


def test_load_component_config_success_and_validation_failure(tmp_path: Path) -> None:
    valid_yaml = tmp_path / "data.yaml"
    valid_yaml.write_text("seed: 99\ncountry_profile: TR\n", encoding="utf-8")
    config = load_component_config(valid_yaml, DataConfig)
    assert config.seed == 99
    assert config.country_profile == "TR"

    invalid_yaml = tmp_path / "invalid_data.yaml"
    invalid_yaml.write_text("seed: -5\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Configuration validation failed for DataConfig"):
        load_component_config(invalid_yaml, DataConfig)


def test_deep_merge_dicts_nested_override() -> None:
    base = {"data": {"seed": 42, "country_profile": "TR"}, "seed": 42}
    override = {"data": {"seed": 7}, "environment": "production"}
    merged = deep_merge_dicts(base, override)
    assert merged == {
        "data": {"seed": 7, "country_profile": "TR"},
        "seed": 42,
        "environment": "production",
    }


def test_load_settings_rejects_invalid_overrides() -> None:
    with pytest.raises(ConfigurationError, match="Application settings validation failed"):
        load_settings(overrides={"seed": -1})


def test_app_settings_defaults() -> None:
    settings = AppSettings()
    assert settings.environment == EnvironmentType.DEVELOPMENT
    assert settings.seed == 42
    assert settings.data.country_profile == "TR"
