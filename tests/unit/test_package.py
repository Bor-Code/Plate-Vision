from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import pytest

import plate_vision


def test_package_constants() -> None:
    assert plate_vision.PACKAGE_NAME == "plate-vision"
    assert plate_vision.MODULE_NAME == "plate_vision"
    assert plate_vision.__version__ == "0.1.0"


def test_get_version_installed() -> None:
    resolved_version = plate_vision.get_version()
    assert resolved_version == "0.1.0"


def test_get_version_fallback_when_not_installed() -> None:
    with patch(
        "plate_vision.version",
        side_effect=PackageNotFoundError("plate-vision"),
    ):
        assert plate_vision.get_version() == plate_vision.__version__


def test_load_subpackage_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown subpackage: invalid_module"):
        plate_vision.load_subpackage("invalid_module")
