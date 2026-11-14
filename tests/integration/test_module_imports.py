from pathlib import Path

import pytest

import plate_vision


@pytest.mark.parametrize("subpackage_name", plate_vision.SUBPACKAGES)
def test_all_subpackages_importable(subpackage_name: str) -> None:
    module = plate_vision.load_subpackage(subpackage_name)
    assert module.__name__ == f"plate_vision.{subpackage_name}"
    assert hasattr(module, "__all__")


def test_py_typed_marker_exists() -> None:
    package_root = Path(plate_vision.__file__).resolve().parent
    marker_path = package_root / "py.typed"
    assert marker_path.is_file()
