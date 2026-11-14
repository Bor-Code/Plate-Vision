from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType
from typing import Final

PACKAGE_NAME: Final[str] = "plate-vision"
MODULE_NAME: Final[str] = "plate_vision"
__version__: Final[str] = "0.1.0"

SUBPACKAGES: Final[tuple[str, ...]] = (
    "config",
    "data",
    "detection",
    "recognition",
    "pipeline",
    "evaluation",
    "export",
    "serving",
    "video",
    "utils",
)


def get_version() -> str:
    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return __version__


def load_subpackage(subpackage_name: str) -> ModuleType:
    if subpackage_name not in SUBPACKAGES:
        msg = f"Unknown subpackage: {subpackage_name}"
        raise ValueError(msg)
    return import_module(f"{MODULE_NAME}.{subpackage_name}")


__all__ = [
    "MODULE_NAME",
    "PACKAGE_NAME",
    "SUBPACKAGES",
    "__version__",
    "get_version",
    "load_subpackage",
]
