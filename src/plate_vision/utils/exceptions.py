from typing import Any, ClassVar


class PlateVisionError(Exception):
    default_error_code: ClassVar[str] = "PLATE_VISION_ERROR"

    def __init__(
        self,
        message: str,
        *,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code if error_code is not None else self.default_error_code
        self.details: dict[str, Any] = dict(details) if details is not None else {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": dict(self.details),
        }


class ConfigurationError(PlateVisionError):
    default_error_code: ClassVar[str] = "CONFIGURATION_ERROR"


class DataIOError(PlateVisionError):
    default_error_code: ClassVar[str] = "DATA_IO_ERROR"


class CorruptedImageError(PlateVisionError):
    default_error_code: ClassVar[str] = "CORRUPTED_IMAGE_ERROR"


class EmptyDetectionError(PlateVisionError):
    default_error_code: ClassVar[str] = "EMPTY_DETECTION_ERROR"


class ModelLoadError(PlateVisionError):
    default_error_code: ClassVar[str] = "MODEL_LOAD_ERROR"


class InvalidInputError(PlateVisionError):
    default_error_code: ClassVar[str] = "INVALID_INPUT_ERROR"


class InferenceTimeoutError(PlateVisionError):
    default_error_code: ClassVar[str] = "INFERENCE_TIMEOUT_ERROR"


class DatasetValidationError(PlateVisionError):
    default_error_code: ClassVar[str] = "DATASET_VALIDATION_ERROR"


__all__ = [
    "ConfigurationError",
    "CorruptedImageError",
    "DataIOError",
    "DatasetValidationError",
    "EmptyDetectionError",
    "InferenceTimeoutError",
    "InvalidInputError",
    "ModelLoadError",
    "PlateVisionError",
]
