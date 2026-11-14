import pytest

from plate_vision.utils.exceptions import (
    ConfigurationError,
    CorruptedImageError,
    DataIOError,
    DatasetValidationError,
    EmptyDetectionError,
    InferenceTimeoutError,
    InvalidInputError,
    ModelLoadError,
    PlateVisionError,
)


def test_base_exception_default_properties() -> None:
    err = PlateVisionError("Unexpected failure")
    assert str(err) == "Unexpected failure"
    assert err.message == "Unexpected failure"
    assert err.error_code == "PLATE_VISION_ERROR"
    assert err.details == {}
    assert err.to_dict() == {
        "error_code": "PLATE_VISION_ERROR",
        "message": "Unexpected failure",
        "details": {},
    }


def test_base_exception_custom_code_and_details() -> None:
    source_details = {"file": "image.jpg", "bytes": 0}
    err = PlateVisionError(
        "File is empty",
        error_code="CUSTOM_EMPTY_FILE",
        details=source_details,
    )
    source_details["file"] = "mutated.jpg"
    serialized = err.to_dict()
    assert serialized["error_code"] == "CUSTOM_EMPTY_FILE"
    assert serialized["details"] == {"file": "image.jpg", "bytes": 0}


@pytest.mark.parametrize(
    ("exc_cls", "expected_code"),
    [
        (ConfigurationError, "CONFIGURATION_ERROR"),
        (DataIOError, "DATA_IO_ERROR"),
        (CorruptedImageError, "CORRUPTED_IMAGE_ERROR"),
        (EmptyDetectionError, "EMPTY_DETECTION_ERROR"),
        (ModelLoadError, "MODEL_LOAD_ERROR"),
        (InvalidInputError, "INVALID_INPUT_ERROR"),
        (InferenceTimeoutError, "INFERENCE_TIMEOUT_ERROR"),
        (DatasetValidationError, "DATASET_VALIDATION_ERROR"),
    ],
)
def test_domain_exception_hierarchy(
    exc_cls: type[PlateVisionError],
    expected_code: str,
) -> None:
    instance = exc_cls("Domain error occurred", details={"step": "validation"})
    assert isinstance(instance, PlateVisionError)
    assert instance.error_code == expected_code
    assert instance.details == {"step": "validation"}
