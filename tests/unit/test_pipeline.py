import pytest

from plate_vision.data.models import BoundingBox
from plate_vision.data.synthetic import encode_grayscale_png
from plate_vision.pipeline.postprocess import postprocess_plate_text
from plate_vision.pipeline.rectification import (
    Point2D,
    decode_image_to_grayscale,
    extract_plate_crop,
    normalize_plate_crop,
    rectify_perspective_crop,
)
from plate_vision.utils.exceptions import CorruptedImageError, InvalidInputError


def test_decode_image_to_grayscale_rejects_empty_oversized_and_non_image() -> None:
    with pytest.raises(InvalidInputError, match="empty"):
        decode_image_to_grayscale(b"")

    grid = [[120 for _ in range(32)] for _ in range(16)]
    png_bytes = encode_grayscale_png(grid)
    with pytest.raises(InvalidInputError, match="exceeds limit"):
        decode_image_to_grayscale(png_bytes, max_payload_bytes=16)

    with pytest.raises(InvalidInputError):
        decode_image_to_grayscale(b"NOT_AN_IMAGE_PAYLOAD_1234567890")

    with pytest.raises(CorruptedImageError):
        decode_image_to_grayscale(b"\x89PNG\r\n\x1a\nTRUNCATED")

    jpeg_bytes = b"\xff\xd8\xff\xc0\x00\x11\x08\x00\x10\x00\x20\x03" + b"\x80" * 64
    decoded_jpeg = decode_image_to_grayscale(jpeg_bytes)
    assert len(decoded_jpeg) == 16
    assert len(decoded_jpeg[0]) == 32


def test_extract_plate_crop_validates_tiny_plates_and_empty_grid() -> None:
    with pytest.raises(InvalidInputError, match="empty"):
        extract_plate_crop([], BoundingBox(x_min=0.0, y_min=0.0, x_max=20.0, y_max=10.0))

    grid = [[100 for _ in range(64)] for _ in range(32)]
    tiny_box = BoundingBox(x_min=5.0, y_min=5.0, x_max=10.0, y_max=9.0)
    with pytest.raises(InvalidInputError, match="smaller than minimum"):
        extract_plate_crop(grid, tiny_box, min_width=12, min_height=6)

    valid_box = BoundingBox(x_min=10.0, y_min=4.0, x_max=40.0, y_max=20.0)
    crop = extract_plate_crop(grid, valid_box)
    assert len(crop) == 16
    assert len(crop[0]) == 30


def test_rectify_perspective_crop_handles_rotated_and_skewed_quadrilateral() -> None:
    grid = [[(x * 3 + y * 5) % 256 for x in range(80)] for y in range(40)]
    corners = (
        Point2D(x=10.0, y=8.0),
        Point2D(x=65.0, y=4.0),
        Point2D(x=70.0, y=28.0),
        Point2D(x=8.0, y=32.0),
    )
    rectified = rectify_perspective_crop(grid, corners, output_width=64, output_height=16)
    assert len(rectified) == 16
    assert len(rectified[0]) == 64

    with pytest.raises(InvalidInputError, match="output_width"):
        rectify_perspective_crop(grid, corners, output_width=4, output_height=4)


def test_normalize_plate_crop_enhances_low_light_and_motion_blur() -> None:
    with pytest.raises(InvalidInputError, match="empty"):
        normalize_plate_crop([])

    low_light = [[20 + (x % 5) for x in range(32)] for _ in range(16)]
    normalized = normalize_plate_crop(low_light, target_width=64, target_height=16)
    assert len(normalized) == 16
    assert len(normalized[0]) == 64
    flat = [px for row in normalized for px in row]
    assert max(flat) == 255
    assert min(flat) == 0


def test_postprocess_plate_text_corrects_confusions_and_validates_format() -> None:
    res = postprocess_plate_text("34 a8 0123", country_code="TR")
    assert res.normalized_text == "34A80123"
    assert res.corrected_text == "34AB0123"
    assert res.is_valid_format is True
    assert res.was_corrected is True

    occluded = postprocess_plate_text("34A", country_code="TR")
    assert occluded.is_valid_format is False
    assert occluded.was_corrected is False
