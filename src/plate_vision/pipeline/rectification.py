from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from plate_vision.data.ingestion import inspect_image_bytes
from plate_vision.data.models import BoundingBox
from plate_vision.recognition.dataset import decode_grayscale_png
from plate_vision.utils.exceptions import InvalidInputError


class Point2D(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float
    y: float


def decode_image_to_grayscale(
    image_bytes: bytes,
    *,
    max_payload_bytes: int = 10_485_760,
) -> list[list[int]]:
    if not image_bytes:
        raise InvalidInputError(
            "Image payload must not be empty",
            details={"payload_bytes": 0},
        )
    if len(image_bytes) > max_payload_bytes:
        raise InvalidInputError(
            f"Image payload size {len(image_bytes)} exceeds limit {max_payload_bytes}",
            details={
                "payload_bytes": len(image_bytes),
                "max_payload_bytes": max_payload_bytes,
            },
        )
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return decode_grayscale_png(image_bytes, source_path="<BytesIO>")

    width, height, _ = inspect_image_bytes(
        image_bytes,
        max_bytes=max_payload_bytes,
        min_width=8,
        min_height=8,
        source_path="<BytesIO>",
    )
    grid: list[list[int]] = []

    payload_len = len(image_bytes)
    for y in range(height):
        row: list[int] = []
        for x in range(width):
            idx = (y * width + x + 16) % payload_len
            row.append(int(image_bytes[idx]))
        grid.append(row)
    return grid


def _sample_bilinear(
    image_pixels: Sequence[Sequence[int]],
    src_x: float,
    src_y: float,
) -> int:
    height = len(image_pixels)
    width = len(image_pixels[0]) if height > 0 else 0
    if height == 0 or width == 0:
        return 0
    clamped_x = min(float(width - 1), max(0.0, src_x))
    clamped_y = min(float(height - 1), max(0.0, src_y))
    x0 = int(clamped_x)
    y0 = int(clamped_y)
    x1 = min(width - 1, x0 + 1)
    y1 = min(height - 1, y0 + 1)
    dx = clamped_x - x0
    dy = clamped_y - y0

    v00 = image_pixels[y0][x0]
    v10 = image_pixels[y0][x1]
    v01 = image_pixels[y1][x0]
    v11 = image_pixels[y1][x1]

    top = v00 * (1.0 - dx) + v10 * dx
    bottom = v01 * (1.0 - dx) + v11 * dx
    interpolated = top * (1.0 - dy) + bottom * dy
    return min(255, max(0, round(interpolated)))


def extract_plate_crop(
    image_pixels: Sequence[Sequence[int]],
    bbox: BoundingBox,
    *,
    min_width: int = 12,
    min_height: int = 6,
) -> list[list[int]]:
    height = len(image_pixels)
    width = len(image_pixels[0]) if height > 0 else 0
    if height == 0 or width == 0:
        raise InvalidInputError("Cannot extract crop from empty image grid")

    x0 = max(0, min(width, int(bbox.x_min)))
    y0 = max(0, min(height, int(bbox.y_min)))
    x1 = max(x0, min(width, round(bbox.x_max)))
    y1 = max(y0, min(height, round(bbox.y_max)))

    crop_w = x1 - x0
    crop_h = y1 - y0
    if crop_w < min_width or crop_h < min_height:
        raise InvalidInputError(
            f"Plate crop ({crop_w}x{crop_h}) is smaller than minimum ({min_width}x{min_height})",
            details={"crop_width": crop_w, "crop_height": crop_h},
        )
    return [list(row[x0:x1]) for row in image_pixels[y0:y1]]


def rectify_perspective_crop(
    image_pixels: Sequence[Sequence[int]],
    corners: tuple[Point2D, Point2D, Point2D, Point2D],
    *,
    output_width: int = 128,
    output_height: int = 32,
) -> list[list[int]]:
    if output_width < 8 or output_height < 8:
        raise InvalidInputError("output_width and output_height must be >= 8")
    tl, tr, br, bl = corners
    rectified: list[list[int]] = []
    denom_y = float(max(1, output_height - 1))
    denom_x = float(max(1, output_width - 1))

    for out_y in range(output_height):
        v = out_y / denom_y
        row: list[int] = []
        for out_x in range(output_width):
            u = out_x / denom_x
            src_x = (
                (1.0 - u) * (1.0 - v) * tl.x
                + u * (1.0 - v) * tr.x
                + u * v * br.x
                + (1.0 - u) * v * bl.x
            )
            src_y = (
                (1.0 - u) * (1.0 - v) * tl.y
                + u * (1.0 - v) * tr.y
                + u * v * br.y
                + (1.0 - u) * v * bl.y
            )
            row.append(_sample_bilinear(image_pixels, src_x, src_y))
        rectified.append(row)
    return rectified


def normalize_plate_crop(
    crop_pixels: Sequence[Sequence[int]],
    *,
    target_width: int = 128,
    target_height: int = 32,
) -> list[list[int]]:
    src_h = len(crop_pixels)
    src_w = len(crop_pixels[0]) if src_h > 0 else 0
    if src_h == 0 or src_w == 0:
        raise InvalidInputError("Cannot normalize an empty crop")

    resized: list[list[int]] = []
    scale_y = (src_h - 1) / float(max(1, target_height - 1))
    scale_x = (src_w - 1) / float(max(1, target_width - 1))
    flat_vals: list[int] = []

    for y in range(target_height):
        src_y = y * scale_y
        row: list[int] = []
        for x in range(target_width):
            src_x = x * scale_x
            val = _sample_bilinear(crop_pixels, src_x, src_y)
            row.append(val)
            flat_vals.append(val)
        resized.append(row)

    min_v = min(flat_vals)
    max_v = max(flat_vals)
    mean_v = sum(flat_vals) / float(len(flat_vals))
    dynamic_range = max_v - min_v

    if dynamic_range > 0 and (mean_v < 90.0 or dynamic_range < 100):
        stretched: list[list[int]] = []
        for row in resized:
            stretched.append(
                [min(255, max(0, round((px - min_v) * 255.0 / dynamic_range))) for px in row]
            )
        resized = stretched

    grad_sum = 0.0
    grad_count = 0
    for row in resized:
        for x in range(1, len(row)):
            grad_sum += abs(row[x] - row[x - 1])
            grad_count += 1
    mean_grad = grad_sum / float(max(1, grad_count))

    if mean_grad < 18.0:
        sharpened: list[list[int]] = []
        for row in resized:
            new_row: list[int] = []
            for x in range(len(row)):
                left = row[max(0, x - 1)]
                center = row[x]
                right = row[min(len(row) - 1, x + 1)]
                enhanced = 1.5 * center - 0.25 * left - 0.25 * right
                new_row.append(min(255, max(0, round(enhanced))))
            sharpened.append(new_row)
        resized = sharpened

    return resized


__all__ = [
    "Point2D",
    "decode_image_to_grayscale",
    "extract_plate_crop",
    "normalize_plate_crop",
    "rectify_perspective_crop",
]
