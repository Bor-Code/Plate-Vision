from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from plate_vision.data.models import BoundingBox
from plate_vision.utils.exceptions import InvalidInputError


class RedactionMode(StrEnum):
    BLUR = "blur"
    PIXELATE = "pixelate"
    SOLID_FILL = "solid_fill"


class RedactionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: RedactionMode = Field(default=RedactionMode.BLUR)
    block_size: int = Field(default=4, ge=2, le=32)
    fill_value: int = Field(default=0, ge=0, le=255)


def _clamp_box(
    bbox: BoundingBox,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x0 = max(0, min(width, int(bbox.x_min)))
    y0 = max(0, min(height, int(bbox.y_min)))
    x1 = max(0, min(width, int(bbox.x_max)))
    y1 = max(0, min(height, int(bbox.y_max)))
    return x0, y0, x1, y1


def _apply_solid_fill(
    output: list[list[int]],
    bounds: tuple[int, int, int, int],
    fill_value: int,
) -> None:
    x0, y0, x1, y1 = bounds
    for y in range(y0, y1):
        for x in range(x0, x1):
            output[y][x] = fill_value


def _apply_pixelate(
    output: list[list[int]],
    source: list[list[int]],
    bounds: tuple[int, int, int, int],
    block_size: int,
) -> None:
    x0, y0, x1, y1 = bounds
    for by in range(y0, y1, block_size):
        by_end = min(y1, by + block_size)
        for bx in range(x0, x1, block_size):
            bx_end = min(x1, bx + block_size)
            total = 0
            count = 0
            for y in range(by, by_end):
                for x in range(bx, bx_end):
                    total += source[y][x]
                    count += 1
            mean_val = total // count if count > 0 else 0
            for y in range(by, by_end):
                for x in range(bx, bx_end):
                    output[y][x] = mean_val


def _apply_box_blur(
    output: list[list[int]],
    source: list[list[int]],
    bounds: tuple[int, int, int, int],
    width: int,
    height: int,
) -> None:
    x0, y0, x1, y1 = bounds
    for y in range(y0, y1):
        ky0 = max(0, y - 1)
        ky1 = min(height, y + 2)
        for x in range(x0, x1):
            kx0 = max(0, x - 1)
            kx1 = min(width, x + 2)
            total = 0
            count = 0
            for ny in range(ky0, ky1):
                for nx in range(kx0, kx1):
                    total += source[ny][nx]
                    count += 1
            output[y][x] = total // count if count > 0 else source[y][x]


def redact_regions(
    image_pixels: Sequence[Sequence[int]],
    bboxes: Sequence[BoundingBox],
    config: RedactionConfig | None = None,
) -> list[list[int]]:
    if not image_pixels or not image_pixels[0]:
        raise InvalidInputError("image_pixels must be a non-empty 2D pixel grid")

    height = len(image_pixels)
    width = len(image_pixels[0])
    source = [[int(val) for val in row] for row in image_pixels]
    if any(len(row) != width for row in source):
        raise InvalidInputError("All rows in image_pixels must have equal width")

    output = [list(row) for row in source]
    active_config = config if config is not None else RedactionConfig()

    for bbox in bboxes:
        bounds = _clamp_box(bbox, width, height)
        x0, y0, x1, y1 = bounds
        if x1 <= x0 or y1 <= y0:
            continue
        if active_config.mode == RedactionMode.SOLID_FILL:
            _apply_solid_fill(output, bounds, active_config.fill_value)
        elif active_config.mode == RedactionMode.PIXELATE:
            _apply_pixelate(output, source, bounds, active_config.block_size)
        else:
            _apply_box_blur(output, source, bounds, width, height)

    return output


__all__ = [
    "RedactionConfig",
    "RedactionMode",
    "redact_regions",
]
