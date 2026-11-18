import math
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from plate_vision.utils.exceptions import InvalidInputError


class QualityIssue(StrEnum):
    LOW_LIGHT = "low_light"
    OVEREXPOSED_GLARE = "overexposed_glare"
    LOW_CONTRAST = "low_contrast"
    MOTION_BLUR = "motion_blur"


class QualityThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    min_mean_brightness: float = Field(default=35.0, ge=0.0, le=255.0)
    max_mean_brightness: float = Field(default=225.0, ge=0.0, le=255.0)
    max_glare_pixel_ratio: float = Field(default=0.25, ge=0.0, le=1.0)
    glare_pixel_cutoff: int = Field(default=245, ge=1, le=255)
    min_contrast_std: float = Field(default=12.0, ge=0.0)
    min_sharpness_gradient: float = Field(default=6.0, ge=0.0)


class ImageQualityReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mean_brightness: float = Field(ge=0.0, le=255.0)
    contrast_std: float = Field(ge=0.0)
    glare_pixel_ratio: float = Field(ge=0.0, le=1.0)
    sharpness_gradient: float = Field(ge=0.0)
    quality_score: float = Field(ge=0.0, le=1.0)
    is_acceptable: bool
    issues: tuple[QualityIssue, ...]


def assess_image_quality(
    pixels: Sequence[Sequence[int]],
    thresholds: QualityThresholds | None = None,
) -> ImageQualityReport:
    if not pixels or not pixels[0]:
        raise InvalidInputError("pixels must be a non-empty 2D pixel grid")

    height = len(pixels)
    width = len(pixels[0])
    grid = [[max(0, min(255, int(val))) for val in row] for row in pixels]
    if any(len(row) != width for row in grid):
        raise InvalidInputError("All rows in pixels must have equal width")

    active = thresholds if thresholds is not None else QualityThresholds()
    total_pixels = height * width
    flat = [val for row in grid for val in row]
    mean_brightness = sum(flat) / float(total_pixels)
    variance = sum((val - mean_brightness) ** 2 for val in flat) / float(total_pixels)
    contrast_std = math.sqrt(variance)
    glare_count = sum(1 for val in flat if val >= active.glare_pixel_cutoff)
    glare_ratio = glare_count / float(total_pixels)

    grad_sum = 0.0
    grad_count = 0
    for y in range(height):
        for x in range(width):
            curr = grid[y][x]
            if x + 1 < width:
                grad_sum += abs(grid[y][x + 1] - curr)
                grad_count += 1
            if y + 1 < height:
                grad_sum += abs(grid[y + 1][x] - curr)
                grad_count += 1
    sharpness_gradient = grad_sum / float(grad_count) if grad_count > 0 else 0.0

    issues: list[QualityIssue] = []
    if mean_brightness < active.min_mean_brightness:
        issues.append(QualityIssue.LOW_LIGHT)
    if mean_brightness > active.max_mean_brightness or glare_ratio > active.max_glare_pixel_ratio:
        issues.append(QualityIssue.OVEREXPOSED_GLARE)
    if contrast_std < active.min_contrast_std:
        issues.append(QualityIssue.LOW_CONTRAST)
    if sharpness_gradient < active.min_sharpness_gradient:
        issues.append(QualityIssue.MOTION_BLUR)

    score = max(0.0, 1.0 - 0.25 * len(issues))
    return ImageQualityReport(
        mean_brightness=round(mean_brightness, 4),
        contrast_std=round(contrast_std, 4),
        glare_pixel_ratio=round(glare_ratio, 6),
        sharpness_gradient=round(sharpness_gradient, 4),
        quality_score=round(score, 4),
        is_acceptable=(len(issues) == 0),
        issues=tuple(issues),
    )


__all__ = [
    "ImageQualityReport",
    "QualityIssue",
    "QualityThresholds",
    "assess_image_quality",
]
