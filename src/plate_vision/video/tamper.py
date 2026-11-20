import math
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from plate_vision.pipeline.rectification import decode_image_to_grayscale
from plate_vision.utils.exceptions import InvalidInputError


class TamperSignal(StrEnum):
    NONE = "none"
    SIGNAL_BLACKOUT = "signal_blackout"
    LENS_OCCLUSION = "lens_occlusion"
    GLARE_OVEREXPOSURE = "glare_overexposure"
    SEVERE_DEFOCUS = "severe_defocus"
    FOV_SHIFT = "fov_shift"


class CameraHealthThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    min_blackout_brightness: float = Field(default=8.0, ge=0.0, le=255.0)
    min_contrast_std: float = Field(default=6.0, ge=0.0)
    max_saturated_ratio: float = Field(default=0.45, gt=0.0, le=1.0)
    min_sharpness_score: float = Field(default=3.0, ge=0.0)
    max_quadrant_shift: float = Field(default=45.0, gt=0.0, le=255.0)
    consecutive_alert_frames: int = Field(default=2, ge=1)


class FrameHealthAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    camera_id: str = Field(min_length=1)
    mean_brightness: float = Field(ge=0.0, le=255.0)
    contrast_std: float = Field(ge=0.0)
    saturated_ratio: float = Field(ge=0.0, le=1.0)
    sharpness_score: float = Field(ge=0.0)
    quadrant_shift: float = Field(ge=0.0)
    signals: tuple[TamperSignal, ...]
    is_compromised: bool
    consecutive_compromised_count: int = Field(ge=0)
    maintenance_alert_triggered: bool


def _mean_slice_brightness(
    grid: Sequence[Sequence[int]],
    y_start: int,
    y_end: int,
    x_start: int,
    x_end: int,
) -> float:
    total = 0
    count = 0
    for y in range(y_start, y_end):
        row = grid[y]
        for x in range(x_start, x_end):
            total += row[x]
            count += 1
    return total / float(count)


def _compute_sharpness(grid: Sequence[Sequence[int]], height: int, width: int) -> float:
    diff_sum = 0
    diff_count = 0
    for y in range(height):
        row = grid[y]
        for x in range(width):
            curr = row[x]
            if x + 1 < width:
                diff_sum += abs(row[x + 1] - curr)
                diff_count += 1
            if y + 1 < height:
                diff_sum += abs(grid[y + 1][x] - curr)
                diff_count += 1
    return diff_sum / float(diff_count)


class CameraTamperMonitor:
    def __init__(self, thresholds: CameraHealthThresholds | None = None) -> None:
        self.thresholds = thresholds if thresholds is not None else CameraHealthThresholds()
        self._references: dict[str, tuple[float, float, float, float]] = {}
        self._consecutive_counts: dict[str, int] = {}

    @staticmethod
    def _validate_grid(
        camera_id: str,
        grid: Sequence[Sequence[int]],
    ) -> tuple[str, int, int]:
        clean_id = camera_id.strip()
        if not clean_id:
            raise InvalidInputError("camera_id must not be empty")
        if len(grid) < 2 or len(grid[0]) < 2:
            raise InvalidInputError("grid must be at least 2x2")
        height = len(grid)
        width = len(grid[0])
        for row in grid:
            if len(row) != width:
                raise InvalidInputError("All rows in grid must have equal width")
            for val in row:
                if val < 0 or val > 255:
                    raise InvalidInputError(
                        "Pixel values must be in [0, 255]",
                        details={"pixel_value": val},
                    )
        return clean_id, height, width

    @staticmethod
    def _compute_quadrant_means(
        grid: Sequence[Sequence[int]],
    ) -> tuple[float, float, float, float]:
        height = len(grid)
        width = len(grid[0])
        mid_y = height // 2
        mid_x = width // 2
        q_tl = _mean_slice_brightness(grid, 0, mid_y, 0, mid_x)
        q_tr = _mean_slice_brightness(grid, 0, mid_y, mid_x, width)
        q_bl = _mean_slice_brightness(grid, mid_y, height, 0, mid_x)
        q_br = _mean_slice_brightness(grid, mid_y, height, mid_x, width)
        return (round(q_tl, 4), round(q_tr, 4), round(q_bl, 4), round(q_br, 4))

    def _collect_signals(
        self,
        mean_brightness: float,
        contrast_std: float,
        saturated_ratio: float,
        sharpness_score: float,
        quadrant_shift: float,
    ) -> tuple[TamperSignal, ...]:
        signals: list[TamperSignal] = []
        if mean_brightness < self.thresholds.min_blackout_brightness:
            signals.append(TamperSignal.SIGNAL_BLACKOUT)
        if (
            contrast_std < self.thresholds.min_contrast_std
            and mean_brightness >= self.thresholds.min_blackout_brightness
        ):
            signals.append(TamperSignal.LENS_OCCLUSION)
        if saturated_ratio > self.thresholds.max_saturated_ratio:
            signals.append(TamperSignal.GLARE_OVEREXPOSURE)
        if sharpness_score < self.thresholds.min_sharpness_score:
            signals.append(TamperSignal.SEVERE_DEFOCUS)
        if quadrant_shift > self.thresholds.max_quadrant_shift:
            signals.append(TamperSignal.FOV_SHIFT)
        return tuple(signals) if signals else (TamperSignal.NONE,)

    def calibrate_reference(
        self,
        camera_id: str,
        grid: Sequence[Sequence[int]],
    ) -> tuple[float, float, float, float]:
        clean_id, _, _ = self._validate_grid(camera_id, grid)
        means = self._compute_quadrant_means(grid)
        self._references[clean_id] = means
        return means

    def evaluate_grid(
        self,
        camera_id: str,
        grid: Sequence[Sequence[int]],
    ) -> FrameHealthAssessment:
        clean_id, height, width = self._validate_grid(camera_id, grid)
        total_pixels = height * width
        flat = [val for row in grid for val in row]
        mean_brightness = sum(flat) / float(total_pixels)
        variance = sum((val - mean_brightness) ** 2 for val in flat) / float(total_pixels)
        contrast_std = math.sqrt(variance)
        saturated_ratio = sum(1 for val in flat if val >= 245) / float(total_pixels)
        sharpness_score = _compute_sharpness(grid, height, width)

        quadrant_shift = 0.0
        reference = self._references.get(clean_id)
        if reference is not None:
            current_means = self._compute_quadrant_means(grid)
            quadrant_shift = max(
                abs(curr - ref) for curr, ref in zip(current_means, reference, strict=True)
            )

        signals = self._collect_signals(
            mean_brightness=mean_brightness,
            contrast_std=contrast_std,
            saturated_ratio=saturated_ratio,
            sharpness_score=sharpness_score,
            quadrant_shift=quadrant_shift,
        )
        is_compromised = signals != (TamperSignal.NONE,)
        if is_compromised:
            count = self._consecutive_counts.get(clean_id, 0) + 1
            self._consecutive_counts[clean_id] = count
        else:
            count = 0
            self._consecutive_counts[clean_id] = 0

        alert_triggered = count >= self.thresholds.consecutive_alert_frames
        return FrameHealthAssessment(
            camera_id=clean_id,
            mean_brightness=round(mean_brightness, 4),
            contrast_std=round(contrast_std, 4),
            saturated_ratio=round(saturated_ratio, 6),
            sharpness_score=round(sharpness_score, 4),
            quadrant_shift=round(quadrant_shift, 4),
            signals=signals,
            is_compromised=is_compromised,
            consecutive_compromised_count=count,
            maintenance_alert_triggered=alert_triggered,
        )

    def evaluate_image_bytes(
        self,
        camera_id: str,
        image_bytes: bytes,
    ) -> FrameHealthAssessment:
        grid = decode_image_to_grayscale(image_bytes)
        return self.evaluate_grid(camera_id, grid)

    def alerted_cameras(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                cam_id
                for cam_id, count in self._consecutive_counts.items()
                if count >= self.thresholds.consecutive_alert_frames
            )
        )


__all__ = [
    "CameraHealthThresholds",
    "CameraTamperMonitor",
    "FrameHealthAssessment",
    "TamperSignal",
]
