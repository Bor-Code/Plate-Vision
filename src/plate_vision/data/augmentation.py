import math
import random
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from plate_vision.data.synthetic import clamp_byte


class AugmentationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = Field(default=True)
    brightness_limit: float = Field(default=0.2, ge=0.0, le=0.8)
    contrast_limit: float = Field(default=0.2, ge=0.0, le=0.8)
    motion_blur_kernel: int = Field(default=3, ge=1, le=9)
    noise_std_limit: float = Field(default=6.0, ge=0.0, le=40.0)
    max_rotation_degrees: float = Field(default=6.0, ge=0.0, le=30.0)
    occlusion_max_fraction: float = Field(default=0.15, ge=0.0, le=0.5)
    probability: float = Field(default=0.5, ge=0.0, le=1.0)


class AugmentationPipeline:
    def __init__(self, config: AugmentationConfig) -> None:
        self.config = config

    def to_albumentations_spec(self) -> list[dict[str, Any]]:
        if not self.config.enabled:
            return []
        return [
            {
                "transform": "RandomBrightnessContrast",
                "brightness_limit": self.config.brightness_limit,
                "contrast_limit": self.config.contrast_limit,
                "p": self.config.probability,
            },
            {
                "transform": "MotionBlur",
                "blur_limit": self.config.motion_blur_kernel,
                "p": self.config.probability,
            },
            {
                "transform": "GaussNoise",
                "std_limit": self.config.noise_std_limit,
                "p": self.config.probability,
            },
            {
                "transform": "Rotate",
                "limit": self.config.max_rotation_degrees,
                "p": self.config.probability,
            },
            {
                "transform": "CoarseDropout",
                "max_fraction": self.config.occlusion_max_fraction,
                "p": self.config.probability,
            },
        ]

    def apply(
        self,
        pixels: list[list[int]],
        rng: random.Random,
    ) -> list[list[int]]:
        h = len(pixels)
        w = len(pixels[0]) if h > 0 else 0
        output = [[pixels[y][x] for x in range(w)] for y in range(h)]
        if not self.config.enabled or h == 0 or w == 0:
            return output

        if rng.random() < self.config.probability:
            alpha = 1.0 + rng.uniform(-self.config.contrast_limit, self.config.contrast_limit)
            beta = 255.0 * rng.uniform(-self.config.brightness_limit, self.config.brightness_limit)
            for y in range(h):
                for x in range(w):
                    output[y][x] = clamp_byte(alpha * (output[y][x] - 128.0) + 128.0 + beta)

        if rng.random() < self.config.probability and self.config.max_rotation_degrees > 0.0:
            angle_rad = math.radians(
                rng.uniform(-self.config.max_rotation_degrees, self.config.max_rotation_degrees)
            )
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            cx = w / 2.0
            cy = h / 2.0
            rotated = [[output[y][x] for x in range(w)] for y in range(h)]
            for y in range(h):
                for x in range(w):
                    dx = x - cx
                    dy = y - cy
                    sx = round(cx + dx * cos_a + dy * sin_a)
                    sy = round(cy - dx * sin_a + dy * cos_a)
                    if 0 <= sx < w and 0 <= sy < h:
                        rotated[y][x] = output[sy][sx]
            output = rotated

        if rng.random() < self.config.probability and self.config.motion_blur_kernel > 1:
            k = self.config.motion_blur_kernel
            half = k // 2
            blurred = [[output[y][x] for x in range(w)] for y in range(h)]
            for y in range(h):
                for x in range(w):
                    acc = 0
                    cnt = 0
                    for offset in range(-half, half + 1):
                        nx = x + offset
                        if 0 <= nx < w:
                            acc += output[y][nx]
                            cnt += 1
                    blurred[y][x] = clamp_byte(acc / max(1, cnt))
            output = blurred

        if rng.random() < self.config.probability and self.config.occlusion_max_fraction > 0.0:
            occ_w = max(1, int(w * self.config.occlusion_max_fraction))
            occ_h = max(1, int(h * self.config.occlusion_max_fraction))
            ox = rng.randint(0, max(0, w - occ_w))
            oy = rng.randint(0, max(0, h - occ_h))
            fill_val = rng.randint(40, 180)
            for y in range(oy, min(h, oy + occ_h)):
                for x in range(ox, min(w, ox + occ_w)):
                    output[y][x] = fill_val

        if rng.random() < self.config.probability and self.config.noise_std_limit > 0.0:
            for y in range(h):
                for x in range(w):
                    output[y][x] = clamp_byte(
                        output[y][x] + rng.gauss(0.0, self.config.noise_std_limit)
                    )

        return output


__all__ = [
    "AugmentationConfig",
    "AugmentationPipeline",
]
