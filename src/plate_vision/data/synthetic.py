import hashlib
import math
import random
import struct
import zlib
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from plate_vision.data.models import (
    BoundingBox,
    DatasetRecord,
    ImageMetadata,
    PlateAnnotation,
)
from plate_vision.data.profiles import CountryProfileProtocol

GLYPH_HEIGHT: Final[int] = 7
GLYPH_WIDTH: Final[int] = 5

GLYPH_BITMAPS: Final[dict[str, tuple[int, ...]]] = {
    "0": (0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110),
    "1": (0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110),
    "2": (0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111),
    "3": (0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110),
    "4": (0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010),
    "5": (0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110),
    "6": (0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110),
    "7": (0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000),
    "8": (0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110),
    "9": (0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100),
    "A": (0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001),
    "B": (0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110),
    "C": (0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110),
    "D": (0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100),
    "E": (0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111),
    "F": (0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000),
    "G": (0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110),
    "H": (0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001),
    "I": (0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110),
    "J": (0b00111, 0b00010, 0b00010, 0b00010, 0b10010, 0b10010, 0b01100),
    "K": (0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001),
    "L": (0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111),
    "M": (0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001),
    "N": (0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001),
    "O": (0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110),
    "P": (0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000),
    "Q": (0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101),
    "R": (0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001),
    "S": (0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110),
    "T": (0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100),
    "U": (0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110),
    "V": (0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100),
    "W": (0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b01010),
    "X": (0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001),
    "Y": (0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100),
    "Z": (0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111),
}


class SyntheticPlateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    image_width: int = Field(default=128, ge=64)
    image_height: int = Field(default=32, ge=16)
    font_scale: int = Field(default=2, ge=1, le=4)
    background_min: int = Field(default=210, ge=128, le=255)
    background_max: int = Field(default=245, ge=128, le=255)
    foreground_min: int = Field(default=15, ge=0, le=100)
    foreground_max: int = Field(default=55, ge=0, le=100)
    perspective_shift: float = Field(default=0.06, ge=0.0, le=0.25)
    noise_std: float = Field(default=4.0, ge=0.0, le=50.0)
    blur_strength: float = Field(default=0.25, ge=0.0, le=1.0)
    lighting_gradient: float = Field(default=0.15, ge=0.0, le=0.5)
    dirt_spots: int = Field(default=2, ge=0, le=20)
    seed: int = Field(default=42, ge=0)


def clamp_byte(value: float) -> int:
    return max(0, min(255, round(value)))


def make_png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", crc)


def encode_grayscale_png(pixels: list[list[int]]) -> bytes:
    height = len(pixels)
    width = len(pixels[0]) if height > 0 else 0
    if width <= 0 or height <= 0:
        msg = "Pixel grid must have positive width and height"
        raise ValueError(msg)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    scanlines = bytearray()
    for row in pixels:
        scanlines.append(0)
        scanlines.extend(clamp_byte(val) for val in row)
    compressed = zlib.compress(bytes(scanlines), level=6)

    return (
        b"\x89PNG\r\n\x1a\n"
        + make_png_chunk(b"IHDR", ihdr)
        + make_png_chunk(b"IDAT", compressed)
        + make_png_chunk(b"IEND", b"")
    )


def render_plate_canvas(
    plate_text: str,
    config: SyntheticPlateConfig,
    rng: random.Random,
) -> tuple[list[list[int]], BoundingBox]:
    w = config.image_width
    h = config.image_height
    bg_lum = rng.randint(config.background_min, config.background_max)
    fg_lum = rng.randint(config.foreground_min, config.foreground_max)

    canvas = [[bg_lum for _ in range(w)] for _ in range(h)]

    band_width = max(6, w // 14)
    for y in range(h):
        for x in range(band_width):
            canvas[y][x] = max(0, fg_lum + 30)

    for x in range(w):
        canvas[0][x] = fg_lum
        canvas[h - 1][x] = fg_lum
    for y in range(h):
        canvas[y][0] = fg_lum
        canvas[y][w - 1] = fg_lum

    scale = config.font_scale
    char_w = GLYPH_WIDTH * scale
    char_h = GLYPH_HEIGHT * scale
    spacing = max(1, scale)
    total_text_w = len(plate_text) * char_w + max(0, len(plate_text) - 1) * spacing

    avail_w = w - band_width - 4
    start_x = band_width + 2 + max(0, (avail_w - total_text_w) // 2)
    start_y = max(1, (h - char_h) // 2)

    for idx, ch in enumerate(plate_text.upper()):
        bitmap = GLYPH_BITMAPS.get(ch, GLYPH_BITMAPS["0"])
        origin_x = start_x + idx * (char_w + spacing)
        for gy, row_bits in enumerate(bitmap):
            for gx in range(GLYPH_WIDTH):
                bit_set = (row_bits >> (GLYPH_WIDTH - 1 - gx)) & 1
                if bit_set:
                    for sy in range(scale):
                        for sx in range(scale):
                            py = start_y + gy * scale + sy
                            px = origin_x + gx * scale + sx
                            if 1 <= py < h - 1 and band_width <= px < w - 1:
                                canvas[py][px] = fg_lum

    bbox = BoundingBox(
        x_min=0.0,
        y_min=0.0,
        x_max=float(w),
        y_max=float(h),
        class_id=0,
    )
    return canvas, bbox


def apply_synthetic_effects(
    canvas: list[list[int]],
    config: SyntheticPlateConfig,
    rng: random.Random,
) -> list[list[int]]:
    h = len(canvas)
    w = len(canvas[0])

    shear_x = rng.uniform(-config.perspective_shift, config.perspective_shift)
    shear_y = rng.uniform(-config.perspective_shift * 0.5, config.perspective_shift * 0.5)
    warped = [[200 for _ in range(w)] for _ in range(h)]
    cx = w / 2.0
    cy = h / 2.0
    for y in range(h):
        for x in range(w):
            src_x = round(x + shear_x * (y - cy))
            src_y = round(y + shear_y * (x - cx))
            if 0 <= src_x < w and 0 <= src_y < h:
                warped[y][x] = canvas[src_y][src_x]

    grad_angle = rng.uniform(0.0, 2.0 * math.pi)
    cos_a = math.cos(grad_angle)
    sin_a = math.sin(grad_angle)
    for y in range(h):
        ny = (y / max(1, h - 1)) - 0.5
        for x in range(w):
            nx = (x / max(1, w - 1)) - 0.5
            factor = 1.0 + config.lighting_gradient * (nx * cos_a + ny * sin_a)
            warped[y][x] = clamp_byte(warped[y][x] * factor)

    for _ in range(config.dirt_spots):
        spot_x = rng.randint(2, w - 3)
        spot_y = rng.randint(2, h - 3)
        spot_lum = rng.randint(70, 140)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                py = spot_y + dy
                px = spot_x + dx
                if 0 <= py < h and 0 <= px < w:
                    warped[py][px] = spot_lum

    if config.blur_strength > 0.0:
        blurred = [[warped[y][x] for x in range(w)] for y in range(h)]
        alpha = config.blur_strength
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                neighbor_avg = (
                    warped[y][x]
                    + warped[y - 1][x]
                    + warped[y + 1][x]
                    + warped[y][x - 1]
                    + warped[y][x + 1]
                ) / 5.0
                blurred[y][x] = clamp_byte((1.0 - alpha) * warped[y][x] + alpha * neighbor_avg)
        warped = blurred

    if config.noise_std > 0.0:
        for y in range(h):
            for x in range(w):
                warped[y][x] = clamp_byte(warped[y][x] + rng.gauss(0.0, config.noise_std))

    return warped


def generate_synthetic_sample(
    plate_text: str,
    config: SyntheticPlateConfig,
    rng: random.Random,
) -> tuple[list[list[int]], bytes, BoundingBox]:
    canvas, bbox = render_plate_canvas(plate_text=plate_text, config=config, rng=rng)
    degraded = apply_synthetic_effects(canvas=canvas, config=config, rng=rng)
    png_bytes = encode_grayscale_png(degraded)
    return degraded, png_bytes, bbox


def generate_synthetic_dataset(
    count: int,
    output_dir: Path,
    config: SyntheticPlateConfig,
    profile: CountryProfileProtocol,
) -> list[DatasetRecord]:
    if count <= 0:
        msg = "count must be a positive integer"
        raise ValueError(msg)

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)
    records: list[DatasetRecord] = []

    for index in range(count):
        plate_text = profile.sample_plate(rng)
        _, png_bytes, bbox = generate_synthetic_sample(
            plate_text=plate_text,
            config=config,
            rng=rng,
        )
        file_path = output_dir / f"synth_{index:05d}_{plate_text}.png"
        file_path.write_bytes(png_bytes)
        digest = hashlib.sha256(png_bytes).hexdigest()

        annotation = PlateAnnotation(
            plate_text=plate_text,
            bbox=bbox,
            country_code=profile.country_code,
        )
        records.append(
            DatasetRecord(
                record_id=f"synth_{index:05d}",
                image=ImageMetadata(
                    path=file_path,
                    width=config.image_width,
                    height=config.image_height,
                    size_bytes=len(png_bytes),
                    sha256=digest,
                    mime_type="image/png",
                ),
                annotations=(annotation,),
                source_dataset="synthetic",
                stratum_key=f"synthetic:{profile.country_code}:{plate_text[:2]}",
            )
        )

    return records


__all__ = [
    "GLYPH_BITMAPS",
    "SyntheticPlateConfig",
    "apply_synthetic_effects",
    "encode_grayscale_png",
    "generate_synthetic_dataset",
    "generate_synthetic_sample",
    "render_plate_canvas",
]
