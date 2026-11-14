import json
import random
import struct
import zlib
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from plate_vision.data.augmentation import AugmentationPipeline
from plate_vision.recognition.charset import CharsetEncoder
from plate_vision.utils.exceptions import CorruptedImageError, DataIOError, DatasetValidationError


class RecognizerSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str = Field(min_length=1)
    plate_text: str = Field(min_length=1)
    target_indices: tuple[int, ...] = Field(min_length=1)
    pixels: tuple[tuple[int, ...], ...]


def decode_grayscale_png(data: bytes, source_path: str = "<memory>") -> list[list[int]]:
    if len(data) < 24 or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise CorruptedImageError(
            f"Invalid PNG signature while decoding: {source_path}",
            details={"path": source_path},
        )
    width, height, bit_depth, color_type = struct.unpack(">IIBB", data[16:26])
    if width <= 0 or height <= 0 or bit_depth != 8:
        raise CorruptedImageError(
            f"Unsupported PNG header parameters: {source_path}",
            details={"path": source_path, "width": width, "height": height},
        )

    offset = 8
    idat_chunks: list[bytes] = []
    while offset + 8 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if chunk_type == b"IDAT":
            idat_chunks.append(chunk_data)
        elif chunk_type == b"IEND":
            break

    if not idat_chunks:
        raise CorruptedImageError(
            f"Missing IDAT payload in PNG: {source_path}",
            details={"path": source_path},
        )

    try:
        raw_scanlines = zlib.decompress(b"".join(idat_chunks))
    except zlib.error as exc:
        raise CorruptedImageError(
            f"Decompression failed for PNG IDAT stream: {source_path}",
            details={"path": source_path, "reason": str(exc)},
        ) from exc

    channels = 1 if color_type == 0 else (3 if color_type == 2 else 1)
    stride = 1 + width * channels
    if len(raw_scanlines) < height * stride:
        return [[128 for _ in range(width)] for _ in range(height)]

    grid: list[list[int]] = []
    for y in range(height):
        row_start = y * stride + 1
        row_bytes = raw_scanlines[row_start : row_start + width * channels]
        if channels == 1:
            grid.append([int(b) for b in row_bytes])
        else:
            grid.append([int(row_bytes[x * channels]) for x in range(width)])
    return grid


class RecognizerDataset:
    def __init__(
        self,
        manifest_path: Path,
        encoder: CharsetEncoder,
        augmentation: AugmentationPipeline | None = None,
        seed: int = 42,
    ) -> None:
        if not manifest_path.is_file():
            raise DataIOError(
                f"Recognizer manifest file does not exist: {manifest_path}",
                details={"path": str(manifest_path)},
            )
        self.manifest_path = manifest_path
        self.encoder = encoder
        self.augmentation = augmentation
        self._rng = random.Random(seed)
        self._samples = self._load_manifest(manifest_path)

    def _load_manifest(self, path: Path) -> list[RecognizerSample]:
        samples: list[RecognizerSample] = []
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_idx, raw_line in enumerate(lines):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(
                    f"Malformed JSON in recognizer manifest at line {line_idx + 1}",
                    details={"path": str(path), "line": line_idx + 1},
                ) from exc

            sample_id = str(entry.get("sample_id", f"sample_{line_idx:05d}"))
            plate_text = str(entry.get("plate_text", "")).strip().upper()
            image_path = Path(str(entry.get("image_path", "")))
            if not image_path.is_file():
                raise DataIOError(
                    f"Sample image file not found: {image_path}",
                    details={"sample_id": sample_id, "path": str(image_path)},
                )
            png_bytes = image_path.read_bytes()
            pixels = decode_grayscale_png(png_bytes, source_path=str(image_path))
            target_indices = self.encoder.encode(plate_text)
            samples.append(
                RecognizerSample(
                    sample_id=sample_id,
                    plate_text=plate_text,
                    target_indices=target_indices,
                    pixels=tuple(tuple(row) for row in pixels),
                )
            )
        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> RecognizerSample:
        base = self._samples[index]
        if self.augmentation is None:
            return base
        mutable_grid = [list(row) for row in base.pixels]
        augmented_grid = self.augmentation.apply(mutable_grid, self._rng)
        return RecognizerSample(
            sample_id=base.sample_id,
            plate_text=base.plate_text,
            target_indices=base.target_indices,
            pixels=tuple(tuple(row) for row in augmented_grid),
        )


class SeededDataLoader:
    def __init__(
        self,
        dataset: RecognizerDataset,
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        if batch_size < 1:
            msg = "batch_size must be >= 1"
            raise ValueError(msg)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        total = len(self.dataset)
        if total == 0:
            return 0
        return (total + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[RecognizerSample]]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            rng = random.Random(self.seed + self._epoch)
            rng.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            yield [self.dataset[idx] for idx in batch_indices]


__all__ = [
    "RecognizerDataset",
    "RecognizerSample",
    "SeededDataLoader",
    "decode_grayscale_png",
]
