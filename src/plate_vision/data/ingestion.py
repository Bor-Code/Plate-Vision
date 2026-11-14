import hashlib
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from pydantic import ValidationError

from plate_vision.config.schema import DataConfig
from plate_vision.data.models import (
    DatasetRecord,
    DatasetStorageProtocol,
    ImageMetadata,
    PlateAnnotation,
)
from plate_vision.utils.exceptions import (
    CorruptedImageError,
    DataIOError,
    DatasetValidationError,
    InvalidInputError,
)

PNG_SIGNATURE: Final[bytes] = b"\x89PNG\r\n\x1a\n"
JPEG_SOI: Final[bytes] = b"\xff\xd8"
RIFF_SIGNATURE: Final[bytes] = b"RIFF"
WEBP_SIGNATURE: Final[bytes] = b"WEBP"
JPEG_SOF_MARKERS: Final[frozenset[int]] = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)


class LocalFileStorage:
    def read_bytes(self, path: Path) -> bytes:
        if not path.exists():
            raise DataIOError(
                f"Image file does not exist: {path}",
                details={"path": str(path)},
            )
        if not path.is_file():
            raise DataIOError(
                f"Path is not a regular file: {path}",
                details={"path": str(path)},
            )
        try:
            return path.read_bytes()
        except OSError as exc:
            raise DataIOError(
                f"Failed to read file bytes: {path}",
                details={"path": str(path), "reason": str(exc)},
            ) from exc

    def write_text(self, path: Path, content: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise DataIOError(
                f"Failed to write file: {path}",
                details={"path": str(path), "reason": str(exc)},
            ) from exc


def parse_png_dimensions(data: bytes, source_path: str) -> tuple[int, int]:
    min_png_header_size = 24
    if len(data) < min_png_header_size or data[12:16] != b"IHDR":
        raise CorruptedImageError(
            f"Truncated or malformed PNG header: {source_path}",
            details={"path": source_path},
        )
    width, height = struct.unpack(">II", data[16:24])
    if width <= 0 or height <= 0:
        raise CorruptedImageError(
            f"Invalid PNG dimensions in header: {source_path}",
            details={"path": source_path, "width": width, "height": height},
        )
    return int(width), int(height)


def parse_jpeg_dimensions(data: bytes, source_path: str) -> tuple[int, int]:
    if len(data) < 4:
        raise CorruptedImageError(
            f"Truncated JPEG stream: {source_path}",
            details={"path": source_path},
        )
    offset = 2
    length = len(data)
    while offset < length:
        while offset < length and data[offset] == 0xFF:
            offset += 1
        if offset >= length:
            break
        marker = data[offset]
        offset += 1
        if marker in {0xD8, 0xD9} or (0xD0 <= marker <= 0xD7):
            continue
        if offset + 2 > length:
            break
        segment_length = struct.unpack(">H", data[offset : offset + 2])[0]
        if segment_length < 2 or offset + segment_length > length:
            raise CorruptedImageError(
                f"Malformed JPEG segment length: {source_path}",
                details={"path": source_path},
            )
        if marker in JPEG_SOF_MARKERS:
            if segment_length < 7:
                raise CorruptedImageError(
                    f"Truncated JPEG SOF segment: {source_path}",
                    details={"path": source_path},
                )
            height, width = struct.unpack(">HH", data[offset + 3 : offset + 7])
            if width <= 0 or height <= 0:
                raise CorruptedImageError(
                    f"Invalid JPEG dimensions: {source_path}",
                    details={"path": source_path, "width": width, "height": height},
                )
            return int(width), int(height)
        offset += segment_length

    raise CorruptedImageError(
        f"No valid SOF marker found in JPEG: {source_path}",
        details={"path": source_path},
    )


def parse_webp_dimensions(data: bytes, source_path: str) -> tuple[int, int]:
    if len(data) < 30:
        raise CorruptedImageError(
            f"Truncated WebP header: {source_path}",
            details={"path": source_path},
        )
    chunk_type = data[12:16]
    if chunk_type == b"VP8X":
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
        return width, height
    if chunk_type == b"VP8 ":
        if data[23:26] != b"\x9d\x01\x2a":
            raise CorruptedImageError(
                f"Invalid VP8 frame tag in WebP: {source_path}",
                details={"path": source_path},
            )
        raw_w, raw_h = struct.unpack("<HH", data[26:30])
        return int(raw_w & 0x3FFF), int(raw_h & 0x3FFF)
    if chunk_type == b"VP8L":
        if data[20] != 0x2F:
            raise CorruptedImageError(
                f"Invalid VP8L signature in WebP: {source_path}",
                details={"path": source_path},
            )
        bits = struct.unpack("<I", data[21:25])[0]
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return int(width), int(height)

    raise CorruptedImageError(
        f"Unsupported WebP chunk type: {source_path}",
        details={"path": source_path},
    )


def inspect_image_bytes(
    data: bytes,
    *,
    max_bytes: int,
    min_width: int,
    min_height: int,
    source_path: str = "<memory>",
) -> tuple[int, int, str]:
    size_bytes = len(data)
    if size_bytes == 0:
        raise CorruptedImageError(
            f"Image payload is empty: {source_path}",
            details={"path": source_path},
        )
    if size_bytes > max_bytes:
        raise InvalidInputError(
            f"Image payload exceeds maximum allowed size ({size_bytes} > {max_bytes})",
            details={"path": source_path, "size_bytes": size_bytes, "max_bytes": max_bytes},
        )

    if data.startswith(PNG_SIGNATURE):
        width, height = parse_png_dimensions(data, source_path)
        mime_type = "image/png"
    elif data.startswith(JPEG_SOI):
        width, height = parse_jpeg_dimensions(data, source_path)
        mime_type = "image/jpeg"
    elif len(data) >= 12 and data[:4] == RIFF_SIGNATURE and data[8:12] == WEBP_SIGNATURE:
        width, height = parse_webp_dimensions(data, source_path)
        mime_type = "image/webp"
    else:
        raise InvalidInputError(
            f"Unsupported or non-image payload format: {source_path}",
            details={"path": source_path},
        )

    if width < min_width or height < min_height:
        raise InvalidInputError(
            f"Image dimensions ({width}x{height}) below minimum ({min_width}x{min_height})",
            details={
                "path": source_path,
                "width": width,
                "height": height,
                "min_width": min_width,
                "min_height": min_height,
            },
        )

    return width, height, mime_type


def validate_record_annotations(
    annotations: Sequence[PlateAnnotation],
    *,
    image_width: int,
    image_height: int,
    record_id: str,
) -> tuple[PlateAnnotation, ...]:
    validated: list[PlateAnnotation] = []
    for annotation in annotations:
        bbox = annotation.bbox
        if bbox.x_max > float(image_width) or bbox.y_max > float(image_height):
            raise DatasetValidationError(
                f"Bounding box exceeds image boundaries in record {record_id}",
                details={
                    "record_id": record_id,
                    "bbox": bbox.model_dump(),
                    "image_width": image_width,
                    "image_height": image_height,
                },
            )
        validated.append(annotation)
    return tuple(validated)


def derive_stratum_key(annotations: Sequence[PlateAnnotation], source_dataset: str) -> str:
    if not annotations:
        return f"{source_dataset}:negative"
    if len(annotations) > 1:
        return f"{source_dataset}:multi_plate"
    primary_text = annotations[0].plate_text.strip().upper()
    prefix = primary_text[:2] if len(primary_text) >= 2 else primary_text
    return f"{source_dataset}:{annotations[0].country_code}:{prefix}"


def ingest_manifest_records(
    raw_records: Sequence[Mapping[str, Any]],
    config: DataConfig,
    storage: DatasetStorageProtocol | None = None,
) -> list[DatasetRecord]:
    active_storage: DatasetStorageProtocol = storage if storage is not None else LocalFileStorage()
    ingested: list[DatasetRecord] = []

    for index, raw_item in enumerate(raw_records):
        record_id = str(raw_item.get("record_id", f"record_{index:06d}")).strip()
        if not record_id:
            raise DatasetValidationError(
                "Dataset record_id must not be empty",
                details={"index": index},
            )
        raw_path = raw_item.get("image_path")
        if not isinstance(raw_path, str | Path):
            raise DatasetValidationError(
                f"Missing or invalid image_path in record {record_id}",
                details={"record_id": record_id},
            )

        image_path = Path(raw_path)
        image_bytes = active_storage.read_bytes(image_path)
        width, height, mime_type = inspect_image_bytes(
            image_bytes,
            max_bytes=config.max_image_bytes,
            min_width=config.min_image_width,
            min_height=config.min_image_height,
            source_path=str(image_path),
        )
        digest = hashlib.sha256(image_bytes).hexdigest()

        raw_annotations = raw_item.get("annotations", ())
        if not isinstance(raw_annotations, Sequence) or isinstance(raw_annotations, str | bytes):
            raise DatasetValidationError(
                f"Annotations must be a sequence in record {record_id}",
                details={"record_id": record_id},
            )

        try:
            parsed_annotations = [
                item if isinstance(item, PlateAnnotation) else PlateAnnotation.model_validate(item)
                for item in raw_annotations
            ]
        except ValidationError as exc:
            raise DatasetValidationError(
                f"Invalid annotation schema in record {record_id}",
                details={"record_id": record_id, "errors": exc.errors()},
            ) from exc

        validated_annotations = validate_record_annotations(
            parsed_annotations,
            image_width=width,
            image_height=height,
            record_id=record_id,
        )
        source_dataset = str(raw_item.get("source_dataset", "default")).strip() or "default"
        explicit_stratum = raw_item.get("stratum_key")
        stratum_key = (
            str(explicit_stratum).strip()
            if isinstance(explicit_stratum, str) and explicit_stratum.strip()
            else derive_stratum_key(validated_annotations, source_dataset)
        )

        record = DatasetRecord(
            record_id=record_id,
            image=ImageMetadata(
                path=image_path,
                width=width,
                height=height,
                size_bytes=len(image_bytes),
                sha256=digest,
                mime_type=mime_type,
            ),
            annotations=validated_annotations,
            source_dataset=source_dataset,
            stratum_key=stratum_key,
        )
        ingested.append(record)

    return ingested


__all__ = [
    "LocalFileStorage",
    "derive_stratum_key",
    "ingest_manifest_records",
    "inspect_image_bytes",
    "parse_jpeg_dimensions",
    "parse_png_dimensions",
    "parse_webp_dimensions",
    "validate_record_annotations",
]
