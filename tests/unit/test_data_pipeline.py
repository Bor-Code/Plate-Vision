import struct
from pathlib import Path
from unittest.mock import patch

import pytest

from plate_vision.config.schema import DataConfig, DatasetSplitRatios
from plate_vision.data import (
    BoundingBox,
    DatasetRecord,
    DatasetSplit,
    ImageMetadata,
    LocalFileStorage,
    PlateAnnotation,
    SplitDataset,
    allocate_stratum_counts,
    deduplicate_records,
    derive_stratum_key,
    ingest_manifest_records,
    inspect_image_bytes,
    parse_jpeg_dimensions,
    parse_png_dimensions,
    parse_webp_dimensions,
    stratified_split,
    verify_no_split_leakage,
)
from plate_vision.utils.exceptions import (
    CorruptedImageError,
    DataIOError,
    DatasetValidationError,
    InvalidInputError,
)


def make_png_bytes(width: int = 128, height: int = 64, extra: bytes = b"") -> bytes:
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr_chunk = struct.pack(">I", 13) + b"IHDR" + ihdr_data + b"\x00\x00\x00\x00"
    iend_chunk = struct.pack(">I", 0) + b"IEND" + b"\x00\x00\x00\x00"
    return b"\x89PNG\r\n\x1a\n" + ihdr_chunk + extra + iend_chunk


def make_jpeg_bytes(width: int = 160, height: int = 80) -> bytes:
    app0 = b"\xff\xe0" + struct.pack(">H", 4) + b"\x00\x00"
    sof0_payload = struct.pack(">BHHB", 8, height, width, 3)
    sof0 = b"\xff\xc0" + struct.pack(">H", len(sof0_payload) + 2) + sof0_payload
    return b"\xff\xd8" + app0 + sof0 + b"\xff\xd9"


def make_webp_vp8x_bytes(width: int = 120, height: int = 40) -> bytes:
    w_minus_1 = (width - 1).to_bytes(3, "little")
    h_minus_1 = (height - 1).to_bytes(3, "little")
    payload = b"VP8X" + struct.pack("<I", 10) + b"\x00\x00\x00\x00" + w_minus_1 + h_minus_1
    riff_size = struct.pack("<I", len(payload) + 4)
    return b"RIFF" + riff_size + b"WEBP" + payload


def make_webp_vp8_bytes(width: int = 100, height: int = 50) -> bytes:
    vp8_header = b"\x00\x00\x00\x9d\x01\x2a" + struct.pack("<HH", width, height)
    chunk = b"VP8 " + struct.pack("<I", len(vp8_header)) + vp8_header
    return b"RIFF" + struct.pack("<I", len(chunk) + 4) + b"WEBP" + chunk


def make_webp_vp8l_bytes(width: int = 96, height: int = 32) -> bytes:
    bits = ((width - 1) & 0x3FFF) | (((height - 1) & 0x3FFF) << 14)
    vp8l_body = b"\x2f" + struct.pack("<I", bits) + b"\x00\x00\x00\x00\x00"
    chunk = b"VP8L" + struct.pack("<I", len(vp8l_body)) + vp8l_body
    return b"RIFF" + struct.pack("<I", len(chunk) + 4) + b"WEBP" + chunk


def test_bounding_box_validation_and_yolo_conversion() -> None:
    with pytest.raises(ValueError, match="x_max"):
        BoundingBox(x_min=50.0, y_min=10.0, x_max=20.0, y_max=30.0)
    with pytest.raises(ValueError, match="y_max"):
        BoundingBox(x_min=10.0, y_min=40.0, x_max=50.0, y_max=20.0)

    bbox = BoundingBox(x_min=20.0, y_min=10.0, x_max=60.0, y_max=30.0, class_id=0)
    assert bbox.width == 40.0
    assert bbox.height == 20.0
    assert bbox.to_yolo(100, 100) == (0, 0.4, 0.2, 0.4, 0.2)
    assert bbox.format_yolo_line(100, 100) == "0 0.400000 0.200000 0.400000 0.200000"

    with pytest.raises(ValueError, match="positive integers"):
        bbox.to_yolo(0, 100)


def test_plate_annotation_rejects_blank_text() -> None:
    bbox = BoundingBox(x_min=0.0, y_min=0.0, x_max=10.0, y_max=10.0)
    with pytest.raises(ValueError, match="plate_text must not be blank"):
        PlateAnnotation(plate_text="   ", bbox=bbox)


def test_image_header_parsers_and_errors() -> None:
    assert parse_png_dimensions(make_png_bytes(200, 100), "ok.png") == (200, 100)
    with pytest.raises(CorruptedImageError, match="Truncated or malformed PNG"):
        parse_png_dimensions(b"\x89PNG\r\n\x1a\nshort", "bad.png")
    with pytest.raises(CorruptedImageError, match="Invalid PNG dimensions"):
        parse_png_dimensions(make_png_bytes(0, 100), "zero.png")

    assert parse_jpeg_dimensions(make_jpeg_bytes(320, 240), "ok.jpg") == (320, 240)
    with pytest.raises(CorruptedImageError, match="Truncated JPEG stream"):
        parse_jpeg_dimensions(b"\xff\xd8", "short.jpg")
    with pytest.raises(CorruptedImageError, match="Malformed JPEG segment length"):
        parse_jpeg_dimensions(b"\xff\xd8\xff\xe0\x00\x01\x00", "bad_len.jpg")
    with pytest.raises(CorruptedImageError, match="Truncated JPEG SOF segment"):
        parse_jpeg_dimensions(b"\xff\xd8\xff\xc0\x00\x04\x08\x00", "bad_sof.jpg")
    with pytest.raises(CorruptedImageError, match="Invalid JPEG dimensions"):
        parse_jpeg_dimensions(make_jpeg_bytes(0, 100), "zero.jpg")
    with pytest.raises(CorruptedImageError, match="No valid SOF marker"):
        parse_jpeg_dimensions(b"\xff\xd8\xff\xd8\xff\xd9", "no_sof.jpg")

    assert parse_webp_dimensions(make_webp_vp8x_bytes(140, 70), "vp8x.webp") == (140, 70)
    assert parse_webp_dimensions(make_webp_vp8_bytes(100, 50), "vp8.webp") == (100, 50)
    assert parse_webp_dimensions(make_webp_vp8l_bytes(96, 32), "vp8l.webp") == (96, 32)
    with pytest.raises(CorruptedImageError, match="Truncated WebP header"):
        parse_webp_dimensions(b"RIFF\x00\x00\x00\x00WEBP", "short.webp")
    bad_vp8 = b"RIFF" + struct.pack("<I", 26) + b"WEBPVP8 " + b"\x00" * 18
    with pytest.raises(CorruptedImageError, match="Invalid VP8 frame tag"):
        parse_webp_dimensions(bad_vp8, "bad_vp8.webp")
    bad_vp8l = b"RIFF" + struct.pack("<I", 26) + b"WEBPVP8L" + b"\x00" * 18
    with pytest.raises(CorruptedImageError, match="Invalid VP8L signature"):
        parse_webp_dimensions(bad_vp8l, "bad_vp8l.webp")
    unknown_webp = b"RIFF" + struct.pack("<I", 26) + b"WEBPVP8U" + b"\x00" * 18
    with pytest.raises(CorruptedImageError, match="Unsupported WebP chunk"):
        parse_webp_dimensions(unknown_webp, "unknown.webp")


def test_inspect_image_bytes_edge_cases() -> None:
    with pytest.raises(CorruptedImageError, match="Image payload is empty"):
        inspect_image_bytes(b"", max_bytes=1024, min_width=16, min_height=16)

    with pytest.raises(InvalidInputError, match="exceeds maximum allowed size"):
        inspect_image_bytes(make_png_bytes(64, 64), max_bytes=10, min_width=16, min_height=16)

    with pytest.raises(InvalidInputError, match="Unsupported or non-image payload"):
        inspect_image_bytes(b"NOT_AN_IMAGE_AT_ALL", max_bytes=1024, min_width=16, min_height=16)

    with pytest.raises(InvalidInputError, match="below minimum"):
        inspect_image_bytes(make_png_bytes(16, 8), max_bytes=1024, min_width=32, min_height=16)

    assert inspect_image_bytes(
        make_jpeg_bytes(64, 32),
        max_bytes=2048,
        min_width=32,
        min_height=16,
    ) == (64, 32, "image/jpeg")
    assert inspect_image_bytes(
        make_webp_vp8x_bytes(64, 32),
        max_bytes=2048,
        min_width=32,
        min_height=16,
    ) == (64, 32, "image/webp")


def test_local_file_storage_errors(tmp_path: Path) -> None:
    storage = LocalFileStorage()
    with pytest.raises(DataIOError, match="Image file does not exist"):
        storage.read_bytes(tmp_path / "missing.png")
    with pytest.raises(DataIOError, match="Path is not a regular file"):
        storage.read_bytes(tmp_path)

    sample = tmp_path / "sample.png"
    sample.write_bytes(make_png_bytes())
    with (
        patch.object(Path, "read_bytes", side_effect=OSError("disk error")),
        pytest.raises(DataIOError, match="Failed to read file bytes"),
    ):
        storage.read_bytes(sample)

    with (
        patch.object(Path, "write_text", side_effect=OSError("disk full")),
        pytest.raises(DataIOError, match="Failed to write file"),
    ):
        storage.write_text(tmp_path / "out.txt", "data")


def test_derive_stratum_key_and_ingest_errors(tmp_path: Path) -> None:
    bbox = BoundingBox(x_min=1.0, y_min=1.0, x_max=20.0, y_max=10.0)
    ann1 = PlateAnnotation(plate_text="34ABC123", bbox=bbox)
    ann2 = PlateAnnotation(plate_text="06DEF45", bbox=bbox)
    assert derive_stratum_key((), "ccpd") == "ccpd:negative"
    assert derive_stratum_key((ann1, ann2), "ccpd") == "ccpd:multi_plate"
    assert derive_stratum_key((ann1,), "tr_set") == "tr_set:TR:34"

    img_path = tmp_path / "img.png"
    img_path.write_bytes(make_png_bytes(64, 32))
    config = DataConfig()

    with pytest.raises(DatasetValidationError, match="record_id must not be empty"):
        ingest_manifest_records([{"record_id": "  ", "image_path": img_path}], config)

    with pytest.raises(DatasetValidationError, match="Missing or invalid image_path"):
        ingest_manifest_records([{"record_id": "r1", "image_path": 123}], config)

    with pytest.raises(DatasetValidationError, match="Annotations must be a sequence"):
        ingest_manifest_records(
            [{"record_id": "r1", "image_path": img_path, "annotations": "invalid"}],
            config,
        )

    with pytest.raises(DatasetValidationError, match="Invalid annotation schema"):
        ingest_manifest_records(
            [
                {
                    "record_id": "r1",
                    "image_path": img_path,
                    "annotations": [{"plate_text": "", "bbox": {}}],
                }
            ],
            config,
        )

    with pytest.raises(DatasetValidationError, match="Bounding box exceeds image boundaries"):
        ingest_manifest_records(
            [
                {
                    "record_id": "r1",
                    "image_path": img_path,
                    "annotations": [
                        {
                            "plate_text": "34AB123",
                            "bbox": {"x_min": 0.0, "y_min": 0.0, "x_max": 999.0, "y_max": 10.0},
                        }
                    ],
                }
            ],
            config,
        )


def test_deduplication_and_splitter_edge_cases() -> None:
    ratios = DatasetSplitRatios(train=0.70, val=0.15, test=0.15)
    assert allocate_stratum_counts(0, ratios) == (0, 0, 0)
    assert allocate_stratum_counts(1, ratios) == (1, 0, 0)
    assert allocate_stratum_counts(2, ratios) == (1, 1, 0)
    assert sum(allocate_stratum_counts(10, ratios)) == 10

    with pytest.raises(DatasetValidationError, match="Cannot split an empty dataset"):
        stratified_split([], ratios, seed=42)

    meta = ImageMetadata(
        path=Path("img.png"),
        width=64,
        height=32,
        size_bytes=100,
        sha256="a" * 64,
        mime_type="image/png",
    )
    rec1 = DatasetRecord(record_id="rec_01", image=meta)
    rec2 = DatasetRecord(record_id="rec_02", image=meta)
    dedup = deduplicate_records([rec2, rec1])
    assert dedup.removed_count == 1
    assert dedup.unique_records == (rec1,)
    assert dedup.duplicate_record_ids == ("rec_02",)

    leaking_split = SplitDataset(train=(rec1,), val=(rec2,), test=())
    assert leaking_split.for_split(DatasetSplit.TRAIN) == (rec1,)
    assert leaking_split.for_split(DatasetSplit.VAL) == (rec2,)
    assert leaking_split.for_split(DatasetSplit.TEST) == ()
    with pytest.raises(DatasetValidationError, match="Duplicate image hashes detected"):
        verify_no_split_leakage(leaking_split)
