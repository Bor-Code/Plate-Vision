from plate_vision.data.deduplication import deduplicate_records
from plate_vision.data.exporters import export_recognizer_manifests, export_yolo_dataset
from plate_vision.data.ingestion import (
    LocalFileStorage,
    derive_stratum_key,
    ingest_manifest_records,
    inspect_image_bytes,
    parse_jpeg_dimensions,
    parse_png_dimensions,
    parse_webp_dimensions,
    validate_record_annotations,
)
from plate_vision.data.models import (
    BoundingBox,
    DatasetRecord,
    DatasetSplit,
    DatasetStorageProtocol,
    DeduplicationResult,
    ImageMetadata,
    PlateAnnotation,
    SplitDataset,
)
from plate_vision.data.splitter import (
    allocate_stratum_counts,
    stratified_split,
    verify_no_split_leakage,
)

__all__ = [
    "BoundingBox",
    "DatasetRecord",
    "DatasetSplit",
    "DatasetStorageProtocol",
    "DeduplicationResult",
    "ImageMetadata",
    "LocalFileStorage",
    "PlateAnnotation",
    "SplitDataset",
    "allocate_stratum_counts",
    "deduplicate_records",
    "derive_stratum_key",
    "export_recognizer_manifests",
    "export_yolo_dataset",
    "ingest_manifest_records",
    "inspect_image_bytes",
    "parse_jpeg_dimensions",
    "parse_png_dimensions",
    "parse_webp_dimensions",
    "stratified_split",
    "validate_record_annotations",
    "verify_no_split_leakage",
]
