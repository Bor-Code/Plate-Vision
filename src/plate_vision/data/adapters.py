import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

from plate_vision.config.schema import DataConfig
from plate_vision.data.ingestion import ingest_manifest_records, inspect_image_bytes
from plate_vision.data.models import DatasetRecord
from plate_vision.utils.exceptions import DataIOError, DatasetValidationError

IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset({".png", ".jpg", ".jpeg", ".webp"})


class RealDatasetFormat(StrEnum):
    JSONL_MANIFEST = "jsonl_manifest"
    YOLO_DIRECTORY = "yolo_directory"
    OPENALPR_SIDECAR = "openalpr_sidecar"


@runtime_checkable
class RealDatasetAdapterProtocol(Protocol):
    def load_records(
        self,
        dataset_dir: Path,
        config: DataConfig,
    ) -> list[DatasetRecord]: ...


def _list_image_files(dataset_dir: Path) -> list[Path]:
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise DataIOError(
            f"Dataset directory does not exist: {dataset_dir}",
            details={"path": str(dataset_dir)},
        )
    direct = [
        p
        for p in sorted(dataset_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if direct:
        return direct
    images_subdir = dataset_dir / "images"
    if images_subdir.is_dir():
        return [
            p
            for p in sorted(images_subdir.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    return []


class JsonlManifestAdapter:
    def __init__(
        self,
        manifest_filename: str = "manifest.jsonl",
        source_dataset: str = "real_jsonl",
    ) -> None:
        self.manifest_filename = manifest_filename
        self.source_dataset = source_dataset

    def load_records(
        self,
        dataset_dir: Path,
        config: DataConfig,
    ) -> list[DatasetRecord]:
        manifest_path = dataset_dir / self.manifest_filename
        if not manifest_path.is_file():
            raise DataIOError(
                f"Manifest file does not exist: {manifest_path}",
                details={"path": str(manifest_path)},
            )
        try:
            lines = manifest_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise DataIOError(
                f"Failed to read manifest file: {manifest_path}",
                details={"path": str(manifest_path), "reason": str(exc)},
            ) from exc

        raw_records: list[dict[str, Any]] = []
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(
                    f"Malformed JSON on line {idx + 1} in {manifest_path}",
                    details={"path": str(manifest_path), "line": idx + 1, "reason": str(exc)},
                ) from exc
            if not isinstance(parsed, dict):
                raise DatasetValidationError(
                    f"Expected JSON object on line {idx + 1} in {manifest_path}",
                    details={"path": str(manifest_path), "line": idx + 1},
                )
            item = dict(parsed)
            if "image_path" in item:
                img_path = Path(str(item["image_path"]))
                if not img_path.is_absolute():
                    img_path = dataset_dir / img_path
                item["image_path"] = img_path
            item.setdefault("source_dataset", self.source_dataset)
            raw_records.append(item)

        return ingest_manifest_records(raw_records, config=config)


class YoloDirectoryAdapter:
    def __init__(
        self,
        default_country_code: str = "TR",
        source_dataset: str = "real_yolo",
    ) -> None:
        self.default_country_code = default_country_code.strip().upper()
        self.source_dataset = source_dataset

    def _find_label_file(self, dataset_dir: Path, image_path: Path) -> Path | None:
        adjacent = image_path.with_suffix(".txt")
        if adjacent.is_file():
            return adjacent
        labels_dir_file = dataset_dir / "labels" / f"{image_path.stem}.txt"
        if labels_dir_file.is_file():
            return labels_dir_file
        return None

    def load_records(
        self,
        dataset_dir: Path,
        config: DataConfig,
    ) -> list[DatasetRecord]:
        image_files = _list_image_files(dataset_dir)
        raw_records: list[dict[str, Any]] = []

        for idx, image_path in enumerate(image_files):
            try:
                image_bytes = image_path.read_bytes()
            except OSError as exc:
                raise DataIOError(
                    f"Failed to read image file: {image_path}",
                    details={"path": str(image_path), "reason": str(exc)},
                ) from exc

            width, height, _ = inspect_image_bytes(
                image_bytes,
                max_bytes=config.max_image_bytes,
                min_width=config.min_image_width,
                min_height=config.min_image_height,
                source_path=str(image_path),
            )

            annotations: list[dict[str, Any]] = []
            label_path = self._find_label_file(dataset_dir, image_path)
            if label_path is not None:
                try:
                    label_lines = label_path.read_text(encoding="utf-8").splitlines()
                except OSError as exc:
                    raise DataIOError(
                        f"Failed to read YOLO label file: {label_path}",
                        details={"path": str(label_path), "reason": str(exc)},
                    ) from exc

                for line_num, raw_line in enumerate(label_lines, start=1):
                    parts = raw_line.strip().split()
                    if not parts:
                        continue
                    if len(parts) < 5:
                        raise DatasetValidationError(
                            f"Invalid YOLO label columns in {label_path}:{line_num}",
                            details={"path": str(label_path), "line": line_num},
                        )
                    try:
                        class_id = int(parts[0])
                        xc = float(parts[1])
                        yc = float(parts[2])
                        nw = float(parts[3])
                        nh = float(parts[4])
                    except ValueError as exc:
                        raise DatasetValidationError(
                            f"Non-numeric YOLO coordinates in {label_path}:{line_num}",
                            details={"path": str(label_path), "line": line_num, "reason": str(exc)},
                        ) from exc

                    x_min = max(0.0, (xc - nw / 2.0) * width)
                    y_min = max(0.0, (yc - nh / 2.0) * height)
                    x_max = min(float(width), (xc + nw / 2.0) * width)
                    y_max = min(float(height), (yc + nh / 2.0) * height)
                    plate_text = parts[5] if len(parts) > 5 else "UNKNOWN"
                    country_code = parts[6].upper() if len(parts) > 6 else self.default_country_code
                    annotations.append(
                        {
                            "plate_text": plate_text,
                            "country_code": country_code,
                            "bbox": {
                                "x_min": round(x_min, 4),
                                "y_min": round(y_min, 4),
                                "x_max": round(x_max, 4),
                                "y_max": round(y_max, 4),
                                "class_id": class_id,
                            },
                        }
                    )

            raw_records.append(
                {
                    "record_id": f"{image_path.stem}_{idx:04d}",
                    "image_path": image_path,
                    "annotations": annotations,
                    "source_dataset": self.source_dataset,
                }
            )

        return ingest_manifest_records(raw_records, config=config)


class OpenAlprSidecarAdapter:
    def __init__(
        self,
        default_country_code: str = "TR",
        source_dataset: str = "real_openalpr",
    ) -> None:
        self.default_country_code = default_country_code.strip().upper()
        self.source_dataset = source_dataset

    def load_records(
        self,
        dataset_dir: Path,
        config: DataConfig,
    ) -> list[DatasetRecord]:
        image_files = _list_image_files(dataset_dir)
        raw_records: list[dict[str, Any]] = []

        for idx, image_path in enumerate(image_files):
            sidecar_path = image_path.with_suffix(".json")
            if not sidecar_path.is_file():
                continue
            try:
                sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))
            except OSError as exc:
                raise DataIOError(
                    f"Failed to read OpenALPR sidecar: {sidecar_path}",
                    details={"path": str(sidecar_path), "reason": str(exc)},
                ) from exc
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(
                    f"Malformed JSON in OpenALPR sidecar: {sidecar_path}",
                    details={"path": str(sidecar_path), "reason": str(exc)},
                ) from exc

            if not isinstance(sidecar_data, dict):
                raise DatasetValidationError(
                    f"Sidecar root must be a JSON object: {sidecar_path}",
                    details={"path": str(sidecar_path)},
                )
            plates = sidecar_data.get("plates", ())
            if not isinstance(plates, list):
                raise DatasetValidationError(
                    f"Sidecar 'plates' must be a list: {sidecar_path}",
                    details={"path": str(sidecar_path)},
                )

            annotations: list[dict[str, Any]] = []
            for plate in plates:
                if not isinstance(plate, dict) or "plate_text" not in plate or "bbox" not in plate:
                    raise DatasetValidationError(
                        f"Sidecar plate entry missing required fields: {sidecar_path}",
                        details={"path": str(sidecar_path)},
                    )
                ann: dict[str, Any] = {
                    "plate_text": str(plate["plate_text"]),
                    "country_code": str(
                        plate.get("country_code", self.default_country_code)
                    ).upper(),
                    "bbox": plate["bbox"],
                }
                if plate.get("corners") is not None:
                    ann["corners"] = plate["corners"]
                annotations.append(ann)

            raw_records.append(
                {
                    "record_id": f"{image_path.stem}_{idx:04d}",
                    "image_path": image_path,
                    "annotations": annotations,
                    "source_dataset": self.source_dataset,
                }
            )

        return ingest_manifest_records(raw_records, config=config)


def load_real_dataset(
    dataset_dir: Path,
    fmt: RealDatasetFormat = RealDatasetFormat.JSONL_MANIFEST,
    config: DataConfig | None = None,
    *,
    default_country_code: str = "TR",
) -> list[DatasetRecord]:
    active_config = config if config is not None else DataConfig()
    if fmt == RealDatasetFormat.JSONL_MANIFEST:
        adapter: RealDatasetAdapterProtocol = JsonlManifestAdapter()
    elif fmt == RealDatasetFormat.YOLO_DIRECTORY:
        adapter = YoloDirectoryAdapter(default_country_code=default_country_code)
    else:
        adapter = OpenAlprSidecarAdapter(default_country_code=default_country_code)
    return adapter.load_records(dataset_dir, active_config)


__all__ = [
    "IMAGE_EXTENSIONS",
    "JsonlManifestAdapter",
    "OpenAlprSidecarAdapter",
    "RealDatasetAdapterProtocol",
    "RealDatasetFormat",
    "YoloDirectoryAdapter",
    "load_real_dataset",
]
