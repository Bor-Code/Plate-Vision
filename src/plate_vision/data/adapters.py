import json
import xml.etree.ElementTree as ET
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

from plate_vision.config.schema import DataConfig
from plate_vision.data.ingestion import ingest_manifest_records, inspect_image_bytes
from plate_vision.data.models import DatasetRecord
from plate_vision.utils.exceptions import DataIOError, DatasetValidationError

IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset({".png", ".jpg", ".jpeg", ".webp"})
CCPD_PROVINCE_CODES: Final[tuple[str, ...]] = (
    "AH",
    "SH",
    "TJ",
    "CQ",
    "HE",
    "SX",
    "NM",
    "LN",
    "JL",
    "HL",
    "JS",
    "ZJ",
    "BJ",
    "FJ",
    "JX",
    "SD",
    "HA",
    "HB",
    "HN",
    "GD",
    "GX",
    "HI",
    "SC",
    "GZ",
    "YN",
    "XZ",
    "SN",
    "GS",
    "QH",
    "NX",
    "XJ",
    "WJ",
    "XY",
    "O",
)
CCPD_ALPHABETS: Final[str] = "ABCDEFGHJKLMNPQRSTUVWXYZ0"
CCPD_ADS: Final[str] = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789O"
_VOC_GENERIC_LABELS: Final[frozenset[str]] = frozenset(
    {"plate", "license_plate", "license-plate", "number_plate", "licence"}
)


class RealDatasetFormat(StrEnum):
    JSONL_MANIFEST = "jsonl_manifest"
    YOLO_DIRECTORY = "yolo_directory"
    OPENALPR_SIDECAR = "openalpr_sidecar"
    PASCAL_VOC_XML = "pascal_voc_xml"
    CCPD_FILENAME = "ccpd_filename"
    COCO_JSON = "coco_json"


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
    for folder_name in ("images", "JPEGImages"):
        subdir = dataset_dir / folder_name
        if subdir.is_dir():
            files = [
                p
                for p in sorted(subdir.iterdir())
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
            if files:
                return files
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


def _parse_ccpd_pair(token: str, source_name: str, field_name: str) -> tuple[float, float]:
    chunks = token.split("&")
    if len(chunks) != 2:
        raise DatasetValidationError(
            f"Malformed CCPD {field_name} pair '{token}' in {source_name}",
            details={"filename": source_name, "field": field_name, "token": token},
        )
    try:
        return float(chunks[0]), float(chunks[1])
    except ValueError as exc:
        raise DatasetValidationError(
            f"Non-numeric CCPD {field_name} pair '{token}' in {source_name}",
            details={"filename": source_name, "field": field_name, "token": token},
        ) from exc


def parse_ccpd_filename(filename: str | Path) -> dict[str, Any]:
    path = Path(filename)
    stem = path.stem
    parts = stem.split("-")
    if len(parts) != 7:
        raise DatasetValidationError(
            f"CCPD filename must have 7 hyphen-separated parts: {path.name}",
            details={"filename": path.name, "part_count": len(parts)},
        )

    try:
        area = float(parts[0])
        tilt_tokens = parts[1].split("_")
        if len(tilt_tokens) != 2:
            raise ValueError("tilt must have 2 parts")
        tilt = (float(tilt_tokens[0]), float(tilt_tokens[1]))
        brightness = int(parts[5])
        blurriness = int(parts[6])
    except ValueError as exc:
        raise DatasetValidationError(
            f"Invalid numeric metadata in CCPD filename: {path.name}",
            details={"filename": path.name, "reason": str(exc)},
        ) from exc

    bbox_tokens = parts[2].split("_")
    if len(bbox_tokens) != 2:
        raise DatasetValidationError(
            f"CCPD bbox field must have 2 vertices: {path.name}",
            details={"filename": path.name, "bbox": parts[2]},
        )
    x1, y1 = _parse_ccpd_pair(bbox_tokens[0], path.name, "bbox")
    x2, y2 = _parse_ccpd_pair(bbox_tokens[1], path.name, "bbox")
    if x1 < 0.0 or y1 < 0.0 or x2 <= x1 or y2 <= y1:
        raise DatasetValidationError(
            f"Invalid CCPD bounding box coordinates in {path.name}",
            details={"filename": path.name, "bbox": [x1, y1, x2, y2]},
        )

    vertex_tokens = parts[3].split("_")
    if len(vertex_tokens) != 4:
        raise DatasetValidationError(
            f"CCPD vertices field must have 4 corners: {path.name}",
            details={"filename": path.name, "vertices": parts[3]},
        )
    br = _parse_ccpd_pair(vertex_tokens[0], path.name, "vertex")
    bl = _parse_ccpd_pair(vertex_tokens[1], path.name, "vertex")
    tl = _parse_ccpd_pair(vertex_tokens[2], path.name, "vertex")
    tr = _parse_ccpd_pair(vertex_tokens[3], path.name, "vertex")

    char_tokens = parts[4].split("_")
    if len(char_tokens) < 7:
        raise DatasetValidationError(
            f"CCPD plate code must have at least 7 character indices: {path.name}",
            details={"filename": path.name, "plate_indices": parts[4]},
        )
    try:
        char_indices = [int(item) for item in char_tokens]
    except ValueError as exc:
        raise DatasetValidationError(
            f"Non-integer CCPD character index in {path.name}",
            details={"filename": path.name, "plate_indices": parts[4]},
        ) from exc

    province_idx = char_indices[0]
    alpha_idx = char_indices[1]
    if not (0 <= province_idx < len(CCPD_PROVINCE_CODES)):
        raise DatasetValidationError(
            f"CCPD province index out of range in {path.name}: {province_idx}",
            details={"filename": path.name, "province_idx": province_idx},
        )
    if not (0 <= alpha_idx < len(CCPD_ALPHABETS)):
        raise DatasetValidationError(
            f"CCPD alphabet index out of range in {path.name}: {alpha_idx}",
            details={"filename": path.name, "alpha_idx": alpha_idx},
        )

    chars: list[str] = [CCPD_PROVINCE_CODES[province_idx], CCPD_ALPHABETS[alpha_idx]]
    for ad_idx in char_indices[2:]:
        if not (0 <= ad_idx < len(CCPD_ADS)):
            raise DatasetValidationError(
                f"CCPD character index out of range in {path.name}: {ad_idx}",
                details={"filename": path.name, "ad_idx": ad_idx},
            )
        chars.append(CCPD_ADS[ad_idx])

    return {
        "plate_text": "".join(chars),
        "bbox": {
            "x_min": round(x1, 4),
            "y_min": round(y1, 4),
            "x_max": round(x2, 4),
            "y_max": round(y2, 4),
            "class_id": 0,
        },
        "corners": (tl, tr, br, bl),
        "area": area,
        "tilt": tilt,
        "brightness": brightness,
        "blurriness": blurriness,
    }


class PascalVocXmlAdapter:
    def __init__(
        self,
        default_country_code: str = "TR",
        annotations_dirname: str = "annotations",
        source_dataset: str = "real_pascal_voc",
        default_plate_text: str = "UNKNOWN",
    ) -> None:
        self.default_country_code = default_country_code.strip().upper()
        self.annotations_dirname = annotations_dirname
        self.source_dataset = source_dataset
        self.default_plate_text = default_plate_text

    def _find_xml_file(self, dataset_dir: Path, image_path: Path) -> Path | None:
        candidates = (
            image_path.with_suffix(".xml"),
            dataset_dir / self.annotations_dirname / f"{image_path.stem}.xml",
            dataset_dir / "Annotations" / f"{image_path.stem}.xml",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def load_records(
        self,
        dataset_dir: Path,
        config: DataConfig,
    ) -> list[DatasetRecord]:
        image_files = _list_image_files(dataset_dir)
        raw_records: list[dict[str, Any]] = []

        for idx, image_path in enumerate(image_files):
            xml_path = self._find_xml_file(dataset_dir, image_path)
            if xml_path is None:
                continue
            try:
                xml_content = xml_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise DataIOError(
                    f"Failed to read Pascal VOC XML file: {xml_path}",
                    details={"path": str(xml_path), "reason": str(exc)},
                ) from exc

            try:
                root = ET.fromstring(xml_content)
            except ET.ParseError as exc:
                raise DatasetValidationError(
                    f"Malformed Pascal VOC XML file: {xml_path}",
                    details={"path": str(xml_path), "reason": str(exc)},
                ) from exc

            annotations: list[dict[str, Any]] = []
            for obj in root.findall("object"):
                bndbox = obj.find("bndbox")
                if bndbox is None:
                    raise DatasetValidationError(
                        f"VOC <object> missing <bndbox> in {xml_path}",
                        details={"path": str(xml_path)},
                    )
                try:
                    x_min = float(bndbox.findtext("xmin", "").strip())
                    y_min = float(bndbox.findtext("ymin", "").strip())
                    x_max = float(bndbox.findtext("xmax", "").strip())
                    y_max = float(bndbox.findtext("ymax", "").strip())
                except ValueError as exc:
                    raise DatasetValidationError(
                        f"Invalid <bndbox> coordinates in {xml_path}",
                        details={"path": str(xml_path), "reason": str(exc)},
                    ) from exc

                explicit_text = (obj.findtext("plate_text") or obj.findtext("text") or "").strip()
                if explicit_text:
                    plate_text = explicit_text
                else:
                    label_name = (obj.findtext("name") or "").strip()
                    plate_text = (
                        self.default_plate_text
                        if not label_name or label_name.lower() in _VOC_GENERIC_LABELS
                        else label_name
                    )

                country_text = (obj.findtext("country_code") or "").strip().upper()
                annotations.append(
                    {
                        "plate_text": plate_text,
                        "country_code": country_text or self.default_country_code,
                        "bbox": {
                            "x_min": round(x_min, 4),
                            "y_min": round(y_min, 4),
                            "x_max": round(x_max, 4),
                            "y_max": round(y_max, 4),
                            "class_id": 0,
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


class CcpdFilenameAdapter:
    def __init__(
        self,
        default_country_code: str = "CN",
        source_dataset: str = "real_ccpd",
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
            parsed = parse_ccpd_filename(image_path)
            raw_records.append(
                {
                    "record_id": f"{image_path.stem}_{idx:04d}",
                    "image_path": image_path,
                    "annotations": [
                        {
                            "plate_text": parsed["plate_text"],
                            "country_code": self.default_country_code,
                            "bbox": parsed["bbox"],
                            "corners": parsed["corners"],
                        }
                    ],
                    "source_dataset": self.source_dataset,
                }
            )

        return ingest_manifest_records(raw_records, config=config)


class CocoJsonAdapter:
    def __init__(
        self,
        annotations_filename: str | None = None,
        default_country_code: str = "TR",
        source_dataset: str = "real_coco",
        default_plate_text: str = "UNKNOWN",
    ) -> None:
        self.annotations_filename = annotations_filename
        self.default_country_code = default_country_code.strip().upper()
        self.source_dataset = source_dataset
        self.default_plate_text = default_plate_text

    def _find_coco_file(self, dataset_dir: Path) -> Path:
        if self.annotations_filename is not None:
            explicit = dataset_dir / self.annotations_filename
            if explicit.is_file():
                return explicit
            raise DataIOError(
                f"COCO annotations file does not exist: {explicit}",
                details={"path": str(explicit)},
            )
        candidates = (
            dataset_dir / "_annotations.coco.json",
            dataset_dir / "annotations.json",
            dataset_dir / "annotations" / "instances_default.json",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise DataIOError(
            f"No COCO annotations file found in directory: {dataset_dir}",
            details={"path": str(dataset_dir)},
        )

    @staticmethod
    def _resolve_image_path(dataset_dir: Path, file_name: str) -> Path:
        direct = dataset_dir / file_name
        if direct.is_file():
            return direct
        return dataset_dir / "images" / file_name

    @staticmethod
    def _extract_corners(ann: dict[str, Any]) -> tuple[tuple[float, float], ...] | None:
        raw_corners = ann.get("corners")
        if isinstance(raw_corners, list) and len(raw_corners) == 4:
            parsed_corners: list[tuple[float, float]] = []
            for pt in raw_corners:
                if not isinstance(pt, list | tuple) or len(pt) != 2:
                    return None
                parsed_corners.append((float(pt[0]), float(pt[1])))
            return tuple(parsed_corners)

        seg = ann.get("segmentation")
        if isinstance(seg, list) and seg and isinstance(seg[0], list) and len(seg[0]) == 8:
            coords = seg[0]
            return (
                (float(coords[0]), float(coords[1])),
                (float(coords[2]), float(coords[3])),
                (float(coords[4]), float(coords[5])),
                (float(coords[6]), float(coords[7])),
            )
        return None

    def load_records(
        self,
        dataset_dir: Path,
        config: DataConfig,
    ) -> list[DatasetRecord]:
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            raise DataIOError(
                f"Dataset directory does not exist: {dataset_dir}",
                details={"path": str(dataset_dir)},
            )

        coco_path = self._find_coco_file(dataset_dir)
        try:
            payload = json.loads(coco_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise DataIOError(
                f"Failed to read COCO annotations file: {coco_path}",
                details={"path": str(coco_path), "reason": str(exc)},
            ) from exc
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(
                f"Malformed JSON in COCO annotations file: {coco_path}",
                details={"path": str(coco_path), "reason": str(exc)},
            ) from exc

        if not isinstance(payload, dict):
            raise DatasetValidationError(
                f"COCO annotations root must be a JSON object: {coco_path}",
                details={"path": str(coco_path)},
            )

        images = payload.get("images")
        annotations = payload.get("annotations")
        if not isinstance(images, list) or not isinstance(annotations, list):
            raise DatasetValidationError(
                f"COCO file must contain 'images' and 'annotations' lists: {coco_path}",
                details={"path": str(coco_path)},
            )

        image_lookup: dict[Any, Path] = {}
        for img in images:
            if not isinstance(img, dict) or "id" not in img or "file_name" not in img:
                raise DatasetValidationError(
                    f"COCO image entry missing 'id' or 'file_name': {coco_path}",
                    details={"path": str(coco_path)},
                )
            image_lookup[img["id"]] = self._resolve_image_path(dataset_dir, str(img["file_name"]))

        grouped_annotations: dict[Any, list[dict[str, Any]]] = {
            img_id: [] for img_id in image_lookup
        }
        for ann in annotations:
            if not isinstance(ann, dict) or "image_id" not in ann or "bbox" not in ann:
                raise DatasetValidationError(
                    f"COCO annotation missing 'image_id' or 'bbox': {coco_path}",
                    details={"path": str(coco_path)},
                )
            image_id = ann["image_id"]
            if image_id not in grouped_annotations:
                raise DatasetValidationError(
                    f"COCO annotation references unknown image_id {image_id}: {coco_path}",
                    details={"path": str(coco_path), "image_id": image_id},
                )
            raw_bbox = ann["bbox"]
            if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
                raise DatasetValidationError(
                    f"COCO bbox must be a 4-element list [x, y, w, h]: {coco_path}",
                    details={"path": str(coco_path), "bbox": raw_bbox},
                )
            try:
                x_min = float(raw_bbox[0])
                y_min = float(raw_bbox[1])
                box_w = float(raw_bbox[2])
                box_h = float(raw_bbox[3])
            except (TypeError, ValueError) as exc:
                raise DatasetValidationError(
                    f"Non-numeric COCO bbox values in {coco_path}",
                    details={"path": str(coco_path), "bbox": raw_bbox},
                ) from exc
            if box_w <= 0.0 or box_h <= 0.0 or x_min < 0.0 or y_min < 0.0:
                raise DatasetValidationError(
                    f"Invalid COCO bbox dimensions in {coco_path}",
                    details={"path": str(coco_path), "bbox": raw_bbox},
                )

            attributes = ann.get("attributes")
            attr_map = attributes if isinstance(attributes, dict) else {}
            raw_plate = (
                ann.get("plate_text")
                or attr_map.get("plate_text")
                or attr_map.get("text")
                or self.default_plate_text
            )
            raw_country = (
                ann.get("country_code") or attr_map.get("country_code") or self.default_country_code
            )
            entry: dict[str, Any] = {
                "plate_text": str(raw_plate).strip() or self.default_plate_text,
                "country_code": str(raw_country).strip().upper() or self.default_country_code,
                "bbox": {
                    "x_min": round(x_min, 4),
                    "y_min": round(y_min, 4),
                    "x_max": round(x_min + box_w, 4),
                    "y_max": round(y_min + box_h, 4),
                    "class_id": int(ann.get("category_id", 0)),
                },
            }
            corners = self._extract_corners(ann)
            if corners is not None:
                entry["corners"] = corners
            grouped_annotations[image_id].append(entry)

        raw_records: list[dict[str, Any]] = []
        for idx, (image_id, image_path) in enumerate(image_lookup.items()):
            raw_records.append(
                {
                    "record_id": f"{image_path.stem}_{idx:04d}",
                    "image_path": image_path,
                    "annotations": grouped_annotations[image_id],
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
    elif fmt == RealDatasetFormat.OPENALPR_SIDECAR:
        adapter = OpenAlprSidecarAdapter(default_country_code=default_country_code)
    elif fmt == RealDatasetFormat.PASCAL_VOC_XML:
        adapter = PascalVocXmlAdapter(default_country_code=default_country_code)
    elif fmt == RealDatasetFormat.CCPD_FILENAME:
        adapter = CcpdFilenameAdapter(default_country_code=default_country_code)
    else:
        adapter = CocoJsonAdapter(default_country_code=default_country_code)
    return adapter.load_records(dataset_dir, active_config)


__all__ = [
    "CCPD_ADS",
    "CCPD_ALPHABETS",
    "CCPD_PROVINCE_CODES",
    "IMAGE_EXTENSIONS",
    "CcpdFilenameAdapter",
    "CocoJsonAdapter",
    "JsonlManifestAdapter",
    "OpenAlprSidecarAdapter",
    "PascalVocXmlAdapter",
    "RealDatasetAdapterProtocol",
    "RealDatasetFormat",
    "YoloDirectoryAdapter",
    "load_real_dataset",
    "parse_ccpd_filename",
]
