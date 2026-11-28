import csv
import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml

from plate_vision.data.ingestion import LocalFileStorage
from plate_vision.data.models import (
    DatasetSplit,
    DatasetStorageProtocol,
    SplitDataset,
)


def export_yolo_dataset(
    split_dataset: SplitDataset,
    output_dir: Path,
    storage: DatasetStorageProtocol | None = None,
) -> Path:
    active_storage: DatasetStorageProtocol = storage if storage is not None else LocalFileStorage()

    for split in (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST):
        records = split_dataset.for_split(split)
        split_list_lines: list[str] = []
        for record in records:
            label_lines = [
                ann.bbox.format_yolo_line(
                    image_width=record.image.width,
                    image_height=record.image.height,
                )
                for ann in record.annotations
            ]
            label_content = "\n".join(label_lines) + ("\n" if label_lines else "")
            label_path = output_dir / "labels" / split.value / f"{record.record_id}.txt"
            active_storage.write_text(label_path, label_content)
            split_list_lines.append(record.image.path.as_posix())

        split_index_path = output_dir / f"{split.value}.txt"
        split_index_content = "\n".join(split_list_lines) + ("\n" if split_list_lines else "")
        active_storage.write_text(split_index_path, split_index_content)

    dataset_yaml_payload = {
        "path": output_dir.as_posix(),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": {0: "license_plate"},
    }
    dataset_yaml_path = output_dir / "dataset.yaml"
    active_storage.write_text(
        dataset_yaml_path,
        yaml.safe_dump(dataset_yaml_payload, sort_keys=False),
    )
    return dataset_yaml_path


def export_recognizer_manifests(
    split_dataset: SplitDataset,
    output_dir: Path,
    storage: DatasetStorageProtocol | None = None,
) -> dict[DatasetSplit, Path]:
    active_storage: DatasetStorageProtocol = storage if storage is not None else LocalFileStorage()
    manifest_paths: dict[DatasetSplit, Path] = {}

    for split in (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST):
        records = split_dataset.for_split(split)
        jsonl_lines: list[str] = []
        csv_buffer = io.StringIO()
        csv_writer = csv.writer(csv_buffer, lineterminator="\n")
        csv_writer.writerow(
            [
                "sample_id",
                "record_id",
                "image_path",
                "plate_text",
                "country_code",
                "x_min",
                "y_min",
                "x_max",
                "y_max",
                "sha256",
            ]
        )

        for record in records:
            for ann_idx, annotation in enumerate(record.annotations):
                sample_id = f"{record.record_id}_{ann_idx:02d}"
                normalized_plate = annotation.plate_text.strip().upper()
                entry = {
                    "sample_id": sample_id,
                    "record_id": record.record_id,
                    "image_path": record.image.path.as_posix(),
                    "plate_text": normalized_plate,
                    "country_code": annotation.country_code,
                    "bbox": {
                        "x_min": annotation.bbox.x_min,
                        "y_min": annotation.bbox.y_min,
                        "x_max": annotation.bbox.x_max,
                        "y_max": annotation.bbox.y_max,
                    },
                    "sha256": record.image.sha256,
                }
                jsonl_lines.append(json.dumps(entry, sort_keys=True))
                csv_writer.writerow(
                    [
                        sample_id,
                        record.record_id,
                        record.image.path.as_posix(),
                        normalized_plate,
                        annotation.country_code,
                        annotation.bbox.x_min,
                        annotation.bbox.y_min,
                        annotation.bbox.x_max,
                        annotation.bbox.y_max,
                        record.image.sha256,
                    ]
                )

        jsonl_path = output_dir / f"recognizer_{split.value}.jsonl"
        csv_path = output_dir / f"recognizer_{split.value}.csv"
        jsonl_content = "\n".join(jsonl_lines) + ("\n" if jsonl_lines else "")
        active_storage.write_text(jsonl_path, jsonl_content)
        active_storage.write_text(csv_path, csv_buffer.getvalue())
        manifest_paths[split] = jsonl_path

    return manifest_paths


def export_coco_dataset(
    split_dataset: SplitDataset,
    output_dir: Path,
    storage: DatasetStorageProtocol | None = None,
) -> dict[DatasetSplit, Path]:
    active_storage: DatasetStorageProtocol = storage if storage is not None else LocalFileStorage()
    coco_paths: dict[DatasetSplit, Path] = {}

    for split in (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST):
        records = split_dataset.for_split(split)
        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        ann_seq = 1

        for img_id, record in enumerate(records, start=1):
            images.append(
                {
                    "id": img_id,
                    "file_name": record.image.path.as_posix(),
                    "width": record.image.width,
                    "height": record.image.height,
                    "record_id": record.record_id,
                }
            )
            for ann in record.annotations:
                box_w = max(0.0, ann.bbox.x_max - ann.bbox.x_min)
                box_h = max(0.0, ann.bbox.y_max - ann.bbox.y_min)
                entry: dict[str, Any] = {
                    "id": ann_seq,
                    "image_id": img_id,
                    "category_id": ann.bbox.class_id,
                    "bbox": [
                        round(ann.bbox.x_min, 4),
                        round(ann.bbox.y_min, 4),
                        round(box_w, 4),
                        round(box_h, 4),
                    ],
                    "area": round(box_w * box_h, 4),
                    "iscrowd": 0,
                    "plate_text": ann.plate_text,
                    "country_code": ann.country_code,
                }
                if ann.corners is not None and len(ann.corners) == 4:
                    entry["segmentation"] = [
                        [round(coord, 4) for pt in ann.corners for coord in pt]
                    ]
                annotations.append(entry)
                ann_seq += 1

        payload: dict[str, Any] = {
            "categories": [{"id": 0, "name": "license_plate", "supercategory": "vehicle"}],
            "images": images,
            "annotations": annotations,
        }
        coco_path = output_dir / f"coco_{split.value}.json"
        active_storage.write_text(coco_path, json.dumps(payload, indent=2))
        coco_paths[split] = coco_path

    return coco_paths


def export_pascal_voc_dataset(
    split_dataset: SplitDataset,
    output_dir: Path,
    storage: DatasetStorageProtocol | None = None,
) -> dict[DatasetSplit, list[Path]]:
    active_storage: DatasetStorageProtocol = storage if storage is not None else LocalFileStorage()
    voc_paths: dict[DatasetSplit, list[Path]] = {}

    for split in (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST):
        records = split_dataset.for_split(split)
        split_xml_paths: list[Path] = []
        for record in records:
            root = ET.Element("annotation")
            ET.SubElement(root, "folder").text = split.value
            ET.SubElement(root, "filename").text = record.image.path.name
            ET.SubElement(root, "path").text = record.image.path.as_posix()

            size_el = ET.SubElement(root, "size")
            ET.SubElement(size_el, "width").text = str(record.image.width)
            ET.SubElement(size_el, "height").text = str(record.image.height)
            ET.SubElement(size_el, "depth").text = "1"

            for ann in record.annotations:
                obj_el = ET.SubElement(root, "object")
                ET.SubElement(obj_el, "name").text = "license_plate"
                ET.SubElement(obj_el, "plate_text").text = ann.plate_text
                ET.SubElement(obj_el, "country_code").text = ann.country_code
                bndbox_el = ET.SubElement(obj_el, "bndbox")
                ET.SubElement(bndbox_el, "xmin").text = str(round(ann.bbox.x_min, 4))
                ET.SubElement(bndbox_el, "ymin").text = str(round(ann.bbox.y_min, 4))
                ET.SubElement(bndbox_el, "xmax").text = str(round(ann.bbox.x_max, 4))
                ET.SubElement(bndbox_el, "ymax").text = str(round(ann.bbox.y_max, 4))

            xml_path = output_dir / split.value / f"{record.record_id}.xml"
            active_storage.write_text(xml_path, ET.tostring(root, encoding="unicode"))
            split_xml_paths.append(xml_path)
        voc_paths[split] = split_xml_paths

    return voc_paths


__all__ = [
    "export_coco_dataset",
    "export_pascal_voc_dataset",
    "export_recognizer_manifests",
    "export_yolo_dataset",
]
