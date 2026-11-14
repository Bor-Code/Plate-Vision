import csv
import io
import json
from pathlib import Path

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


__all__ = [
    "export_recognizer_manifests",
    "export_yolo_dataset",
]
