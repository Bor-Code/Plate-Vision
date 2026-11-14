import json
from pathlib import Path

import yaml

from plate_vision.config.schema import DataConfig
from plate_vision.data import (
    DatasetSplit,
    deduplicate_records,
    export_recognizer_manifests,
    export_yolo_dataset,
    ingest_manifest_records,
    stratified_split,
)
from tests.unit.test_data_pipeline import make_png_bytes


def test_end_to_end_dataset_ingestion_dedup_split_and_export(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)

    raw_manifest = []
    for idx in range(12):
        img_path = raw_dir / f"plate_{idx:02d}.png"
        unique_payload = make_png_bytes(128, 64, extra=f"sample_{idx}".encode())
        img_path.write_bytes(unique_payload)
        province = "34" if idx < 6 else "06"
        raw_manifest.append(
            {
                "record_id": f"rec_{idx:02d}",
                "image_path": img_path,
                "source_dataset": "tr_plates_v1",
                "annotations": [
                    {
                        "plate_text": f"{province}ABC{100 + idx}",
                        "country_code": "TR",
                        "bbox": {
                            "x_min": 16.0,
                            "y_min": 16.0,
                            "x_max": 112.0,
                            "y_max": 48.0,
                        },
                    }
                ],
            }
        )

    dup_path = raw_dir / "plate_dup.png"
    dup_path.write_bytes((raw_dir / "plate_00.png").read_bytes())
    raw_manifest.append(
        {
            "record_id": "rec_99_dup",
            "image_path": dup_path,
            "source_dataset": "tr_plates_v1",
            "annotations": [],
        }
    )

    config = DataConfig(seed=42)
    ingested = ingest_manifest_records(raw_manifest, config)
    assert len(ingested) == 13

    dedup_result = deduplicate_records(ingested)
    assert dedup_result.removed_count == 1
    assert len(dedup_result.unique_records) == 12

    split_a = stratified_split(dedup_result.unique_records, config.split_ratios, seed=config.seed)
    split_b = stratified_split(dedup_result.unique_records, config.split_ratios, seed=config.seed)
    assert split_a == split_b
    assert len(split_a.train) + len(split_a.val) + len(split_a.test) == 12

    yolo_dir = tmp_path / "processed" / "yolo"
    dataset_yaml_path = export_yolo_dataset(split_a, yolo_dir)
    assert dataset_yaml_path.is_file()
    yolo_cfg = yaml.safe_load(dataset_yaml_path.read_text(encoding="utf-8"))
    assert yolo_cfg["names"] == {0: "license_plate"}

    first_train_rec = split_a.train[0]
    first_label_file = yolo_dir / "labels" / "train" / f"{first_train_rec.record_id}.txt"
    assert first_label_file.is_file()
    assert first_label_file.read_text(encoding="utf-8").startswith("0 0.500000 0.500000")

    rec_dir = tmp_path / "processed" / "recognizer"
    manifest_paths = export_recognizer_manifests(split_a, rec_dir)
    assert set(manifest_paths.keys()) == {
        DatasetSplit.TRAIN,
        DatasetSplit.VAL,
        DatasetSplit.TEST,
    }
    train_jsonl_lines = manifest_paths[DatasetSplit.TRAIN].read_text(encoding="utf-8").splitlines()
    assert len(train_jsonl_lines) == len(split_a.train)
    parsed_first = json.loads(train_jsonl_lines[0])
    assert parsed_first["country_code"] == "TR"
    assert (rec_dir / "recognizer_train.csv").is_file()
