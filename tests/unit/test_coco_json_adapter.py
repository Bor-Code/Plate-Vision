import json
from pathlib import Path

import pytest

from plate_vision.config.schema import DataConfig
from plate_vision.data import (
    CocoJsonAdapter,
    RealDatasetFormat,
    encode_grayscale_png,
    load_real_dataset,
)
from plate_vision.utils.exceptions import DataIOError, DatasetValidationError


def _write_png(path: Path, width: int = 640, height: int = 480) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = [[128 for _ in range(width)] for _ in range(height)]
    path.write_bytes(encode_grayscale_png(pixels))


def test_coco_json_adapter_loads_bbox_attributes_and_segmentation(tmp_path: Path) -> None:
    ds_dir = tmp_path / "coco"
    _write_png(ds_dir / "car_01.png", 640, 480)
    _write_png(ds_dir / "images" / "car_02.png", 640, 480)

    coco_payload = {
        "images": [
            {"id": 1, "file_name": "car_01.png", "width": 640, "height": 480},
            {"id": 2, "file_name": "car_02.png", "width": 640, "height": 480},
        ],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "category_id": 0,
                "bbox": [100.0, 200.0, 160.0, 45.0],
                "segmentation": [[100.0, 200.0, 260.0, 200.0, 260.0, 245.0, 100.0, 245.0]],
                "attributes": {"plate_text": "34ABC123", "country_code": "tr"},
            },
            {
                "id": 11,
                "image_id": 2,
                "category_id": 1,
                "bbox": [50.0, 80.0, 120.0, 35.0],
            },
        ],
    }
    (ds_dir / "_annotations.coco.json").write_text(json.dumps(coco_payload), encoding="utf-8")

    cfg = DataConfig(min_image_width=16, min_image_height=16)
    records = load_real_dataset(
        ds_dir,
        RealDatasetFormat.COCO_JSON,
        config=cfg,
        default_country_code="DE",
    )
    assert len(records) == 2
    assert records[0].source_dataset == "real_coco"
    ann0 = records[0].annotations[0]
    assert ann0.plate_text == "34ABC123"
    assert ann0.country_code == "TR"
    assert ann0.bbox.x_max == 260.0
    assert ann0.bbox.y_max == 245.0
    assert ann0.corners == (
        (100.0, 200.0),
        (260.0, 200.0),
        (260.0, 245.0),
        (100.0, 245.0),
    )

    ann1 = records[1].annotations[0]
    assert ann1.plate_text == "UNKNOWN"
    assert ann1.country_code == "DE"
    assert ann1.bbox.class_id == 1


def test_coco_json_adapter_errors(tmp_path: Path) -> None:
    cfg = DataConfig(min_image_width=16, min_image_height=16)

    with pytest.raises(DataIOError):
        CocoJsonAdapter().load_records(tmp_path / "missing_dir", cfg)

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(DataIOError):
        CocoJsonAdapter().load_records(empty_dir, cfg)

    (empty_dir / "annotations.json").write_text("{bad json", encoding="utf-8")
    with pytest.raises(DatasetValidationError):
        CocoJsonAdapter(annotations_filename="annotations.json").load_records(empty_dir, cfg)

    (empty_dir / "annotations.json").write_text(json.dumps({"images": []}), encoding="utf-8")
    with pytest.raises(DatasetValidationError):
        CocoJsonAdapter(annotations_filename="annotations.json").load_records(empty_dir, cfg)

    _write_png(empty_dir / "car_01.png", 640, 480)
    bad_bbox = {
        "images": [{"id": 1, "file_name": "car_01.png"}],
        "annotations": [{"id": 1, "image_id": 1, "bbox": [10.0, 20.0, -5.0, 30.0]}],
    }
    (empty_dir / "annotations.json").write_text(json.dumps(bad_bbox), encoding="utf-8")
    with pytest.raises(DatasetValidationError):
        CocoJsonAdapter(annotations_filename="annotations.json").load_records(empty_dir, cfg)
