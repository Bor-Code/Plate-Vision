import json
from pathlib import Path

import pytest

from plate_vision.config.schema import DataConfig
from plate_vision.data.adapters import (
    JsonlManifestAdapter,
    OpenAlprSidecarAdapter,
    RealDatasetAdapterProtocol,
    RealDatasetFormat,
    YoloDirectoryAdapter,
    load_real_dataset,
)
from plate_vision.data.synthetic import encode_grayscale_png
from plate_vision.utils.exceptions import DataIOError, DatasetValidationError


def _write_png(path: Path, width: int = 128, height: int = 64) -> Path:
    raw = encode_grayscale_png([[140 for _ in range(width)] for _ in range(height)])
    path.write_bytes(raw)
    return path


def test_jsonl_manifest_adapter_and_error_cases(tmp_path: Path) -> None:
    cfg = DataConfig()
    adapter = JsonlManifestAdapter()
    assert isinstance(adapter, RealDatasetAdapterProtocol)

    with pytest.raises(DataIOError):
        adapter.load_records(tmp_path / "missing", cfg)

    ds_dir = tmp_path / "jsonl_ds"
    ds_dir.mkdir(parents=True)
    _write_png(ds_dir / "img_01.png")
    record_line = json.dumps(
        {
            "record_id": "real_01",
            "image_path": "img_01.png",
            "annotations": [
                {
                    "plate_text": "34ABC123",
                    "country_code": "TR",
                    "bbox": {"x_min": 10.0, "y_min": 12.0, "x_max": 90.0, "y_max": 36.0},
                }
            ],
        }
    )
    (ds_dir / "manifest.jsonl").write_text(record_line + "\n", encoding="utf-8")

    records = load_real_dataset(ds_dir, RealDatasetFormat.JSONL_MANIFEST, config=cfg)
    assert len(records) == 1
    assert records[0].source_dataset == "real_jsonl"
    assert records[0].annotations[0].plate_text == "34ABC123"

    (ds_dir / "bad.jsonl").write_text("{invalid json\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError):
        JsonlManifestAdapter(manifest_filename="bad.jsonl").load_records(ds_dir, cfg)


def test_yolo_directory_adapter_and_validation(tmp_path: Path) -> None:
    cfg = DataConfig()
    ds_dir = tmp_path / "yolo_ds"
    (ds_dir / "images").mkdir(parents=True)
    (ds_dir / "labels").mkdir(parents=True)
    _write_png(ds_dir / "images" / "car_01.png", width=100, height=50)
    (ds_dir / "labels" / "car_01.txt").write_text(
        "0 0.5 0.5 0.6 0.4 06XYZ789 TR\n", encoding="utf-8"
    )

    records = load_real_dataset(
        ds_dir,
        RealDatasetFormat.YOLO_DIRECTORY,
        config=cfg,
        default_country_code="DE",
    )
    assert len(records) == 1
    ann = records[0].annotations[0]
    assert ann.plate_text == "06XYZ789"
    assert ann.country_code == "TR"
    assert ann.bbox.x_min == pytest.approx(20.0)
    assert ann.bbox.x_max == pytest.approx(80.0)

    (ds_dir / "labels" / "car_01.txt").write_text("0 0.5\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError):
        YoloDirectoryAdapter().load_records(ds_dir, cfg)


def test_openalpr_sidecar_adapter_and_validation(tmp_path: Path) -> None:
    cfg = DataConfig()
    ds_dir = tmp_path / "openalpr_ds"
    ds_dir.mkdir(parents=True)
    _write_png(ds_dir / "gate_01.png", width=128, height=64)
    sidecar = {
        "plates": [
            {
                "plate_text": "BAB1234",
                "bbox": {"x_min": 16.0, "y_min": 14.0, "x_max": 100.0, "y_max": 42.0},
                "corners": [[16.0, 14.0], [100.0, 14.0], [100.0, 42.0], [16.0, 42.0]],
            }
        ]
    }
    (ds_dir / "gate_01.json").write_text(json.dumps(sidecar), encoding="utf-8")

    records = load_real_dataset(
        ds_dir,
        RealDatasetFormat.OPENALPR_SIDECAR,
        config=cfg,
        default_country_code="DE",
    )
    assert len(records) == 1
    assert records[0].annotations[0].country_code == "DE"
    assert records[0].annotations[0].corners is not None

    (ds_dir / "gate_01.json").write_text(
        json.dumps({"plates": [{"plate_text": "X"}]}), encoding="utf-8"
    )
    with pytest.raises(DatasetValidationError):
        OpenAlprSidecarAdapter().load_records(ds_dir, cfg)
