import json
from pathlib import Path

from plate_vision.data import encode_grayscale_png
from plate_vision.data.cli import main as prepare_dataset_main


def _write_png(path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = [[128 for _ in range(width)] for _ in range(height)]
    path.write_bytes(encode_grayscale_png(pixels))


def test_prepare_dataset_cli_supports_coco_json(tmp_path: Path) -> None:
    coco_dir = tmp_path / "coco_raw"
    out_dir = tmp_path / "coco_prepared"
    _write_png(coco_dir / "images" / "van_01.png", 640, 480)

    payload = {
        "images": [{"id": 1, "file_name": "van_01.png", "width": 640, "height": 480}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 0,
                "bbox": [110.0, 180.0, 150.0, 40.0],
                "plate_text": "34COCO99",
                "country_code": "TR",
            }
        ],
    }
    (coco_dir / "_annotations.coco.json").write_text(json.dumps(payload), encoding="utf-8")

    exit_code = prepare_dataset_main(
        [
            "--output-dir",
            str(out_dir),
            "--real-dataset-dir",
            str(coco_dir),
            "--real-format",
            "coco_json",
            "--synthetic-count",
            "0",
            "--country-code",
            "TR",
        ]
    )
    assert exit_code == 0
    summary = json.loads((out_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    assert summary["unique_records"] == 1
    assert summary["real_records_loaded"] == 1
    assert (out_dir / "yolo" / "dataset.yaml").is_file()
    train_manifest = (out_dir / "recognizer" / "recognizer_train.jsonl").read_text(encoding="utf-8")
    assert "34COCO99" in train_manifest
