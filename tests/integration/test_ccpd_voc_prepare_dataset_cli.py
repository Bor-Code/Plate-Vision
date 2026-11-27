import json
from pathlib import Path

from plate_vision.data import encode_grayscale_png
from plate_vision.data.cli import main as prepare_dataset_main


def _write_png(path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = [[128 for _ in range(width)] for _ in range(height)]
    path.write_bytes(encode_grayscale_png(pixels))


def test_prepare_dataset_cli_supports_pascal_voc_xml(tmp_path: Path) -> None:
    voc_dir = tmp_path / "voc_raw"
    out_dir = tmp_path / "voc_prepared"
    _write_png(voc_dir / "images" / "truck_01.png", 640, 480)
    (voc_dir / "annotations").mkdir(parents=True, exist_ok=True)
    (voc_dir / "annotations" / "truck_01.xml").write_text(
        "<annotation><object>"
        "<name>34XYZ99</name>"
        "<bndbox><xmin>120</xmin><ymin>200</ymin><xmax>280</xmax><ymax>250</ymax></bndbox>"
        "</object></annotation>",
        encoding="utf-8",
    )

    exit_code = prepare_dataset_main(
        [
            "--output-dir",
            str(out_dir),
            "--real-dataset-dir",
            str(voc_dir),
            "--real-format",
            "pascal_voc_xml",
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
    assert "34XYZ99" in train_manifest


def test_prepare_dataset_cli_supports_ccpd_filename(tmp_path: Path) -> None:
    ccpd_dir = tmp_path / "ccpd_raw"
    out_dir = tmp_path / "ccpd_prepared"
    _write_png(
        ccpd_dir
        / "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.png",
        800,
        600,
    )

    exit_code = prepare_dataset_main(
        [
            "--output-dir",
            str(out_dir),
            "--real-dataset-dir",
            str(ccpd_dir),
            "--real-format",
            "ccpd_filename",
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
    assert "AHAY339S" in train_manifest
