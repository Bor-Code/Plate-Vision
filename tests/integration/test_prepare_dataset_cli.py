import json
from pathlib import Path

from plate_vision.data.cli import main
from plate_vision.data.synthetic import encode_grayscale_png


def test_prepare_dataset_cli_end_to_end(tmp_path: Path) -> None:
    out_dir = tmp_path / "prepared"
    exit_code = main(
        [
            "--output-dir",
            str(out_dir),
            "--country-code",
            "ES",
            "--synthetic-count",
            "12",
            "--seed",
            "7",
        ]
    )
    assert exit_code == 0

    summary_path = out_dir / "dataset_summary.json"
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["country_code"] == "ES"
    assert summary["real_records_loaded"] == 0
    assert summary["synthetic_records_generated"] == 12
    assert summary["total_generated"] == 12
    assert summary["unique_records"] == 12
    assert summary["train_count"] + summary["val_count"] + summary["test_count"] == 12
    assert Path(summary["yolo_dataset_yaml"]).is_file()
    assert Path(summary["recognizer_manifests"]["train"]).is_file()


def test_prepare_dataset_cli_real_only_and_hybrid(tmp_path: Path) -> None:
    real_dir = tmp_path / "real_yolo"
    (real_dir / "images").mkdir(parents=True)
    (real_dir / "labels").mkdir(parents=True)
    for idx in range(4):
        grid = [[110 + idx * 12 for _ in range(128)] for _ in range(64)]
        (real_dir / "images" / f"plate_{idx}.png").write_bytes(encode_grayscale_png(grid))
        (real_dir / "labels" / f"plate_{idx}.txt").write_text(
            f"0 0.5 0.5 0.5 0.4 34ABC10{idx} TR\n",
            encoding="utf-8",
        )

    real_only_out = tmp_path / "out_real"
    exit_real = main(
        [
            "--output-dir",
            str(real_only_out),
            "--real-dataset-dir",
            str(real_dir),
            "--real-format",
            "yolo_directory",
            "--synthetic-count",
            "0",
            "--country-code",
            "TR",
        ]
    )
    assert exit_real == 0
    real_summary = json.loads((real_only_out / "dataset_summary.json").read_text(encoding="utf-8"))
    assert real_summary["real_records_loaded"] == 4
    assert real_summary["synthetic_records_generated"] == 0
    assert real_summary["unique_records"] == 4

    hybrid_out = tmp_path / "out_hybrid"
    exit_hybrid = main(
        [
            "--output-dir",
            str(hybrid_out),
            "--real-dataset-dir",
            str(real_dir),
            "--real-format",
            "yolo_directory",
            "--synthetic-count",
            "6",
            "--country-code",
            "TR",
            "--seed",
            "42",
        ]
    )
    assert exit_hybrid == 0
    hybrid_summary = json.loads((hybrid_out / "dataset_summary.json").read_text(encoding="utf-8"))
    assert hybrid_summary["real_records_loaded"] == 4
    assert hybrid_summary["synthetic_records_generated"] == 6
    assert hybrid_summary["unique_records"] == 10
