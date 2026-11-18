import json
from pathlib import Path

from plate_vision.data.cli import main


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
    assert summary["total_generated"] == 12
    assert summary["unique_records"] == 12
    assert summary["train_count"] + summary["val_count"] + summary["test_count"] == 12
    assert Path(summary["yolo_dataset_yaml"]).is_file()
    assert Path(summary["recognizer_manifests"]["train"]).is_file()
