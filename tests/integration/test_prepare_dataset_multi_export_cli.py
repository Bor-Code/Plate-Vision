import json
from pathlib import Path

from plate_vision.data.cli import main as prepare_dataset_main


def test_prepare_dataset_cli_exports_coco_and_voc_manifests(tmp_path: Path) -> None:
    out_dir = tmp_path / "prepared_multi"
    exit_code = prepare_dataset_main(
        [
            "--output-dir",
            str(out_dir),
            "--country-code",
            "TR",
            "--synthetic-count",
            "6",
            "--export-coco",
            "--export-voc",
            "--seed",
            "17",
        ]
    )
    assert exit_code == 0

    summary = json.loads((out_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    assert "coco_manifests" in summary
    assert "voc_xml_counts" in summary
    assert Path(summary["coco_manifests"]["train"]).is_file()
    coco_payload = json.loads(Path(summary["coco_manifests"]["train"]).read_text(encoding="utf-8"))
    assert coco_payload["categories"][0]["name"] == "license_plate"
    assert len(coco_payload["images"]) == summary["train_count"]

    total_voc = sum(summary["voc_xml_counts"].values())
    assert total_voc == summary["unique_records"]
