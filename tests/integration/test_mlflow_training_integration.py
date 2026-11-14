from __future__ import annotations

from typing import TYPE_CHECKING

from plate_vision.config.schema import DataConfig
from plate_vision.data import (
    SyntheticPlateConfig,
    TurkeyCountryProfile,
    export_recognizer_manifests,
    export_yolo_dataset,
    generate_synthetic_dataset,
    stratified_split,
)
from plate_vision.detection import run_train_detector
from plate_vision.recognition import run_train_recognizer

if TYPE_CHECKING:
    from pathlib import Path


def test_detector_and_recognizer_cli_mlflow_tracking(tmp_path: Path) -> None:
    records = generate_synthetic_dataset(
        count=8,
        output_dir=tmp_path / "synth",
        config=SyntheticPlateConfig(seed=42),
        profile=TurkeyCountryProfile(),
    )
    splits = stratified_split(records, DataConfig().split_ratios, seed=42)
    dataset_yaml = export_yolo_dataset(splits, tmp_path / "yolo")
    manifest_dir = tmp_path / "manifests"
    export_recognizer_manifests(splits, manifest_dir)

    tracking_dir = tmp_path / "mlruns"

    det_cfg = tmp_path / "det.yaml"
    det_cfg.write_text("epochs: 2\nwarmup_epochs: 1\n", encoding="utf-8")
    run_train_detector(
        [
            "--config",
            str(det_cfg),
            "--dataset-yaml",
            str(dataset_yaml),
            "--output-dir",
            str(tmp_path / "ckpt_det"),
            "--tracking-dir",
            str(tracking_dir),
            "--seed",
            "42",
        ]
    )

    rec_cfg = tmp_path / "rec.yaml"
    rec_cfg.write_text("epochs: 2\nbatch_size: 4\nwarmup_epochs: 1\n", encoding="utf-8")
    run_train_recognizer(
        [
            "--config",
            str(rec_cfg),
            "--manifest-dir",
            str(manifest_dir),
            "--output-dir",
            str(tmp_path / "ckpt_rec"),
            "--tracking-dir",
            str(tracking_dir),
            "--seed",
            "42",
        ]
    )

    det_runs = list((tracking_dir / "detector").iterdir())
    rec_runs = list((tracking_dir / "recognizer").iterdir())
    assert len(det_runs) == 1
    assert len(rec_runs) == 1
    assert (det_runs[0] / "artifacts" / "best.ckpt.json").is_file()
    assert (rec_runs[0] / "artifacts" / "best.ckpt.json").is_file()
    assert (det_runs[0] / "tags" / "dataset_version").is_file()
    assert (rec_runs[0] / "tags" / "dataset_version").is_file()
