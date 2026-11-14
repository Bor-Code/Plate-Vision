from __future__ import annotations

from typing import TYPE_CHECKING

from plate_vision.config.schema import DataConfig, DetectorConfig
from plate_vision.data import (
    SyntheticPlateConfig,
    TurkeyCountryProfile,
    export_yolo_dataset,
    generate_synthetic_dataset,
    stratified_split,
)
from plate_vision.detection import YoloPlateDetector, run_train_detector

if TYPE_CHECKING:
    from pathlib import Path


def test_train_detector_cli_and_resume_flow(tmp_path: Path) -> None:
    synth_dir = tmp_path / "synth"
    records = generate_synthetic_dataset(
        count=10,
        output_dir=synth_dir,
        config=SyntheticPlateConfig(seed=42),
        profile=TurkeyCountryProfile(),
    )
    splits = stratified_split(records, DataConfig().split_ratios, seed=42)
    yolo_dir = tmp_path / "yolo"
    dataset_yaml = export_yolo_dataset(splits, yolo_dir)

    det_config_path = tmp_path / "detector.yaml"
    det_config_path.write_text(
        "model_architecture: yolo11n.pt\n"
        "input_size: 640\n"
        "epochs: 3\n"
        "warmup_epochs: 1\n"
        "early_stopping_patience: 5\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "checkpoints"
    summary = run_train_detector(
        [
            "--config",
            str(det_config_path),
            "--dataset-yaml",
            str(dataset_yaml),
            "--output-dir",
            str(out_dir),
            "--seed",
            "42",
        ]
    )
    assert summary.epochs_completed == 3
    assert summary.best_checkpoint_path.is_file()
    assert summary.last_checkpoint_path.is_file()
    assert (out_dir / "training_summary.json").is_file()
    assert summary.test_metrics.map50 > 0.5

    det_config_resume = tmp_path / "detector_resume.yaml"
    det_config_resume.write_text(
        "model_architecture: yolo11n.pt\n"
        "input_size: 640\n"
        "epochs: 5\n"
        "warmup_epochs: 1\n"
        "early_stopping_patience: 5\n",
        encoding="utf-8",
    )
    resumed_summary = run_train_detector(
        [
            "--config",
            str(det_config_resume),
            "--dataset-yaml",
            str(dataset_yaml),
            "--output-dir",
            str(out_dir),
            "--resume-from",
            str(summary.last_checkpoint_path),
            "--seed",
            "42",
        ]
    )
    assert resumed_summary.epochs_completed == 3

    detector = YoloPlateDetector(
        config=DetectorConfig(),
        checkpoint_path=resumed_summary.best_checkpoint_path,
    )
    sample_bytes = records[0].image.path.read_bytes()
    detections = detector.detect(sample_bytes, image_width=128, image_height=32)
    assert len(detections) == 1
