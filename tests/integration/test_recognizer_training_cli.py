from __future__ import annotations

from typing import TYPE_CHECKING

from plate_vision.config.schema import DataConfig
from plate_vision.data import (
    SyntheticPlateConfig,
    TurkeyCountryProfile,
    export_recognizer_manifests,
    generate_synthetic_dataset,
    stratified_split,
)
from plate_vision.recognition import run_train_recognizer

if TYPE_CHECKING:
    from pathlib import Path


def test_train_recognizer_cli_and_resume_flow(tmp_path: Path) -> None:
    records = generate_synthetic_dataset(
        count=10,
        output_dir=tmp_path / "synth",
        config=SyntheticPlateConfig(seed=42),
        profile=TurkeyCountryProfile(),
    )
    splits = stratified_split(records, DataConfig().split_ratios, seed=42)
    manifest_dir = tmp_path / "manifests"
    export_recognizer_manifests(splits, manifest_dir)

    rec_config_path = tmp_path / "recognizer.yaml"
    rec_config_path.write_text(
        "epochs: 3\n"
        "batch_size: 4\n"
        "warmup_epochs: 1\n"
        "early_stopping_patience: 2\n"
        "mixed_precision: true\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "checkpoints"
    summary = run_train_recognizer(
        [
            "--config",
            str(rec_config_path),
            "--manifest-dir",
            str(manifest_dir),
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

    rec_config_resume = tmp_path / "recognizer_resume.yaml"
    rec_config_resume.write_text(
        "epochs: 5\n"
        "batch_size: 4\n"
        "warmup_epochs: 1\n"
        "early_stopping_patience: 1\n"
        "mixed_precision: true\n",
        encoding="utf-8",
    )
    resumed = run_train_recognizer(
        [
            "--config",
            str(rec_config_resume),
            "--manifest-dir",
            str(manifest_dir),
            "--output-dir",
            str(out_dir),
            "--resume-from",
            str(summary.last_checkpoint_path),
            "--seed",
            "42",
        ]
    )
    assert resumed.stopped_early
    assert resumed.epochs_completed == 2
