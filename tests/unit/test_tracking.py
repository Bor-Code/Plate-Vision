from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from plate_vision.utils import (
    DataIOError,
    MlflowExperimentTracker,
    compute_dataset_version_hash,
    resolve_git_commit_hash,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_git_commit_and_dataset_version_hashes(tmp_path: Path) -> None:
    assert resolve_git_commit_hash(tmp_path) == "unknown"

    git_dir = tmp_path / ".git"
    refs_dir = git_dir / "refs" / "heads"
    refs_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (refs_dir / "main").write_text("deadbeef1234567890\n", encoding="utf-8")
    assert resolve_git_commit_hash(tmp_path) == "deadbeef1234567890"

    (git_dir / "HEAD").write_text("1234567890abcdef\n", encoding="utf-8")
    assert resolve_git_commit_hash(tmp_path) == "1234567890abcdef"

    with pytest.raises(DataIOError, match="Dataset path does not exist"):
        compute_dataset_version_hash(tmp_path / "missing")

    single_file = tmp_path / "data.txt"
    single_file.write_text("sample_content", encoding="utf-8")
    file_hash = compute_dataset_version_hash(single_file)
    assert len(file_hash) == 64

    sub_dir = tmp_path / "dataset_dir"
    sub_dir.mkdir()
    (sub_dir / "a.txt").write_text("1", encoding="utf-8")
    (sub_dir / "b.txt").write_text("2", encoding="utf-8")
    dir_hash = compute_dataset_version_hash(sub_dir)
    assert len(dir_hash) == 64


def test_mlflow_experiment_tracker_lifecycle(tmp_path: Path) -> None:
    tracker = MlflowExperimentTracker(
        tracking_dir=tmp_path / "mlruns",
        experiment_name="unit_exp",
        seed=42,
        repo_root=tmp_path,
    )
    with pytest.raises(DataIOError, match="without an active run"):
        tracker.log_params({"lr": 0.01})

    run_id = tracker.start_run("run_01", tags={"dataset_version": "v1_hash"})
    assert tracker.active_run_id == run_id
    run_dir = tracker.active_run_dir
    assert run_dir is not None
    assert run_dir.is_dir()

    tracker.log_params({"lr": 0.001, "layers": [2, 4]})
    tracker.log_metrics({"loss": 0.25, "acc": 0.9}, step=1)
    tracker.log_metrics({"loss": 0.15, "acc": 0.95}, step=2)

    artifact = tmp_path / "summary.json"
    artifact.write_text('{"ok": true}\n', encoding="utf-8")
    tracker.log_artifact(artifact)

    with pytest.raises(DataIOError, match="Artifact file does not exist"):
        tracker.log_artifact(tmp_path / "missing_artifact.json")

    tracker.end_run()
    with pytest.raises(DataIOError, match="without an active run"):
        tracker.log_params({"after": 1})

    assert (run_dir / "params" / "lr").read_text(encoding="utf-8") == "0.001"
    assert (run_dir / "tags" / "dataset_version").read_text(encoding="utf-8") == "v1_hash"
    assert (run_dir / "artifacts" / "summary.json").is_file()
    loss_lines = (run_dir / "metrics" / "loss").read_text(encoding="utf-8").splitlines()
    assert len(loss_lines) == 2
    meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))
    assert meta["status"] == "FINISHED"
