import hashlib
import json
import shutil
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import yaml

from plate_vision.utils.exceptions import DataIOError
from plate_vision.utils.logging import collect_environment_info


class ExperimentTrackerProtocol(Protocol):
    def start_run(
        self,
        run_name: str,
        *,
        tags: Mapping[str, str] | None = None,
    ) -> str: ...

    def log_params(self, params: Mapping[str, Any]) -> None: ...

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        step: int | None = None,
    ) -> None: ...

    def log_artifact(self, artifact_path: Path) -> None: ...

    def end_run(self, status: str = "FINISHED") -> None: ...


def resolve_git_commit_hash(repo_root: Path | None = None) -> str:
    root = repo_root if repo_root is not None else Path.cwd()
    git_dir = root / ".git"
    head_file = git_dir / "HEAD"
    if not head_file.is_file():
        return "unknown"
    try:
        head_content = head_file.read_text(encoding="utf-8").strip()
        if head_content.startswith("ref: "):
            ref_rel = head_content[5:].strip()
            ref_file = git_dir / ref_rel
            if ref_file.is_file():
                return ref_file.read_text(encoding="utf-8").strip()
            return "unknown"
        return head_content
    except OSError:
        return "unknown"


def compute_dataset_version_hash(dataset_path: Path) -> str:
    if not dataset_path.exists():
        raise DataIOError(
            f"Dataset path does not exist for versioning: {dataset_path}",
            details={"path": str(dataset_path)},
        )
    hasher = hashlib.sha256()
    if dataset_path.is_file():
        hasher.update(dataset_path.read_bytes())
        return hasher.hexdigest()

    for file_path in sorted(p for p in dataset_path.rglob("*") if p.is_file()):
        rel_name = file_path.relative_to(dataset_path).as_posix()
        hasher.update(rel_name.encode("utf-8"))
        hasher.update(file_path.read_bytes())
    return hasher.hexdigest()


class MlflowExperimentTracker:
    def __init__(
        self,
        tracking_dir: Path = Path("mlruns"),
        experiment_name: str = "plate-vision",
        seed: int = 42,
        repo_root: Path | None = None,
    ) -> None:
        self.tracking_dir = tracking_dir
        self.experiment_name = experiment_name
        self.seed = seed
        self.repo_root = repo_root
        self._active_run_id: str | None = None
        self._active_run_dir: Path | None = None

    @property
    def active_run_id(self) -> str | None:
        return self._active_run_id

    @property
    def active_run_dir(self) -> Path | None:
        return self._active_run_dir

    def start_run(
        self,
        run_name: str,
        *,
        tags: Mapping[str, str] | None = None,
    ) -> str:
        exp_dir = self.tracking_dir / self.experiment_name
        run_id = uuid.uuid4().hex
        run_dir = exp_dir / run_id
        for subdir in ("params", "metrics", "tags", "artifacts"):
            (run_dir / subdir).mkdir(parents=True, exist_ok=True)

        self._active_run_id = run_id
        self._active_run_dir = run_dir

        env_info = collect_environment_info(seed=self.seed)
        git_sha = resolve_git_commit_hash(self.repo_root)
        merged_tags: dict[str, str] = {
            "mlflow.runName": run_name,
            "git_commit": git_sha,
            "python_version": str(env_info["python_version"]),
            "platform": str(env_info["platform"]),
            "package_version": str(env_info["package_version"]),
            "seed": str(self.seed),
        }
        if tags is not None:
            for k, v in tags.items():
                merged_tags[str(k)] = str(v)

        for tag_key, tag_val in merged_tags.items():
            safe_key = tag_key.replace("/", "_")
            (run_dir / "tags" / safe_key).write_text(tag_val, encoding="utf-8")

        meta = {
            "run_id": run_id,
            "run_name": run_name,
            "experiment_name": self.experiment_name,
            "status": "RUNNING",
            "start_time": int(time.time() * 1000),
            "git_commit": git_sha,
        }
        (run_dir / "meta.yaml").write_text(
            yaml.safe_dump(meta, sort_keys=False),
            encoding="utf-8",
        )
        return run_id

    def _require_run_dir(self) -> Path:
        if self._active_run_dir is None:
            raise DataIOError("Cannot log to MlflowExperimentTracker without an active run")
        return self._active_run_dir

    def log_params(self, params: Mapping[str, Any]) -> None:
        run_dir = self._require_run_dir()
        for key, value in params.items():
            safe_key = str(key).replace("/", "_")
            serialized = (
                json.dumps(value, sort_keys=True)
                if isinstance(value, dict | list | tuple)
                else str(value)
            )
            (run_dir / "params" / safe_key).write_text(serialized, encoding="utf-8")

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        step: int | None = None,
    ) -> None:
        run_dir = self._require_run_dir()
        timestamp_ms = int(time.time() * 1000)
        step_val = step if step is not None else 0
        for key, value in metrics.items():
            safe_key = str(key).replace("/", "_")
            metric_file = run_dir / "metrics" / safe_key
            line = f"{timestamp_ms} {float(value):.6f} {step_val}\n"
            with metric_file.open("a", encoding="utf-8") as handle:
                handle.write(line)

    def log_artifact(self, artifact_path: Path) -> None:
        run_dir = self._require_run_dir()
        if not artifact_path.is_file():
            raise DataIOError(
                f"Artifact file does not exist: {artifact_path}",
                details={"path": str(artifact_path)},
            )
        dest = run_dir / "artifacts" / artifact_path.name
        shutil.copyfile(artifact_path, dest)

    def end_run(self, status: str = "FINISHED") -> None:
        run_dir = self._require_run_dir()
        meta_path = run_dir / "meta.yaml"
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        if isinstance(meta, dict):
            meta["status"] = status
            meta["end_time"] = int(time.time() * 1000)
            meta_path.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
        self._active_run_id = None
        self._active_run_dir = None


__all__ = [
    "ExperimentTrackerProtocol",
    "MlflowExperimentTracker",
    "compute_dataset_version_hash",
    "resolve_git_commit_hash",
]
