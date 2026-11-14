# Experiment Tracking with MLflow

## Overview

Both training pipelines (`DetectorTrainer` and `RecognizerTrainer`) integrate with `ExperimentTrackerProtocol` via `MlflowExperimentTracker` (`plate_vision.utils.tracking`) to persist complete experiment lineage under the MLflow FileStore directory structure (`mlruns/<experiment>/<run_id>/`).

## Tracked Metadata and Artifacts

Each training run records:

- **Tags (`tags/`)**:
  - `git_commit`: Resolved Git commit hash (`resolve_git_commit_hash`).
  - `dataset_version`: Deterministic SHA-256 content hash of the dataset manifest or `dataset.yaml` (`compute_dataset_version_hash`).
  - `python_version`, `platform`, `package_version`, and `seed`.
- **Parameters (`params/`)**: Full serialized model and training configuration (`DetectorConfig` or `RecognizerConfig`).
- **Metrics (`metrics/`)**: Step-indexed validation metrics per epoch (`val_map50`, `val_map50_95`, `exact_match_accuracy`, `character_error_rate`, `positional_accuracy`, `learning_rate`) and final test split metrics.
- **Artifacts (`artifacts/`)**: `best.ckpt.json` and `last.ckpt.json` checkpoints.
