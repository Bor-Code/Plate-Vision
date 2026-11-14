import argparse
from collections.abc import Sequence
from pathlib import Path

from plate_vision.config import DetectorConfig, load_component_config
from plate_vision.detection.models import DetectorTrainingSummary
from plate_vision.detection.trainer import DetectorTrainer
from plate_vision.utils.logging import configure_logging, get_logger, log_environment_info
from plate_vision.utils.tracking import MlflowExperimentTracker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="train_detector")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/detection/default.yaml"),
        help="Path to detector YAML configuration",
    )
    parser.add_argument(
        "--dataset-yaml",
        type=Path,
        required=True,
        help="Path to YOLO dataset.yaml file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/detector"),
        help="Directory to store checkpoints and training summary",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional checkpoint path to resume training from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed",
    )
    parser.add_argument(
        "--tracking-dir",
        type=Path,
        default=None,
        help="Optional MLflow tracking directory",
    )
    return parser


def run_train_detector(argv: Sequence[str] | None = None) -> DetectorTrainingSummary:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    configure_logging(level="INFO", json_format=True, service_name="train-detector")
    logger = get_logger("plate_vision.detection.cli")
    seed = int(args.seed)
    log_environment_info(logger, seed=seed)

    config = load_component_config(Path(args.config), DetectorConfig)
    tracker = (
        MlflowExperimentTracker(
            tracking_dir=Path(args.tracking_dir),
            experiment_name="detector",
            seed=seed,
        )
        if args.tracking_dir is not None
        else None
    )
    trainer = DetectorTrainer(config=config, seed=seed, tracker=tracker)

    summary = trainer.train(
        dataset_yaml=Path(args.dataset_yaml),
        output_dir=Path(args.output_dir),
        resume_from=Path(args.resume_from) if args.resume_from is not None else None,
    )
    summary_path = Path(args.output_dir) / "training_summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    logger.info(
        "detector_training_completed",
        epochs_completed=summary.epochs_completed,
        best_epoch=summary.best_epoch,
        stopped_early=summary.stopped_early,
        map50=summary.best_metrics.map50,
        map50_95=summary.best_metrics.map50_95,
        summary_path=str(summary_path),
    )
    return summary


def main() -> int:
    run_train_detector()
    return 0


__all__ = [
    "build_parser",
    "main",
    "run_train_detector",
]
