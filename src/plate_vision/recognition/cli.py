import argparse
from collections.abc import Sequence
from pathlib import Path

from plate_vision.config import RecognizerConfig, load_component_config
from plate_vision.recognition.charset import CharsetEncoder
from plate_vision.recognition.dataset import RecognizerDataset
from plate_vision.recognition.trainer import RecognizerTrainer, RecognizerTrainingSummary
from plate_vision.utils.logging import configure_logging, get_logger, log_environment_info
from plate_vision.utils.tracking import MlflowExperimentTracker


def build_recognizer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="train_recognizer")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/recognition/default.yaml"),
        help="Path to recognizer YAML configuration",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        required=True,
        help="Directory containing recognizer_{train,val,test}.jsonl manifests",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/recognizer"),
        help="Directory to write recognizer checkpoints and training summary",
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


def run_train_recognizer(argv: Sequence[str] | None = None) -> RecognizerTrainingSummary:
    parser = build_recognizer_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    configure_logging(level="INFO", json_format=True, service_name="train-recognizer")
    logger = get_logger("plate_vision.recognition.cli")
    seed = int(args.seed)
    log_environment_info(logger, seed=seed)

    config = load_component_config(Path(args.config), RecognizerConfig)
    encoder = CharsetEncoder()
    manifest_dir = Path(args.manifest_dir)

    train_dataset = RecognizerDataset(
        manifest_dir / "recognizer_train.jsonl",
        encoder=encoder,
        seed=seed,
    )
    val_dataset = RecognizerDataset(
        manifest_dir / "recognizer_val.jsonl",
        encoder=encoder,
        seed=seed,
    )
    test_dataset = RecognizerDataset(
        manifest_dir / "recognizer_test.jsonl",
        encoder=encoder,
        seed=seed,
    )

    trainer = RecognizerTrainer(
        config=config,
        encoder=encoder,
        seed=seed,
        tracker=(
            MlflowExperimentTracker(
                tracking_dir=Path(args.tracking_dir),
                experiment_name="recognizer",
                seed=seed,
            )
            if args.tracking_dir is not None
            else None
        ),
    )
    output_dir = Path(args.output_dir)
    summary = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        output_dir=output_dir,
        resume_from=Path(args.resume_from) if args.resume_from is not None else None,
    )

    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    logger.info(
        "recognizer_training_completed",
        epochs_completed=summary.epochs_completed,
        best_epoch=summary.best_epoch,
        stopped_early=summary.stopped_early,
        exact_match_accuracy=summary.best_metrics.exact_match_accuracy,
        character_error_rate=summary.best_metrics.character_error_rate,
        summary_path=str(summary_path),
    )
    return summary


def main() -> int:
    run_train_recognizer()
    return 0


__all__ = [
    "build_recognizer_parser",
    "main",
    "run_train_recognizer",
]
