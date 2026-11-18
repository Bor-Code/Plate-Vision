import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from plate_vision.config.schema import DatasetSplitRatios
from plate_vision.data.deduplication import deduplicate_records
from plate_vision.data.exporters import export_recognizer_manifests, export_yolo_dataset
from plate_vision.data.profiles import get_country_profile
from plate_vision.data.splitter import stratified_split
from plate_vision.data.synthetic import SyntheticPlateConfig, generate_synthetic_dataset
from plate_vision.utils.exceptions import InvalidInputError
from plate_vision.utils.logging import configure_logging, get_logger, log_environment_info


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="prepare_dataset",
        description="Generate synthetic plates, deduplicate, split, and export datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root output directory for generated data and exports",
    )
    parser.add_argument(
        "--country-code",
        type=str,
        default="TR",
        help="Country profile code for plate synthesis",
    )
    parser.add_argument(
        "--synthetic-count",
        type=int,
        default=30,
        help="Number of synthetic plate images to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def run_prepare_dataset(args: argparse.Namespace) -> dict[str, Any]:
    synthetic_count = int(args.synthetic_count)
    if synthetic_count < 1:
        raise InvalidInputError(
            "synthetic_count must be >= 1",
            details={"synthetic_count": synthetic_count},
        )
    seed = int(args.seed)
    output_dir = Path(args.output_dir)
    profile = get_country_profile(str(args.country_code))

    raw_dir = output_dir / "raw_synthetic"
    records = generate_synthetic_dataset(
        count=synthetic_count,
        output_dir=raw_dir,
        config=SyntheticPlateConfig(seed=seed),
        profile=profile,
    )
    dedup = deduplicate_records(records)
    split_ds = stratified_split(dedup.unique_records, DatasetSplitRatios(), seed=seed)

    yolo_yaml = export_yolo_dataset(split_ds, output_dir / "yolo")
    manifests = export_recognizer_manifests(split_ds, output_dir / "recognizer")

    summary: dict[str, Any] = {
        "country_code": profile.country_code,
        "total_generated": len(records),
        "unique_records": len(dedup.unique_records),
        "duplicates_removed": dedup.removed_count,
        "train_count": len(split_ds.train),
        "val_count": len(split_ds.val),
        "test_count": len(split_ds.test),
        "yolo_dataset_yaml": str(yolo_yaml),
        "recognizer_manifests": {split.value: str(path) for split, path in manifests.items()},
    }
    summary_path = output_dir / "dataset_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(level="INFO", json_format=True, service_name="prepare-dataset")
    logger = get_logger("plate_vision.data.cli")
    log_environment_info(logger, seed=int(args.seed))
    summary = run_prepare_dataset(args)
    logger.info("dataset_prepared", **summary)
    return 0


__all__ = [
    "main",
    "parse_args",
    "run_prepare_dataset",
]
