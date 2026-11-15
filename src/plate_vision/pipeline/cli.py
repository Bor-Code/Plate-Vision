import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from plate_vision.export.runtime import OnnxDetectorRuntime, OnnxRecognizerRuntime
from plate_vision.pipeline.runner import PlateVisionPipeline


def build_predict_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict_plate",
        description="Run license plate recognition on one or more images using ONNX models",
    )
    parser.add_argument(
        "--detector-onnx",
        type=Path,
        required=True,
        help="Path to detector.onnx",
    )
    parser.add_argument(
        "--recognizer-onnx",
        type=Path,
        required=True,
        help="Path to recognizer.onnx",
    )
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        required=True,
        help="Path to input image file (can be passed multiple times)",
    )
    parser.add_argument(
        "--country-code",
        type=str,
        default="TR",
        help="Country profile code",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Path to write JSON inference results",
    )
    return parser


def run_predict_cli(argv: Sequence[str] | None = None) -> list[dict[str, Any]]:
    parser = build_predict_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    detector = OnnxDetectorRuntime(args.detector_onnx)
    recognizer = OnnxRecognizerRuntime(args.recognizer_onnx)
    pipeline = PlateVisionPipeline(
        detector=detector,
        recognizer=recognizer,
        country_code=args.country_code,
    )

    records: list[dict[str, Any]] = []
    for image_path in args.image:
        raw_bytes = image_path.read_bytes()
        res = pipeline.predict_bytes(raw_bytes)
        records.append(
            {
                "image_path": str(image_path),
                "result": res.model_dump(),
            }
        )

    out_path: Path = args.output_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def main() -> None:
    run_predict_cli()


if __name__ == "__main__":
    main()
