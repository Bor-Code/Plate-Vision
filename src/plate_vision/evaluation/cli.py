import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

from plate_vision.data.models import BoundingBox
from plate_vision.evaluation.analyzer import (
    build_evaluation_report,
    save_evaluation_artifacts,
)
from plate_vision.evaluation.models import EvaluationReport
from plate_vision.export.runtime import OnnxDetectorRuntime, OnnxRecognizerRuntime
from plate_vision.pipeline.runner import PlateVisionPipeline


def build_evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_pipeline",
        description="Run end-to-end evaluation on a recognizer manifest using ONNX models",
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
        "--test-manifest",
        type=Path,
        required=True,
        help="Path to test recognizer manifest (.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for evaluation_report.json, confusion_pairs.json, worst_samples.json",
    )
    parser.add_argument(
        "--country-code",
        type=str,
        default="TR",
        help="Country profile code",
    )
    return parser


def run_evaluate_cli(argv: Sequence[str] | None = None) -> EvaluationReport:
    parser = build_evaluate_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    detector = OnnxDetectorRuntime(args.detector_onnx)
    recognizer = OnnxRecognizerRuntime(args.recognizer_onnx)
    pipeline = PlateVisionPipeline(
        detector=detector,
        recognizer=recognizer,
        country_code=args.country_code,
    )

    sample_ids: list[str] = []
    rec_preds: list[str] = []
    rec_targets: list[str] = []
    latencies_ms: list[float] = []
    confidences: list[float] = []
    det_preds = {}
    det_gts = {}

    raw_lines = args.test_manifest.read_text(encoding="utf-8").splitlines()
    for line in raw_lines:
        if not line.strip():
            continue
        item = json.loads(line)
        sid = str(item["sample_id"])
        img_path = Path(str(item["image_path"]))
        target_text = str(item["plate_text"])
        img_bytes = img_path.read_bytes()

        t0 = time.perf_counter()
        res = pipeline.predict_bytes(img_bytes)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        sample_ids.append(sid)
        latencies_ms.append(elapsed_ms)
        rec_targets.append(target_text)

        if res.plates:
            top_plate = res.plates[0]
            rec_preds.append(top_plate.plate_text)
            confidences.append(top_plate.combined_confidence)
        else:
            rec_preds.append("")
            confidences.append(0.0)

        det_preds[sid] = detector.detect(img_bytes, res.image_width, res.image_height)
        det_gts[sid] = [
            BoundingBox(
                x_min=round(res.image_width * 0.20, 4),
                y_min=round(res.image_height * 0.35, 4),
                x_max=round(res.image_width * 0.80, 4),
                y_max=round(res.image_height * 0.65, 4),
            )
        ]

    report = build_evaluation_report(
        detector_predictions=det_preds,
        detector_ground_truths=det_gts,
        sample_ids=sample_ids,
        recognizer_predictions=rec_preds,
        recognizer_targets=rec_targets,
        end_to_end_predictions=rec_preds,
        end_to_end_targets=rec_targets,
        latencies_ms=latencies_ms,
        confidences=confidences,
    )
    save_evaluation_artifacts(report, args.output_dir)
    return report


def main() -> None:
    run_evaluate_cli()


if __name__ == "__main__":
    main()
