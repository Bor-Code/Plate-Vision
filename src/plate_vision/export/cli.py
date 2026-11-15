import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from plate_vision.export.onnx_exporter import (
    export_detector_to_onnx,
    export_recognizer_to_onnx,
    quantize_onnx_model,
)


def build_export_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export_models",
        description="Export detector and recognizer checkpoints to ONNX",
    )
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        required=True,
        help="Path to detector checkpoint (.ckpt.json)",
    )
    parser.add_argument(
        "--recognizer-checkpoint",
        type=Path,
        required=True,
        help="Path to recognizer checkpoint (.ckpt.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for exported ONNX models",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--quantize-int8",
        action="store_true",
        help="Also export dynamically quantized int8 ONNX artifacts",
    )
    return parser


def run_export_cli(argv: Sequence[str] | None = None) -> dict[str, str]:
    parser = build_export_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    det_onnx = out_dir / "detector.onnx"
    rec_onnx = out_dir / "recognizer.onnx"

    det_meta = export_detector_to_onnx(
        args.detector_checkpoint,
        det_onnx,
        opset_version=args.opset_version,
    )
    rec_meta = export_recognizer_to_onnx(
        args.recognizer_checkpoint,
        rec_onnx,
        opset_version=args.opset_version,
    )

    manifest: dict[str, str] = {
        "detector_onnx": str(det_onnx),
        "detector_sha256": det_meta.sha256,
        "recognizer_onnx": str(rec_onnx),
        "recognizer_sha256": rec_meta.sha256,
    }

    if args.quantize_int8:
        det_int8 = out_dir / "detector_int8.onnx"
        rec_int8 = out_dir / "recognizer_int8.onnx"
        det_q = quantize_onnx_model(det_onnx, det_int8)
        rec_q = quantize_onnx_model(rec_onnx, rec_int8)
        manifest["detector_int8_onnx"] = str(det_int8)
        manifest["detector_int8_sha256"] = det_q.sha256
        manifest["recognizer_int8_onnx"] = str(rec_int8)
        manifest["recognizer_int8_sha256"] = rec_q.sha256

    manifest_path = out_dir / "export_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    run_export_cli()


if __name__ == "__main__":
    main()
