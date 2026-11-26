import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from plate_vision.export.runtime import OnnxDetectorRuntime, OnnxRecognizerRuntime
from plate_vision.pipeline.runner import PlateVisionPipeline
from plate_vision.utils.exceptions import InvalidInputError
from plate_vision.video.analytics import CalibrationZone, VideoTrafficAnalyzer
from plate_vision.video.processor import VideoFrameInput, VideoStreamProcessor
from plate_vision.video.tamper import CameraTamperMonitor

VIDEO_FRAME_EXTENSIONS: Final[frozenset[str]] = frozenset({".png", ".jpg", ".jpeg", ".webp"})


def build_video_predict_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict_video",
        description="Run video stream ALPR tracking, traffic analytics, and optional tamper checks",
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
        "--frames-dir",
        type=Path,
        default=None,
        help="Optional directory containing sequential frame images",
    )
    parser.add_argument(
        "--frame",
        type=Path,
        action="append",
        default=None,
        help="Path to an input video frame image (can be passed multiple times)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Frame rate in frames per second",
    )
    parser.add_argument(
        "--country-code",
        type=str,
        default="TR",
        help="Country profile code",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default="CAM_01",
        help="Camera identifier for tamper health checks and metadata",
    )
    parser.add_argument(
        "--pixels-per-meter",
        type=float,
        default=25.0,
        help="Calibration scale in pixels per meter for speed estimation",
    )
    parser.add_argument(
        "--speed-limit-kmh",
        type=float,
        default=90.0,
        help="Speed limit threshold in km/h for speeding detection",
    )
    parser.add_argument(
        "--check-tamper",
        action="store_true",
        help="Enable camera blackout, occlusion, glare, and defocus health checks",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Path to write structured video inference JSON report",
    )
    return parser


def _collect_frame_paths(
    frames_dir: Path | None,
    frame_paths: Sequence[Path] | None,
) -> list[Path]:
    collected: list[Path] = []
    if frame_paths is not None:
        for path in frame_paths:
            if not path.is_file():
                raise InvalidInputError(
                    f"Frame file does not exist: {path}",
                    details={"path": str(path)},
                )
            collected.append(path)
    if frames_dir is not None:
        if not frames_dir.exists() or not frames_dir.is_dir():
            raise InvalidInputError(
                f"Frames directory does not exist: {frames_dir}",
                details={"frames_dir": str(frames_dir)},
            )
        for path in sorted(frames_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in VIDEO_FRAME_EXTENSIONS:
                collected.append(path)
    if not collected:
        raise InvalidInputError("No video frame files provided via --frame or --frames-dir")
    return collected


def run_predict_video_cli(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = build_video_predict_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    fps = float(args.fps)
    if fps <= 0.0:
        raise InvalidInputError("fps must be > 0.0", details={"fps": fps})

    frame_paths = _collect_frame_paths(args.frames_dir, args.frame)
    detector = OnnxDetectorRuntime(args.detector_onnx)
    recognizer = OnnxRecognizerRuntime(args.recognizer_onnx)
    pipeline = PlateVisionPipeline(
        detector=detector,
        recognizer=recognizer,
        country_code=args.country_code,
    )

    frame_inputs = [
        VideoFrameInput(
            frame_index=idx,
            timestamp_ms=round((idx * 1000.0) / fps, 4),
            image_bytes=path.read_bytes(),
        )
        for idx, path in enumerate(frame_paths)
    ]

    processor = VideoStreamProcessor(pipeline=pipeline)
    summary = processor.process_stream(frame_inputs)
    analyzer = VideoTrafficAnalyzer(
        zone=CalibrationZone(
            pixels_per_meter=float(args.pixels_per_meter),
            speed_limit_kmh=float(args.speed_limit_kmh),
        )
    )
    traffic_report = analyzer.analyze(summary)

    tamper_assessments: list[dict[str, Any]] = []
    tamper_alerted: list[str] = []
    if args.check_tamper:
        tamper_monitor = CameraTamperMonitor()
        for frame in frame_inputs:
            assessment = tamper_monitor.evaluate_image_bytes(args.camera_id, frame.image_bytes)
            tamper_assessments.append(assessment.model_dump(mode="json"))
        tamper_alerted = list(tamper_monitor.alerted_cameras())

    payload: dict[str, Any] = {
        "camera_id": str(args.camera_id),
        "fps": fps,
        "frames_processed": summary.frames_processed,
        "unique_tracks": summary.unique_tracks,
        "final_tracks": [track.model_dump(mode="json") for track in summary.final_tracks],
        "traffic_analytics": traffic_report.model_dump(mode="json"),
        "tamper_alerted_cameras": tamper_alerted,
        "tamper_assessments": tamper_assessments,
    }

    out_path: Path = args.output_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    run_predict_video_cli()


__all__ = [
    "VIDEO_FRAME_EXTENSIONS",
    "build_video_predict_parser",
    "main",
    "run_predict_video_cli",
]
