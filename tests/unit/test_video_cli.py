from pathlib import Path

import pytest

from plate_vision.utils.exceptions import InvalidInputError
from plate_vision.video.cli import build_video_predict_parser, run_predict_video_cli


def test_video_cli_parser_and_validation(tmp_path: Path) -> None:
    parser = build_video_predict_parser()
    det = tmp_path / "det.onnx"
    rec = tmp_path / "rec.onnx"
    frames_dir = tmp_path / "frames"
    out_json = tmp_path / "video_report.json"

    args = parser.parse_args(
        [
            "--detector-onnx",
            str(det),
            "--recognizer-onnx",
            str(rec),
            "--frames-dir",
            str(frames_dir),
            "--fps",
            "30.0",
            "--camera-id",
            "GATE_02",
            "--check-tamper",
            "--output-json",
            str(out_json),
        ]
    )
    assert args.detector_onnx == det
    assert args.recognizer_onnx == rec
    assert args.frames_dir == frames_dir
    assert args.fps == pytest.approx(30.0)
    assert args.camera_id == "GATE_02"
    assert args.check_tamper is True
    assert args.output_json == out_json

    with pytest.raises(InvalidInputError):
        run_predict_video_cli(
            [
                "--detector-onnx",
                str(det),
                "--recognizer-onnx",
                str(rec),
                "--frames-dir",
                str(frames_dir),
                "--fps",
                "0.0",
                "--output-json",
                str(out_json),
            ]
        )

    with pytest.raises(InvalidInputError):
        run_predict_video_cli(
            [
                "--detector-onnx",
                str(det),
                "--recognizer-onnx",
                str(rec),
                "--frames-dir",
                str(frames_dir),
                "--output-json",
                str(out_json),
            ]
        )

    frames_dir.mkdir(parents=True)
    with pytest.raises(InvalidInputError):
        run_predict_video_cli(
            [
                "--detector-onnx",
                str(det),
                "--recognizer-onnx",
                str(rec),
                "--frames-dir",
                str(frames_dir),
                "--output-json",
                str(out_json),
            ]
        )

    with pytest.raises(InvalidInputError):
        run_predict_video_cli(
            [
                "--detector-onnx",
                str(det),
                "--recognizer-onnx",
                str(rec),
                "--frame",
                str(tmp_path / "missing_frame.png"),
                "--output-json",
                str(out_json),
            ]
        )
