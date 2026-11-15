from pathlib import Path

from plate_vision.evaluation.cli import build_evaluate_parser
from plate_vision.export.cli import build_export_parser
from plate_vision.pipeline.cli import build_predict_parser


def test_export_cli_parser_arguments(tmp_path: Path) -> None:
    parser = build_export_parser()
    det = tmp_path / "det.ckpt.json"
    rec = tmp_path / "rec.ckpt.json"
    out = tmp_path / "out"
    args = parser.parse_args(
        [
            "--detector-checkpoint",
            str(det),
            "--recognizer-checkpoint",
            str(rec),
            "--output-dir",
            str(out),
            "--quantize-int8",
        ]
    )
    assert args.detector_checkpoint == det
    assert args.recognizer_checkpoint == rec
    assert args.output_dir == out
    assert args.quantize_int8 is True


def test_evaluate_and_predict_cli_parsers(tmp_path: Path) -> None:
    eval_parser = build_evaluate_parser()
    eval_args = eval_parser.parse_args(
        [
            "--detector-onnx",
            str(tmp_path / "d.onnx"),
            "--recognizer-onnx",
            str(tmp_path / "r.onnx"),
            "--test-manifest",
            str(tmp_path / "test.jsonl"),
            "--output-dir",
            str(tmp_path / "eval"),
            "--country-code",
            "DE",
        ]
    )
    assert eval_args.country_code == "DE"

    pred_parser = build_predict_parser()
    pred_args = pred_parser.parse_args(
        [
            "--detector-onnx",
            str(tmp_path / "d.onnx"),
            "--recognizer-onnx",
            str(tmp_path / "r.onnx"),
            "--image",
            str(tmp_path / "1.png"),
            "--image",
            str(tmp_path / "2.png"),
            "--output-json",
            str(tmp_path / "preds.json"),
        ]
    )
    assert len(pred_args.image) == 2
