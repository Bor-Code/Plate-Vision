import json
from pathlib import Path

from plate_vision.config.schema import RecognizerConfig
from plate_vision.data.synthetic import encode_grayscale_png
from plate_vision.detection.models import DetectorCheckpoint, DetectorMetrics
from plate_vision.detection.trainer import save_detector_checkpoint
from plate_vision.evaluation.cli import run_evaluate_cli
from plate_vision.export.cli import run_export_cli
from plate_vision.pipeline.cli import run_predict_cli
from plate_vision.recognition.charset import CharsetEncoder
from plate_vision.recognition.model import CRNNModel
from plate_vision.recognition.trainer import (
    RecognizerCheckpoint,
    RecognizerEpochMetrics,
    save_recognizer_checkpoint,
)


def test_end_to_end_cli_export_evaluate_and_predict(tmp_path: Path) -> None:
    det_ckpt = tmp_path / "det.ckpt.json"
    save_detector_checkpoint(
        DetectorCheckpoint(
            epoch=2,
            best_fitness=0.91,
            metrics=DetectorMetrics(precision=0.92, recall=0.90, map50=0.93, map50_95=0.908),
            model_architecture="yolo11n.pt",
            input_size=640,
            state={"score_bias": 0.91},
        ),
        det_ckpt,
    )

    encoder = CharsetEncoder()
    model = CRNNModel(config=RecognizerConfig(), encoder=encoder, seed=42)
    sample_crop = [[120 for _ in range(128)] for _ in range(32)]
    model.compute_ctc_loss_and_update(
        sample_crop,
        encoder.encode("34ABC123"),
        learning_rate=0.01,
        grad_clip_norm=5.0,
    )
    rec_ckpt = tmp_path / "rec.ckpt.json"
    save_recognizer_checkpoint(
        RecognizerCheckpoint(
            epoch=2,
            best_exact_match=1.0,
            best_cer=0.0,
            metrics=RecognizerEpochMetrics(
                epoch=2,
                train_loss=0.1,
                val_loss=0.1,
                exact_match_accuracy=1.0,
                character_error_rate=0.0,
                positional_accuracy=1.0,
                learning_rate=0.001,
                amp_scale=1024.0,
            ),
            amp_scale=1024.0,
            charset="".join(encoder.characters),
            model_state=model.state_dict(),
        ),
        rec_ckpt,
    )

    onnx_dir = tmp_path / "onnx_out"
    export_manifest = run_export_cli(
        [
            "--detector-checkpoint",
            str(det_ckpt),
            "--recognizer-checkpoint",
            str(rec_ckpt),
            "--output-dir",
            str(onnx_dir),
            "--quantize-int8",
        ]
    )
    assert (onnx_dir / "detector.onnx").is_file()
    assert (onnx_dir / "recognizer.onnx").is_file()
    assert (onnx_dir / "detector_int8.onnx").is_file()
    assert (onnx_dir / "recognizer_int8.onnx").is_file()
    assert "detector_int8_sha256" in export_manifest

    img_path = tmp_path / "sample.png"
    img_path.write_bytes(encode_grayscale_png([[120 for _ in range(160)] for _ in range(80)]))

    manifest_path = tmp_path / "test_manifest.jsonl"
    manifest_entry = {
        "sample_id": "s_01",
        "image_path": str(img_path),
        "plate_text": "34ABC123",
        "country_code": "TR",
    }
    manifest_path.write_text(json.dumps(manifest_entry) + "\n", encoding="utf-8")

    eval_dir = tmp_path / "eval_out"
    eval_report = run_evaluate_cli(
        [
            "--detector-onnx",
            str(onnx_dir / "detector.onnx"),
            "--recognizer-onnx",
            str(onnx_dir / "recognizer.onnx"),
            "--test-manifest",
            str(manifest_path),
            "--output-dir",
            str(eval_dir),
        ]
    )
    assert eval_report.end_to_end.plate_level_accuracy == 1.0
    assert (eval_dir / "evaluation_report.json").is_file()

    preds_json = tmp_path / "preds.json"
    preds = run_predict_cli(
        [
            "--detector-onnx",
            str(onnx_dir / "detector.onnx"),
            "--recognizer-onnx",
            str(onnx_dir / "recognizer.onnx"),
            "--image",
            str(img_path),
            "--output-json",
            str(preds_json),
        ]
    )
    assert len(preds) == 1
    assert preds[0]["result"]["plates"][0]["plate_text"] == "34ABC123"
    assert preds_json.is_file()
