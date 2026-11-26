import json
from pathlib import Path

from plate_vision.config.schema import RecognizerConfig
from plate_vision.data.synthetic import encode_grayscale_png
from plate_vision.detection.models import DetectorCheckpoint, DetectorMetrics
from plate_vision.detection.trainer import save_detector_checkpoint
from plate_vision.export.cli import run_export_cli
from plate_vision.recognition.charset import CharsetEncoder
from plate_vision.recognition.model import CRNNModel
from plate_vision.recognition.trainer import (
    RecognizerCheckpoint,
    RecognizerEpochMetrics,
    save_recognizer_checkpoint,
)
from plate_vision.video.cli import run_predict_video_cli


def test_predict_video_cli_end_to_end_workflow(tmp_path: Path) -> None:
    det_ckpt = tmp_path / "det.ckpt.json"
    save_detector_checkpoint(
        DetectorCheckpoint(
            epoch=2,
            best_fitness=0.92,
            metrics=DetectorMetrics(precision=0.93, recall=0.91, map50=0.94, map50_95=0.91),
            model_architecture="yolo11n.pt",
            input_size=640,
            state={"score_bias": 0.92},
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
    run_export_cli(
        [
            "--detector-checkpoint",
            str(det_ckpt),
            "--recognizer-checkpoint",
            str(rec_ckpt),
            "--output-dir",
            str(onnx_dir),
        ]
    )

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True)
    textured = [[35 if (x + y) % 2 == 0 else 185 for x in range(160)] for y in range(80)]
    (frames_dir / "frame_001.png").write_bytes(encode_grayscale_png(textured))
    (frames_dir / "frame_002.png").write_bytes(encode_grayscale_png(textured))
    extra_frame = tmp_path / "extra_003.png"
    extra_frame.write_bytes(encode_grayscale_png(textured))

    out_json = tmp_path / "video_out" / "report.json"
    result = run_predict_video_cli(
        [
            "--detector-onnx",
            str(onnx_dir / "detector.onnx"),
            "--recognizer-onnx",
            str(onnx_dir / "recognizer.onnx"),
            "--frames-dir",
            str(frames_dir),
            "--frame",
            str(extra_frame),
            "--fps",
            "25.0",
            "--camera-id",
            "HIGHWAY_01",
            "--check-tamper",
            "--output-json",
            str(out_json),
        ]
    )

    assert out_json.is_file()
    written = json.loads(out_json.read_text(encoding="utf-8"))
    assert written == result
    assert result["camera_id"] == "HIGHWAY_01"
    assert result["frames_processed"] == 3
    assert result["unique_tracks"] >= 1
    assert len(result["final_tracks"]) >= 1
    assert result["traffic_analytics"]["total_vehicles"] >= 1
    assert len(result["tamper_assessments"]) == 3
    assert result["tamper_alerted_cameras"] == []
