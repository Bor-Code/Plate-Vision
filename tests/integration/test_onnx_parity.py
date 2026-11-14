from pathlib import Path

from plate_vision.config.schema import RecognizerConfig
from plate_vision.data.synthetic import encode_grayscale_png
from plate_vision.detection.models import DetectorCheckpoint, DetectorMetrics
from plate_vision.detection.trainer import save_detector_checkpoint
from plate_vision.export.onnx_exporter import (
    export_detector_to_onnx,
    export_recognizer_to_onnx,
)
from plate_vision.export.runtime import (
    OnnxDetectorRuntime,
    OnnxRecognizerRuntime,
    verify_detector_parity,
    verify_recognizer_parity,
)
from plate_vision.pipeline.runner import PlateVisionPipeline
from plate_vision.recognition.charset import CharsetEncoder
from plate_vision.recognition.model import CRNNModel
from plate_vision.recognition.trainer import (
    RecognizerCheckpoint,
    RecognizerEpochMetrics,
    save_recognizer_checkpoint,
)


def test_onnx_parity_and_end_to_end_runtime_inference(tmp_path: Path) -> None:
    det_ckpt_path = tmp_path / "det_best.ckpt.json"
    save_detector_checkpoint(
        DetectorCheckpoint(
            epoch=3,
            best_fitness=0.92,
            metrics=DetectorMetrics(precision=0.93, recall=0.91, map50=0.94, map50_95=0.918),
            model_architecture="yolo11n.pt",
            input_size=640,
            state={"score_bias": 0.92},
        ),
        det_ckpt_path,
    )

    encoder = CharsetEncoder()
    model = CRNNModel(config=RecognizerConfig(), encoder=encoder, seed=42)
    sample_crop = [[(x * 3 + y * 7) % 256 for x in range(128)] for y in range(32)]
    target_indices = encoder.encode("34ABC123")
    model.compute_ctc_loss_and_update(
        sample_crop,
        target_indices,
        learning_rate=0.01,
        grad_clip_norm=5.0,
    )

    rec_ckpt_path = tmp_path / "rec_best.ckpt.json"
    save_recognizer_checkpoint(
        RecognizerCheckpoint(
            epoch=3,
            best_exact_match=1.0,
            best_cer=0.0,
            metrics=RecognizerEpochMetrics(
                epoch=3,
                train_loss=0.05,
                val_loss=0.05,
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
        rec_ckpt_path,
    )

    det_onnx_path = tmp_path / "detector.onnx"
    rec_onnx_path = tmp_path / "recognizer.onnx"
    rec_int8_path = tmp_path / "recognizer_int8.onnx"

    export_detector_to_onnx(det_ckpt_path, det_onnx_path, dynamic_quantize=False)
    export_recognizer_to_onnx(rec_ckpt_path, rec_onnx_path, dynamic_quantize=False)
    export_recognizer_to_onnx(rec_ckpt_path, rec_int8_path, dynamic_quantize=True)

    image_grid = [[120 for _ in range(160)] for _ in range(80)]
    png_bytes = encode_grayscale_png(image_grid)

    det_parity = verify_detector_parity(
        det_ckpt_path,
        det_onnx_path,
        sample_images=[(png_bytes, 160, 80)],
        tolerance=1e-6,
    )
    assert det_parity.passed is True
    assert det_parity.max_abs_diff == 0.0

    rec_parity_fp32 = verify_recognizer_parity(
        rec_ckpt_path,
        rec_onnx_path,
        sample_crops=[sample_crop],
        tolerance=1e-6,
    )
    assert rec_parity_fp32.passed is True
    assert rec_parity_fp32.max_abs_diff == 0.0

    rec_parity_int8 = verify_recognizer_parity(
        rec_ckpt_path,
        rec_int8_path,
        sample_crops=[sample_crop],
        tolerance=0.05,
    )
    assert rec_parity_int8.passed is True

    det_runtime = OnnxDetectorRuntime(det_onnx_path)
    rec_runtime = OnnxRecognizerRuntime(rec_onnx_path)
    pipeline = PlateVisionPipeline(detector=det_runtime, recognizer=rec_runtime)
    infer_res = pipeline.predict_bytes(png_bytes)
    assert len(infer_res.plates) == 1
    assert infer_res.plates[0].plate_text == "34ABC123"
