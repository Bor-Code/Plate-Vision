from pathlib import Path

import pytest

from plate_vision.config.schema import RecognizerConfig
from plate_vision.detection.models import DetectorCheckpoint, DetectorMetrics
from plate_vision.detection.trainer import save_detector_checkpoint
from plate_vision.export.onnx_exporter import (
    export_detector_to_onnx,
    export_recognizer_to_onnx,
    quantize_onnx_model,
    unpack_onnx_payload,
)
from plate_vision.export.runtime import OnnxDetectorRuntime, OnnxRecognizerRuntime
from plate_vision.recognition.charset import CharsetEncoder
from plate_vision.recognition.model import CRNNModel
from plate_vision.recognition.trainer import (
    RecognizerCheckpoint,
    RecognizerEpochMetrics,
    save_recognizer_checkpoint,
)
from plate_vision.utils.exceptions import ModelLoadError


def _write_sample_detector_ckpt(path: Path) -> Path:
    ckpt = DetectorCheckpoint(
        epoch=5,
        best_fitness=0.91,
        metrics=DetectorMetrics(precision=0.92, recall=0.90, map50=0.93, map50_95=0.908),
        model_architecture="yolo11n.pt",
        input_size=640,
        state={"score_bias": 0.91},
    )
    save_detector_checkpoint(ckpt, path)
    return path


def _write_sample_recognizer_ckpt(path: Path) -> Path:
    encoder = CharsetEncoder()
    model = CRNNModel(config=RecognizerConfig(), encoder=encoder, seed=42)
    metrics = RecognizerEpochMetrics(
        epoch=5,
        train_loss=0.2,
        val_loss=0.18,
        exact_match_accuracy=0.95,
        character_error_rate=0.02,
        positional_accuracy=0.97,
        learning_rate=0.001,
        amp_scale=1024.0,
    )
    ckpt = RecognizerCheckpoint(
        epoch=5,
        best_exact_match=0.95,
        best_cer=0.02,
        metrics=metrics,
        amp_scale=1024.0,
        charset="".join(encoder.characters),
        model_state=model.state_dict(),
    )
    save_recognizer_checkpoint(ckpt, path)
    return path


def test_export_and_quantize_detector_and_recognizer(tmp_path: Path) -> None:
    det_ckpt = _write_sample_detector_ckpt(tmp_path / "det.ckpt.json")
    det_onnx = tmp_path / "detector.onnx"
    det_meta = export_detector_to_onnx(det_ckpt, det_onnx)
    assert det_meta.model_type == "detector"
    assert det_meta.quantized is False
    assert det_onnx.is_file()

    det_q_onnx = tmp_path / "detector_int8.onnx"
    det_q_meta = quantize_onnx_model(det_onnx, det_q_onnx)
    assert det_q_meta.quantized is True

    rec_ckpt = _write_sample_recognizer_ckpt(tmp_path / "rec.ckpt.json")
    rec_onnx = tmp_path / "recognizer.onnx"
    rec_meta = export_recognizer_to_onnx(rec_ckpt, rec_onnx, dynamic_quantize=False)
    assert rec_meta.model_type == "recognizer"
    assert rec_meta.quantized is False

    rec_q_onnx = tmp_path / "recognizer_int8.onnx"
    rec_q_meta = quantize_onnx_model(rec_onnx, rec_q_onnx)
    assert rec_q_meta.quantized is True


def test_unpack_and_runtime_reject_corrupted_or_mismatched_models(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError, match="does not exist"):
        unpack_onnx_payload(tmp_path / "missing.onnx")

    bad_file = tmp_path / "bad.onnx"
    bad_file.write_bytes(b"NOT_ONNX_HEADER")
    with pytest.raises(ModelLoadError, match="Invalid ONNX binary header"):
        unpack_onnx_payload(bad_file)

    det_ckpt = _write_sample_detector_ckpt(tmp_path / "det.ckpt.json")
    det_onnx = tmp_path / "detector.onnx"
    export_detector_to_onnx(det_ckpt, det_onnx)

    with pytest.raises(ModelLoadError, match="Expected recognizer"):
        OnnxRecognizerRuntime(det_onnx)

    rec_ckpt = _write_sample_recognizer_ckpt(tmp_path / "rec.ckpt.json")
    rec_onnx = tmp_path / "recognizer.onnx"
    export_recognizer_to_onnx(rec_ckpt, rec_onnx)

    with pytest.raises(ModelLoadError, match="Expected detector"):
        OnnxDetectorRuntime(rec_onnx)
