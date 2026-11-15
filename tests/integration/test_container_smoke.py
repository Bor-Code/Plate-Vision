import base64
from pathlib import Path

from plate_vision.config.schema import RecognizerConfig, ServingConfig
from plate_vision.data.synthetic import encode_grayscale_png
from plate_vision.detection.models import DetectorCheckpoint, DetectorMetrics
from plate_vision.detection.trainer import save_detector_checkpoint
from plate_vision.export.onnx_exporter import (
    export_detector_to_onnx,
    export_recognizer_to_onnx,
)
from plate_vision.recognition.charset import CharsetEncoder
from plate_vision.recognition.model import CRNNModel
from plate_vision.recognition.trainer import (
    RecognizerCheckpoint,
    RecognizerEpochMetrics,
    save_recognizer_checkpoint,
)
from plate_vision.serving.app import AsgiTestClient, create_app


def test_end_to_end_container_service_smoke_with_onnx_artifacts(tmp_path: Path) -> None:
    det_ckpt = tmp_path / "det.ckpt.json"
    save_detector_checkpoint(
        DetectorCheckpoint(
            epoch=2,
            best_fitness=0.90,
            metrics=DetectorMetrics(precision=0.91, recall=0.89, map50=0.92, map50_95=0.898),
            model_architecture="yolo11n.pt",
            input_size=640,
            state={"score_bias": 0.93},
        ),
        det_ckpt,
    )

    encoder = CharsetEncoder()
    model = CRNNModel(config=RecognizerConfig(), encoder=encoder, seed=42)
    sample_crop = [[120 for _ in range(128)] for _ in range(32)]
    model.compute_ctc_loss_and_update(
        sample_crop,
        encoder.encode("06ABC123"),
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

    det_onnx = tmp_path / "detector.onnx"
    rec_onnx = tmp_path / "recognizer.onnx"
    export_detector_to_onnx(det_ckpt, det_onnx)
    export_recognizer_to_onnx(rec_ckpt, rec_onnx)

    serving_cfg = ServingConfig(
        detector_onnx_path=det_onnx,
        recognizer_onnx_path=rec_onnx,
    )
    app = create_app(config=serving_cfg)
    client = AsgiTestClient(app)

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["ready"] is True

    png_bytes = encode_grayscale_png([[120 for _ in range(160)] for _ in range(80)])
    b64_img = base64.b64encode(png_bytes).decode("ascii")
    pred = client.post("/v1/predict", json={"image_base64": b64_img, "mime_type": "image/png"})
    assert pred.status_code == 200
    assert len(pred.json()["plates"]) == 1
    assert pred.json()["plates"][0]["plate_text"] == "06ABC123"
