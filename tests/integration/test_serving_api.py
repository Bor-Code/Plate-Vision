import base64
from collections.abc import Sequence
from pathlib import Path

from plate_vision.config.schema import ServingConfig
from plate_vision.data.models import BoundingBox
from plate_vision.data.synthetic import encode_grayscale_png
from plate_vision.detection.models import DetectionPrediction
from plate_vision.pipeline.runner import PlateVisionPipeline
from plate_vision.recognition.decoding import DecodedPlate
from plate_vision.serving.app import AsgiTestClient, create_app


class _FakeDetector:
    def detect(
        self,
        image_bytes: bytes,
        image_width: int,
        image_height: int,
    ) -> list[DetectionPrediction]:
        _ = (image_bytes, image_width, image_height)
        return [
            DetectionPrediction(
                bbox=BoundingBox(x_min=10.0, y_min=8.0, x_max=90.0, y_max=36.0),
                confidence=0.96,
                class_id=0,
            )
        ]


class _FakeRecognizer:
    def recognize(
        self,
        crop_pixels: Sequence[Sequence[int]],
        *,
        use_beam_search: bool = False,
    ) -> DecodedPlate:
        _ = (crop_pixels, use_beam_search)
        return DecodedPlate(text="34ABC123", confidence=0.94, token_indices=(1, 2))


def _sample_png_bytes() -> bytes:
    grid = [[130 for _ in range(120)] for _ in range(48)]
    return encode_grayscale_png(grid)


def test_serving_endpoints_single_upload_batch_probes_and_metrics() -> None:
    pipeline = PlateVisionPipeline(
        detector=_FakeDetector(),
        recognizer=_FakeRecognizer(),
    )
    config = ServingConfig(max_batch_size=2, max_payload_bytes=32768)
    app = create_app(config=config, pipeline=pipeline)
    client = AsgiTestClient(app)

    health_resp = client.get("/healthz")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "ok"

    ready_resp = client.get("/readyz")
    assert ready_resp.status_code == 200
    assert ready_resp.json()["ready"] is True

    raw_png = _sample_png_bytes()
    b64_png = base64.b64encode(raw_png).decode("ascii")

    single_resp = client.post(
        "/v1/predict",
        json={"image_base64": b64_png, "mime_type": "image/png"},
    )
    assert single_resp.status_code == 200
    assert single_resp.json()["plates"][0]["plate_text"] == "34ABC123"

    upload_resp = client.post(
        "/v1/predict/upload",
        raw_body=raw_png,
        content_type="image/png",
    )
    assert upload_resp.status_code == 200
    assert upload_resp.json()["plates"][0]["plate_text"] == "34ABC123"

    batch_resp = client.post(
        "/v1/predict/batch",
        json={
            "items": [
                {"image_base64": b64_png, "mime_type": "image/png"},
                {"image_base64": b64_png, "mime_type": "image/png"},
            ]
        },
    )
    assert batch_resp.status_code == 200
    assert batch_resp.json()["total_plates"] == 2

    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    assert "plate_vision_plates_detected_total 4" in metrics_resp.text


def test_serving_error_handling_and_unready_probe(tmp_path: Path) -> None:
    unready_cfg = ServingConfig(
        detector_onnx_path=tmp_path / "missing_det.onnx",
        recognizer_onnx_path=tmp_path / "missing_rec.onnx",
        max_batch_size=1,
        max_payload_bytes=1024,
    )
    app = create_app(config=unready_cfg, pipeline=None)
    client = AsgiTestClient(app)

    ready_resp = client.get("/readyz")
    assert ready_resp.status_code == 503
    assert ready_resp.json()["ready"] is False

    bad_mime_resp = client.post(
        "/v1/predict",
        json={"image_base64": "AAAA", "mime_type": "application/pdf"},
    )
    assert bad_mime_resp.status_code == 422
    assert bad_mime_resp.json()["error_code"] == "INVALID_INPUT_ERROR"

    bad_b64_resp = client.post(
        "/v1/predict",
        json={"image_base64": "!!!not_base64!!!", "mime_type": "image/png"},
    )
    assert bad_b64_resp.status_code == 422

    too_many_batch = client.post(
        "/v1/predict/batch",
        json={
            "items": [
                {"image_base64": "AAAA", "mime_type": "image/png"},
                {"image_base64": "AAAA", "mime_type": "image/png"},
            ]
        },
    )
    assert too_many_batch.status_code == 422

    bad_upload = client.post(
        "/v1/predict/upload",
        raw_body=b"hello",
        content_type="text/plain",
    )
    assert bad_upload.status_code == 422
