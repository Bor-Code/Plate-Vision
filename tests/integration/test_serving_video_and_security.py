import base64
from collections.abc import Sequence

from plate_vision.config.schema import ServingConfig
from plate_vision.data.models import BoundingBox
from plate_vision.data.synthetic import encode_grayscale_png
from plate_vision.detection.models import DetectionPrediction
from plate_vision.pipeline.runner import PlateVisionPipeline
from plate_vision.recognition.decoding import DecodedPlate
from plate_vision.serving.app import AsgiTestClient, create_app


class _StubDetector:
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
                confidence=0.95,
                class_id=0,
            )
        ]


class _StubGermanRecognizer:
    def recognize(
        self,
        crop_pixels: Sequence[Sequence[int]],
        *,
        use_beam_search: bool = False,
    ) -> DecodedPlate:
        _ = (crop_pixels, use_beam_search)
        return DecodedPlate(text="B AB 1234", confidence=0.92, token_indices=(1, 2))


def _sample_b64_png() -> str:
    grid = [[130 for _ in range(120)] for _ in range(48)]
    raw_png = encode_grayscale_png(grid)
    return base64.b64encode(raw_png).decode("ascii")


def test_serving_api_key_rate_limiting_and_request_id_headers() -> None:
    pipeline = PlateVisionPipeline(
        detector=_StubDetector(),
        recognizer=_StubGermanRecognizer(),
    )
    config = ServingConfig(
        api_keys=("valid-api-key",),
        rate_limit_requests_per_minute=2,
        max_batch_size=3,
    )
    app = create_app(config=config, pipeline=pipeline)
    client = AsgiTestClient(app)

    health_resp = client.get("/healthz", headers={"X-Request-ID": "req-health-1"})
    assert health_resp.status_code == 200
    assert health_resp.headers["x-request-id"] == "req-health-1"

    unauth_resp = client.post(
        "/v1/predict",
        json={"image_base64": _sample_b64_png(), "mime_type": "image/png"},
        headers={"X-Request-ID": "req-unauth-1"},
    )
    assert unauth_resp.status_code == 401
    assert unauth_resp.headers["x-request-id"] == "req-unauth-1"
    assert unauth_resp.json()["error_code"] == "UNAUTHORIZED"

    b64_img = _sample_b64_png()
    ok_resp_1 = client.post(
        "/v1/predict",
        json={"image_base64": b64_img, "mime_type": "image/png", "country_code": "DE"},
        headers={"X-API-Key": "valid-api-key", "X-Request-ID": "req-de-1"},
    )
    assert ok_resp_1.status_code == 200
    assert ok_resp_1.headers["x-request-id"] == "req-de-1"
    assert ok_resp_1.headers["x-ratelimit-remaining"] == "1"
    plate_res = ok_resp_1.json()["plates"][0]
    assert plate_res["plate_text"] == "BAB1234"
    assert plate_res["country_code"] == "DE"
    assert plate_res["is_valid_format"] is True

    ok_resp_2 = client.post(
        "/v1/predict/video",
        json={
            "frames": [
                {"image_base64": b64_img, "mime_type": "image/png"},
                {"image_base64": b64_img, "mime_type": "image/png"},
            ],
            "fps": 25.0,
        },
        headers={"X-API-Key": "valid-api-key"},
    )
    assert ok_resp_2.status_code == 200
    video_data = ok_resp_2.json()
    assert video_data["frames_processed"] == 2
    assert video_data["unique_tracks"] == 1

    throttled_resp = client.post(
        "/v1/predict",
        json={"image_base64": b64_img, "mime_type": "image/png"},
        headers={"X-API-Key": "valid-api-key"},
    )
    assert throttled_resp.status_code == 429
    assert throttled_resp.json()["error_code"] == "RATE_LIMIT_EXCEEDED"
    assert throttled_resp.headers["x-ratelimit-remaining"] == "0"


def test_serving_video_batch_size_limit() -> None:
    pipeline = PlateVisionPipeline(
        detector=_StubDetector(),
        recognizer=_StubGermanRecognizer(),
    )
    config = ServingConfig(max_batch_size=1)
    app = create_app(config=config, pipeline=pipeline)
    client = AsgiTestClient(app)
    b64_img = _sample_b64_png()

    resp = client.post(
        "/v1/predict/video",
        json={
            "frames": [
                {"image_base64": b64_img, "mime_type": "image/png"},
                {"image_base64": b64_img, "mime_type": "image/png"},
            ],
            "fps": 30.0,
        },
    )
    assert resp.status_code == 422
    assert resp.json()["error_code"] == "INVALID_INPUT_ERROR"
