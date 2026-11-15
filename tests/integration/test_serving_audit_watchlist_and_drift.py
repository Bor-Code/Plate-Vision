import base64
from collections.abc import Sequence

from plate_vision.config.schema import ServingConfig
from plate_vision.data.models import BoundingBox
from plate_vision.data.synthetic import encode_grayscale_png
from plate_vision.detection.models import DetectionPrediction
from plate_vision.evaluation.monitoring import DriftDetector
from plate_vision.pipeline.runner import PlateVisionPipeline
from plate_vision.pipeline.watchlist import (
    WatchlistCategory,
    WatchlistEntry,
    WatchlistMatcher,
)
from plate_vision.recognition.decoding import DecodedPlate
from plate_vision.serving.app import AsgiTestClient, create_app
from plate_vision.utils.storage import InMemoryPredictionStore


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


class _StubRecognizer:
    def recognize(
        self,
        crop_pixels: Sequence[Sequence[int]],
        *,
        use_beam_search: bool = False,
    ) -> DecodedPlate:
        _ = (crop_pixels, use_beam_search)
        return DecodedPlate(text="34ABC123", confidence=0.93, token_indices=(1,))


def _sample_b64_png() -> str:
    raw = encode_grayscale_png([[135 for _ in range(120)] for _ in range(48)])
    return base64.b64encode(raw).decode("ascii")


def test_serving_audit_watchlist_and_drift_endpoints() -> None:
    store = InMemoryPredictionStore(max_records=100)
    pipeline = PlateVisionPipeline(
        detector=_StubDetector(),
        recognizer=_StubRecognizer(),
    )
    watchlist = WatchlistMatcher(
        entries=[
            WatchlistEntry(
                plate_text="34ABC123",
                country_code="TR",
                category=WatchlistCategory.STOLEN,
                note="flag alert",
            )
        ],
        max_edit_distance=1,
    )
    drift_detector = DriftDetector(
        baseline_brightness=[120.0 + (i % 10) for i in range(30)],
        baseline_contrast=[40.0 + (i % 8) for i in range(30)],
        baseline_confidences=[0.88 + 0.002 * (i % 10) for i in range(30)],
        psi_threshold=0.25,
        num_bins=5,
    )
    app = create_app(
        config=ServingConfig(),
        pipeline=pipeline,
        storage=store,
        watchlist=watchlist,
        drift_detector=drift_detector,
    )
    client = AsgiTestClient(app)

    empty_audit = client.get("/v1/audit/recent?limit=10")
    assert empty_audit.status_code == 200
    assert empty_audit.json()["count"] == 0

    b64_png = _sample_b64_png()
    pred_resp = client.post(
        "/v1/predict",
        json={"image_base64": b64_png, "mime_type": "image/png"},
    )
    assert pred_resp.status_code == 200

    audit_resp = client.get("/v1/audit/recent?limit=5")
    assert audit_resp.status_code == 200
    assert audit_resp.json()["count"] == 1
    assert audit_resp.json()["records"][0]["plate_texts"] == ["34ABC123"]

    bad_audit = client.get("/v1/audit/recent?limit=0")
    assert bad_audit.status_code == 422

    wl_hit = client.post(
        "/v1/watchlist/check",
        json={"plate_text": "34-ABD-123", "country_code": "TR"},
    )
    assert wl_hit.status_code == 200
    assert wl_hit.json()["matched"] is True
    assert wl_hit.json()["matches"][0]["edit_distance"] == 1

    drift_ok = client.post(
        "/v1/monitoring/drift",
        json={
            "brightness": [120.0 + (i % 10) for i in range(30)],
            "contrast": [40.0 + (i % 8) for i in range(30)],
            "confidences": [0.88 + 0.002 * (i % 10) for i in range(30)],
        },
    )
    assert drift_ok.status_code == 200
    assert drift_ok.json()["drift_detected"] is False
    assert drift_ok.json()["status"] == "stable"

    drift_alert = client.post(
        "/v1/monitoring/drift",
        json={
            "brightness": [25.0 + (i % 5) for i in range(30)],
            "contrast": [40.0 + (i % 8) for i in range(30)],
            "confidences": [0.40 + 0.002 * (i % 10) for i in range(30)],
        },
    )
    assert drift_alert.status_code == 200
    assert drift_alert.json()["drift_detected"] is True
    assert drift_alert.json()["status"] == "drift_alert"


def test_serving_drift_endpoint_returns_503_when_unconfigured() -> None:
    app = create_app(config=ServingConfig(), drift_detector=None)
    client = AsgiTestClient(app)

    resp = client.post(
        "/v1/monitoring/drift",
        json={
            "brightness": [120.0],
            "contrast": [40.0],
            "confidences": [0.9],
        },
    )
    assert resp.status_code == 503
    assert resp.json()["error_code"] == "MODEL_LOAD_ERROR"
