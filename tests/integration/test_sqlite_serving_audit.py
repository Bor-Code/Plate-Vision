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
from plate_vision.utils.storage import SqlitePredictionStore


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
        return DecodedPlate(text="34DEF456", confidence=0.94, token_indices=(1,))


def test_serving_persists_predictions_to_sqlite_and_reloads_across_instances(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "serving_audit.db"
    store = SqlitePredictionStore(db_path=db_path, max_records=100)
    pipeline = PlateVisionPipeline(
        detector=_StubDetector(),
        recognizer=_StubRecognizer(),
        storage=store,
    )
    app = create_app(config=ServingConfig(), pipeline=pipeline, storage=store)
    client = AsgiTestClient(app)

    png_bytes = encode_grayscale_png([[130 for _ in range(120)] for _ in range(48)])
    b64_png = base64.b64encode(png_bytes).decode("ascii")

    resp = client.post(
        "/v1/predict",
        json={"image_base64": b64_png, "mime_type": "image/png"},
    )
    assert resp.status_code == 200
    assert resp.json()["plates"][0]["plate_text"] == "34DEF456"

    audit_resp = client.get("/v1/audit/recent?limit=10")
    assert audit_resp.status_code == 200
    assert audit_resp.json()["count"] == 1

    reopened_store = SqlitePredictionStore(db_path=db_path, max_records=100)
    persisted = reopened_store.list_records()
    assert len(persisted) == 1
    assert persisted[0].plate_texts == ("34DEF456",)
    assert reopened_store.query_by_country("TR")[0].country_code == "TR"
