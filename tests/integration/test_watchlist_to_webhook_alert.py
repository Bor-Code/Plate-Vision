import json
from collections.abc import Sequence

from plate_vision.data.models import BoundingBox
from plate_vision.data.synthetic import encode_grayscale_png
from plate_vision.detection.models import DetectionPrediction
from plate_vision.pipeline.runner import PlateVisionPipeline
from plate_vision.pipeline.watchlist import (
    WatchlistCategory,
    WatchlistEntry,
    WatchlistMatcher,
)
from plate_vision.recognition.decoding import DecodedPlate
from plate_vision.serving.webhooks import (
    AlertEventType,
    InMemoryWebhookTransport,
    WebhookAlertDispatcher,
    WebhookAlertEvent,
    verify_hmac_sha256_signature,
)


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
                bbox=BoundingBox(x_min=10.0, y_min=8.0, x_max=96.0, y_max=36.0),
                confidence=0.96,
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
        return DecodedPlate(text="34 A8C 123", confidence=0.94, token_indices=(1,))


def test_watchlist_detection_dispatches_signed_webhook_alert() -> None:
    pipeline = PlateVisionPipeline(
        detector=_StubDetector(),
        recognizer=_StubRecognizer(),
    )
    png_bytes = encode_grayscale_png([[135 for _ in range(120)] for _ in range(48)])
    result = pipeline.predict_bytes(png_bytes)

    matcher = WatchlistMatcher(
        entries=[
            WatchlistEntry(
                plate_text="34ABC123",
                country_code="TR",
                category=WatchlistCategory.STOLEN,
                note="Dispatch patrol unit",
            )
        ],
        max_edit_distance=0,
    )
    transport = InMemoryWebhookTransport(status_codes=(200,))
    dispatcher = WebhookAlertDispatcher(
        "https://soc.internal/alpr-webhook",
        "webhook-signing-key",
        transport=transport,
    )

    for idx, plate in enumerate(result.plates):
        matches = matcher.check(plate.plate_text, country_code=plate.country_code)
        if matches:
            event = WebhookAlertEvent(
                event_id=f"alert-{idx}",
                event_type=AlertEventType.WATCHLIST_HIT,
                timestamp_epoch_ms=1700000100000,
                country_code=plate.country_code,
                plate_text=plate.plate_text,
                payload={"category": matches[0].entry.category.value},
            )
            delivery = dispatcher.dispatch(event)
            assert delivery.delivered is True

    assert len(transport.sent_requests) == 1
    _, raw_body, req_headers = transport.sent_requests[0]
    assert verify_hmac_sha256_signature(
        "webhook-signing-key",
        raw_body,
        req_headers["X-PlateVision-Signature"],
    )
    parsed = json.loads(raw_body.decode("utf-8"))
    assert parsed["plate_text"] == "34ABC123"
    assert parsed["payload"]["category"] == "stolen"
