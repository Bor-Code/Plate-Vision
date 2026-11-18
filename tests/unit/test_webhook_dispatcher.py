import pytest

from plate_vision.serving.webhooks import (
    AlertEventType,
    InMemoryWebhookTransport,
    WebhookAlertDispatcher,
    WebhookAlertEvent,
    WebhookTransportProtocol,
    compute_hmac_sha256_signature,
    verify_hmac_sha256_signature,
)
from plate_vision.utils.exceptions import ConfigurationError


def test_webhook_dispatcher_validation_and_signature() -> None:
    with pytest.raises(ConfigurationError):
        WebhookAlertDispatcher("", "secret")
    with pytest.raises(ConfigurationError):
        WebhookAlertDispatcher("https://example.com/hook", "")
    with pytest.raises(ConfigurationError):
        WebhookAlertDispatcher("https://example.com/hook", "secret", max_retries=-1)

    sig = compute_hmac_sha256_signature("my-secret", b"hello")
    assert sig.startswith("sha256=")
    assert verify_hmac_sha256_signature("my-secret", b"hello", sig) is True
    assert verify_hmac_sha256_signature("wrong", b"hello", sig) is False


def test_webhook_dispatcher_retries_and_dead_letter_queue() -> None:
    transport = InMemoryWebhookTransport(status_codes=(503, 202), fail_attempts=(1,))
    assert isinstance(transport, WebhookTransportProtocol)

    dispatcher = WebhookAlertDispatcher(
        "https://alerts.internal/hooks/alpr",
        "super-secret-key",
        transport=transport,
        max_retries=2,
    )
    event = WebhookAlertEvent(
        event_id="evt-001",
        event_type=AlertEventType.WATCHLIST_HIT,
        timestamp_epoch_ms=1700000000000,
        country_code="TR",
        plate_text="34ABC123",
        payload={"category": "stolen"},
    )

    rec = dispatcher.dispatch(event)
    assert rec.delivered is True
    assert rec.attempts == 2
    assert rec.status_code == 202
    assert dispatcher.dead_letter_queue == []

    _, body, headers = transport.sent_requests[-1]
    assert verify_hmac_sha256_signature(
        "super-secret-key",
        body,
        headers["X-PlateVision-Signature"],
    )

    failing_transport = InMemoryWebhookTransport(status_codes=(500,))
    failing_dispatcher = WebhookAlertDispatcher(
        "https://alerts.internal/hooks/alpr",
        "super-secret-key",
        transport=failing_transport,
        max_retries=1,
    )
    fail_rec = failing_dispatcher.dispatch(event)
    assert fail_rec.delivered is False
    assert fail_rec.attempts == 2
    assert fail_rec.status_code == 500
    assert len(failing_dispatcher.dead_letter_queue) == 1
