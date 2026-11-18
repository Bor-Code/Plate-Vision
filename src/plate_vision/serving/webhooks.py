import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from plate_vision.utils.exceptions import ConfigurationError, DataIOError


class AlertEventType(StrEnum):
    WATCHLIST_HIT = "watchlist_hit"
    SPEEDING_VIOLATION = "speeding_violation"
    DRIFT_ALERT = "drift_alert"


class WebhookAlertEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(min_length=1)
    event_type: AlertEventType
    timestamp_epoch_ms: int = Field(ge=0)
    country_code: str = Field(min_length=2)
    plate_text: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)


class WebhookDeliveryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(min_length=1)
    endpoint_url: str = Field(min_length=1)
    signature: str = Field(min_length=1)
    delivered: bool
    attempts: int = Field(ge=1)
    status_code: int | None = None
    error_message: str | None = None


@runtime_checkable
class WebhookTransportProtocol(Protocol):
    def send(
        self,
        url: str,
        body: bytes,
        headers: Mapping[str, str],
    ) -> int: ...


class InMemoryWebhookTransport:
    def __init__(
        self,
        status_codes: Sequence[int] = (200,),
        *,
        fail_attempts: Sequence[int] = (),
    ) -> None:
        self._status_codes = list(status_codes) if status_codes else [200]
        self._fail_attempts = set(fail_attempts)
        self._call_index = 0
        self.sent_requests: list[tuple[str, bytes, dict[str, str]]] = []

    def send(
        self,
        url: str,
        body: bytes,
        headers: Mapping[str, str],
    ) -> int:
        attempt_no = self._call_index + 1
        self._call_index += 1
        self.sent_requests.append((url, body, dict(headers)))
        if attempt_no in self._fail_attempts:
            raise DataIOError(
                f"Simulated transport failure on attempt {attempt_no}",
                details={"url": url, "attempt": attempt_no},
            )
        idx = min(attempt_no - 1, len(self._status_codes) - 1)
        return self._status_codes[idx]


def compute_hmac_sha256_signature(secret: str, payload_bytes: bytes) -> str:
    digest = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def verify_hmac_sha256_signature(
    secret: str,
    payload_bytes: bytes,
    signature: str,
) -> bool:
    expected = compute_hmac_sha256_signature(secret, payload_bytes)
    return hmac.compare_digest(expected, signature)


class WebhookAlertDispatcher:
    def __init__(
        self,
        endpoint_url: str,
        signing_secret: str,
        transport: WebhookTransportProtocol | None = None,
        *,
        max_retries: int = 2,
    ) -> None:
        if not endpoint_url.strip():
            raise ConfigurationError("endpoint_url must not be empty")
        if not signing_secret.strip():
            raise ConfigurationError("signing_secret must not be empty")
        if max_retries < 0:
            raise ConfigurationError(
                "max_retries must be >= 0",
                details={"max_retries": max_retries},
            )
        self.endpoint_url = endpoint_url.strip()
        self.signing_secret = signing_secret
        self.transport: WebhookTransportProtocol = (
            transport if transport is not None else InMemoryWebhookTransport()
        )
        self.max_retries = max_retries
        self.dead_letter_queue: list[WebhookAlertEvent] = []
        self.delivery_history: list[WebhookDeliveryRecord] = []

    def dispatch(self, event: WebhookAlertEvent) -> WebhookDeliveryRecord:
        body = json.dumps(event.model_dump(mode="json"), sort_keys=True).encode("utf-8")
        signature = compute_hmac_sha256_signature(self.signing_secret, body)
        headers = {
            "Content-Type": "application/json",
            "X-PlateVision-Signature": signature,
            "X-PlateVision-Event": event.event_type.value,
        }

        max_attempts = self.max_retries + 1
        last_status: int | None = None
        last_error: str | None = None
        delivered = False
        attempts = 0

        for attempt in range(1, max_attempts + 1):
            attempts = attempt
            try:
                status_code = self.transport.send(self.endpoint_url, body, headers)
                last_status = status_code
                if 200 <= status_code < 300:
                    delivered = True
                    last_error = None
                    break
                last_error = f"Webhook returned HTTP {status_code}"
            except DataIOError as exc:
                last_status = None
                last_error = exc.message

        if not delivered:
            self.dead_letter_queue.append(event)

        record = WebhookDeliveryRecord(
            event_id=event.event_id,
            endpoint_url=self.endpoint_url,
            signature=signature,
            delivered=delivered,
            attempts=attempts,
            status_code=last_status,
            error_message=last_error,
        )
        self.delivery_history.append(record)
        return record


__all__ = [
    "AlertEventType",
    "InMemoryWebhookTransport",
    "WebhookAlertDispatcher",
    "WebhookAlertEvent",
    "WebhookDeliveryRecord",
    "WebhookTransportProtocol",
    "compute_hmac_sha256_signature",
    "verify_hmac_sha256_signature",
]
