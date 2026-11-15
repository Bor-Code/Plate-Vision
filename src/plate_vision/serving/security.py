import secrets
import time
from collections import deque

from pydantic import BaseModel, ConfigDict, Field


class RateLimitDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed: bool
    remaining: int = Field(ge=0)
    reset_after_seconds: float = Field(ge=0.0)


class SlidingWindowRateLimiter:
    def __init__(
        self,
        max_requests_per_window: int = 600,
        window_seconds: float = 60.0,
    ) -> None:
        if max_requests_per_window < 1:
            msg = "max_requests_per_window must be >= 1"
            raise ValueError(msg)
        if window_seconds <= 0.0:
            msg = "window_seconds must be > 0.0"
            raise ValueError(msg)
        self.max_requests_per_window = max_requests_per_window
        self.window_seconds = window_seconds
        self._history: dict[str, deque[float]] = {}

    def check(
        self,
        client_key: str,
        now: float | None = None,
    ) -> RateLimitDecision:
        current_time = time.monotonic() if now is None else now
        cutoff = current_time - self.window_seconds
        bucket = self._history.get(client_key)
        if bucket is None:
            bucket = deque()
            self._history[client_key] = bucket

        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

        if len(bucket) >= self.max_requests_per_window:
            oldest = bucket[0]
            reset_after = max(0.0, self.window_seconds - (current_time - oldest))
            return RateLimitDecision(
                allowed=False,
                remaining=0,
                reset_after_seconds=round(reset_after, 4),
            )

        bucket.append(current_time)
        oldest = bucket[0]
        reset_after = max(0.0, self.window_seconds - (current_time - oldest))
        remaining = max(0, self.max_requests_per_window - len(bucket))
        return RateLimitDecision(
            allowed=True,
            remaining=remaining,
            reset_after_seconds=round(reset_after, 4),
        )


def verify_api_key(
    provided_key: str | None,
    configured_keys: tuple[str, ...],
) -> bool:
    if not configured_keys:
        return True
    if not provided_key:
        return False
    return any(secrets.compare_digest(provided_key, valid_key) for valid_key in configured_keys)


__all__ = [
    "RateLimitDecision",
    "SlidingWindowRateLimiter",
    "verify_api_key",
]
