import pytest

from plate_vision.serving.security import (
    SlidingWindowRateLimiter,
    verify_api_key,
)


def test_sliding_window_rate_limiter_allows_and_throttles() -> None:
    limiter = SlidingWindowRateLimiter(max_requests_per_window=2, window_seconds=10.0)
    d1 = limiter.check("client-a", now=100.0)
    assert d1.allowed is True
    assert d1.remaining == 1

    d2 = limiter.check("client-a", now=102.0)
    assert d2.allowed is True
    assert d2.remaining == 0

    d3 = limiter.check("client-a", now=105.0)
    assert d3.allowed is False
    assert d3.remaining == 0
    assert d3.reset_after_seconds == pytest.approx(5.0)

    d4 = limiter.check("client-a", now=111.0)
    assert d4.allowed is True
    assert d4.remaining == 0


def test_sliding_window_rate_limiter_validation() -> None:
    with pytest.raises(ValueError, match="max_requests_per_window"):
        SlidingWindowRateLimiter(max_requests_per_window=0)
    with pytest.raises(ValueError, match="window_seconds"):
        SlidingWindowRateLimiter(window_seconds=0.0)


def test_verify_api_key_behavior() -> None:
    assert verify_api_key(None, ()) is True
    assert verify_api_key("any-key", ()) is True
    assert verify_api_key(None, ("secret-1",)) is False
    assert verify_api_key("wrong-key", ("secret-1", "secret-2")) is False
    assert verify_api_key("secret-2", ("secret-1", "secret-2")) is True
