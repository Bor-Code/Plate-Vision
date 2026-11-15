import pytest
from pydantic import ValidationError

from plate_vision.pipeline.watchlist import (
    WatchlistCategory,
    WatchlistEntry,
    WatchlistMatch,
)
from plate_vision.serving.schemas import (
    AuditLogResponse,
    DriftEvaluationRequest,
    WatchlistCheckRequest,
    WatchlistCheckResponse,
)
from plate_vision.utils.storage import build_audit_record


def test_serving_extended_schemas_validation() -> None:
    record = build_audit_record(
        b"sample-bytes",
        country_code="TR",
        plate_texts=("34ABC123",),
        confidences=(0.95,),
        latency_ms=2.5,
        cached=False,
    )
    audit_resp = AuditLogResponse(records=(record,), count=1)
    assert audit_resp.count == 1
    assert audit_resp.records[0].country_code == "TR"

    wl_req = WatchlistCheckRequest(plate_text="34ABC123", country_code="TR")
    assert wl_req.plate_text == "34ABC123"

    entry = WatchlistEntry(
        plate_text="34ABC123",
        country_code="TR",
        category=WatchlistCategory.VIP,
    )
    match = WatchlistMatch(
        entry=entry,
        matched_plate_text="34ABC123",
        edit_distance=0,
        exact_match=True,
    )
    wl_resp = WatchlistCheckResponse(matches=(match,), matched=True)
    assert wl_resp.matched is True

    drift_req = DriftEvaluationRequest(
        brightness=(120.0, 121.0),
        contrast=(40.0, 42.0),
        confidences=(0.9, 0.91),
    )
    assert len(drift_req.brightness) == 2

    with pytest.raises(ValidationError):
        DriftEvaluationRequest(brightness=(), contrast=(1.0,), confidences=(0.5,))
    with pytest.raises(ValidationError):
        WatchlistCheckRequest(plate_text="", country_code="TR")
