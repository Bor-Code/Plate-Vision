from pathlib import Path

import pytest

from plate_vision.utils.exceptions import ConfigurationError
from plate_vision.utils.storage import (
    PredictionStorageProtocol,
    SqlitePredictionStore,
    build_audit_record,
)


def test_sqlite_prediction_store_crud_retention_and_country_query(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        SqlitePredictionStore(tmp_path / "invalid.db", max_records=0)

    db_path = tmp_path / "audit" / "predictions.db"
    store = SqlitePredictionStore(db_path=db_path, max_records=3)
    assert isinstance(store, PredictionStorageProtocol)
    assert store.count() == 0
    assert store.list_records() == ()

    r1 = build_audit_record(
        b"img-1",
        country_code="TR",
        plate_texts=("34ABC123",),
        confidences=(0.91,),
        latency_ms=4.1,
        cached=False,
    )
    r2 = build_audit_record(
        b"img-2",
        country_code="DE",
        plate_texts=("BAB1234",),
        confidences=(0.88,),
        latency_ms=3.2,
        cached=False,
    )
    r3 = build_audit_record(
        b"img-3",
        country_code="TR",
        plate_texts=("06XYZ999",),
        confidences=(0.95,),
        latency_ms=2.0,
        cached=True,
    )
    r4 = build_audit_record(
        b"img-4",
        country_code="FR",
        plate_texts=("AB123CD",),
        confidences=(0.93,),
        latency_ms=1.8,
        cached=False,
    )

    for rec in (r1, r2, r3, r4):
        store.save(rec)

    assert store.count() == 3
    all_records = store.list_records()
    assert len(all_records) == 3
    assert [r.country_code for r in all_records] == ["DE", "TR", "FR"]

    recent_two = store.list_records(limit=2)
    assert [r.plate_texts[0] for r in recent_two] == ["06XYZ999", "AB123CD"]
    assert store.list_records(limit=0) == ()

    tr_records = store.query_by_country("tr", limit=10)
    assert len(tr_records) == 1
    assert tr_records[0].plate_texts == ("06XYZ999",)
    assert store.query_by_country("TR", limit=0) == ()
