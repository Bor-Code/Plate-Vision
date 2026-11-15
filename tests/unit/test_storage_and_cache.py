from pathlib import Path

import pytest

from plate_vision.utils.exceptions import ConfigurationError
from plate_vision.utils.storage import (
    InMemoryPredictionStore,
    LocalJsonlPredictionStore,
    PredictionCache,
    build_audit_record,
)


def test_prediction_cache_lru_and_metrics() -> None:
    with pytest.raises(ConfigurationError):
        PredictionCache[str](max_entries=0)

    cache = PredictionCache[str](max_entries=2)
    assert cache.get("k1") is None
    assert cache.misses == 1

    cache.put("k1", "v1")
    cache.put("k2", "v2")
    assert cache.size == 2
    assert cache.get("k1") == "v1"
    assert cache.hits == 1

    cache.put("k3", "v3")
    assert cache.get("k2") is None
    assert cache.get("k1") == "v1"
    assert cache.get("k3") == "v3"

    cache.clear()
    assert cache.size == 0
    assert cache.hits == 0
    assert cache.misses == 0


def test_in_memory_and_local_jsonl_prediction_stores(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        InMemoryPredictionStore(max_records=0)

    mem_store = InMemoryPredictionStore(max_records=2)
    rec1 = build_audit_record(
        b"payload-1",
        country_code="TR",
        plate_texts=("34ABC123",),
        confidences=(0.91,),
        latency_ms=4.2,
        cached=False,
    )
    rec2 = build_audit_record(
        b"payload-2",
        country_code="DE",
        plate_texts=("BAB1234",),
        confidences=(0.88,),
        latency_ms=3.1,
        cached=False,
    )
    rec3 = build_audit_record(
        b"payload-3",
        country_code="GB",
        plate_texts=("AB51ABC",),
        confidences=(0.95,),
        latency_ms=0.5,
        cached=True,
    )
    mem_store.save(rec1)
    mem_store.save(rec2)
    mem_store.save(rec3)
    assert len(mem_store.list_records()) == 2
    assert mem_store.list_records(limit=1)[0].country_code == "GB"
    assert mem_store.list_records(limit=0) == ()

    jsonl_path = tmp_path / "audit" / "predictions.jsonl"
    jsonl_store = LocalJsonlPredictionStore(jsonl_path)
    assert jsonl_store.list_records() == ()
    jsonl_store.save(rec1)
    jsonl_store.save(rec2)
    loaded = jsonl_store.list_records()
    assert len(loaded) == 2
    assert loaded[0].plate_texts == ("34ABC123",)
    assert len(jsonl_store.list_records(limit=1)) == 1
    assert jsonl_store.list_records(limit=0) == ()
