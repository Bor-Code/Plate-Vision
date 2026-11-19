import pytest

from plate_vision.utils.exceptions import InvalidInputError
from plate_vision.video.corridor import CorridorConfig, HighwayCorridorMonitor


def test_highway_corridor_validation_and_empty_summary() -> None:
    monitor = HighwayCorridorMonitor()
    empty = monitor.summarize()
    assert empty.corridor_id == "CORRIDOR_01"
    assert empty.total_passages == 0
    assert empty.active_in_corridor == 0
    assert empty.violation_count == 0
    assert empty.violation_rate == 0.0
    assert empty.mean_speed_kmh == 0.0
    assert empty.max_speed_kmh == 0.0
    assert empty.total_toll_revenue == 0.0
    assert empty.total_fine_revenue == 0.0

    with pytest.raises(InvalidInputError):
        monitor.record_entry("34ABC123", checkpoint_id="   ", timestamp_ms=1000)
    with pytest.raises(InvalidInputError):
        monitor.record_entry("34ABC123", checkpoint_id="CP1", timestamp_ms=-1)
    with pytest.raises(InvalidInputError):
        monitor.record_entry("A", checkpoint_id="CP1", timestamp_ms=1000)
    with pytest.raises(InvalidInputError):
        monitor.record_exit("34ABC123", checkpoint_id="CP2", timestamp_ms=5000)

    monitor.record_entry("34-ABC-123", checkpoint_id="CP1", timestamp_ms=10_000)
    with pytest.raises(InvalidInputError):
        monitor.record_exit("34ABC123", checkpoint_id="CP2", timestamp_ms=10_000)
    with pytest.raises(InvalidInputError):
        monitor.record_exit("34ABC123", checkpoint_id="CP2", timestamp_ms=9_000)


def test_highway_corridor_passages_tolerance_and_summary() -> None:
    config = CorridorConfig(
        corridor_id="O4_EAST",
        distance_km=12.0,
        speed_limit_kmh=100.0,
        tolerance_percent=10.0,
        toll_rate_per_km=3.0,
        speeding_fine_amount=1200.0,
        currency="TRY",
    )
    monitor = HighwayCorridorMonitor(config=config)

    obs1 = monitor.record_entry(
        "34 abc 123", country_code="tr", checkpoint_id="IN_1", timestamp_ms=0
    )
    assert obs1.plate_text == "34ABC123"
    assert obs1.country_code == "TR"
    assert len(monitor.active_entries()) == 1

    rec1 = monitor.record_exit(
        "34-ABC-123",
        country_code="TR",
        checkpoint_id="OUT_1",
        timestamp_ms=432_000,
    )
    assert rec1.passage_id == "pass_000001"
    assert rec1.travel_time_seconds == pytest.approx(432.0)
    assert rec1.average_speed_kmh == pytest.approx(100.0)
    assert rec1.effective_speed_limit_kmh == pytest.approx(110.0)
    assert rec1.is_speed_violation is False
    assert rec1.toll_amount == pytest.approx(36.0)
    assert rec1.fine_amount == 0.0
    assert rec1.total_charge == pytest.approx(36.0)

    monitor.record_entry("06XYZ99", country_code="TR", checkpoint_id="IN_1", timestamp_ms=10_000)
    rec2 = monitor.record_exit(
        "06-XYZ-99",
        country_code="TR",
        checkpoint_id="OUT_2",
        timestamp_ms=310_000,
    )
    assert rec2.passage_id == "pass_000002"
    assert rec2.average_speed_kmh == pytest.approx(144.0)
    assert rec2.is_speed_violation is True
    assert rec2.toll_amount == pytest.approx(36.0)
    assert rec2.fine_amount == pytest.approx(1200.0)
    assert rec2.total_charge == pytest.approx(1236.0)

    monitor.record_entry("35DEF45", country_code="TR", checkpoint_id="IN_2", timestamp_ms=50_000)
    assert len(monitor.active_entries()) == 1
    assert len(monitor.passages()) == 2

    summary = monitor.summarize()
    assert summary.corridor_id == "O4_EAST"
    assert summary.total_passages == 2
    assert summary.active_in_corridor == 1
    assert summary.violation_count == 1
    assert summary.violation_rate == pytest.approx(0.5)
    assert summary.mean_speed_kmh == pytest.approx(122.0)
    assert summary.max_speed_kmh == pytest.approx(144.0)
    assert summary.total_toll_revenue == pytest.approx(72.0)
    assert summary.total_fine_revenue == pytest.approx(1200.0)
    assert summary.currency == "TRY"
