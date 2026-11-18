import pytest

from plate_vision.pipeline.watchlist import (
    WatchlistCategory,
    WatchlistEntry,
    WatchlistMatcher,
)
from plate_vision.utils.exceptions import ConfigurationError, InvalidInputError
from plate_vision.video.parking import (
    GateDirection,
    ParkingLotManager,
    ParkingTariffConfig,
)


def test_parking_lot_manager_validation_and_capacity() -> None:
    assert GateDirection.ENTRY.value == "entry"
    assert GateDirection.EXIT.value == "exit"

    with pytest.raises(ConfigurationError):
        ParkingLotManager(capacity=0)

    manager = ParkingLotManager(capacity=1)
    with pytest.raises(InvalidInputError):
        manager.record_entry("34ABC123", timestamp_ms=-1)
    with pytest.raises(InvalidInputError):
        manager.record_entry("A", timestamp_ms=1000)
    with pytest.raises(InvalidInputError):
        manager.record_exit("34ABC123", timestamp_ms=-5)

    d1 = manager.record_entry("34ABC123", timestamp_ms=10_000)
    assert d1.open_barrier is True
    assert manager.occupancy_count == 1
    assert len(manager.active_sessions()) == 1

    dup = manager.record_entry("34-ABC-123", timestamp_ms=12_000)
    assert dup.open_barrier is False
    assert dup.reason == "already_parked"

    full = manager.record_entry("06XYZ999", timestamp_ms=15_000)
    assert full.open_barrier is False
    assert full.reason == "lot_full"

    with pytest.raises(InvalidInputError):
        manager.record_exit("34ABC123", timestamp_ms=5_000)

    unknown = manager.record_exit("99ZZ999", timestamp_ms=20_000)
    assert unknown.open_barrier is False
    assert unknown.reason == "no_active_session"


def test_parking_lot_manager_billing_grace_vip_and_blocklist() -> None:
    watchlist = WatchlistMatcher(
        entries=[
            WatchlistEntry(
                plate_text="34VIP001", country_code="TR", category=WatchlistCategory.VIP
            ),
            WatchlistEntry(
                plate_text="34STL002", country_code="TR", category=WatchlistCategory.STOLEN
            ),
        ]
    )
    tariff = ParkingTariffConfig(
        grace_period_minutes=15.0,
        hourly_rate=60.0,
        daily_max_rate=200.0,
        vip_free_parking=True,
    )
    manager = ParkingLotManager(tariff=tariff, watchlist=watchlist, capacity=10)

    blocked = manager.record_entry("34STL002", timestamp_ms=0)
    assert blocked.open_barrier is False
    assert blocked.reason == "watchlist_blocked"

    vip_in = manager.record_entry("34VIP001", timestamp_ms=0)
    assert vip_in.open_barrier is True
    vip_out = manager.record_exit("34VIP001", timestamp_ms=120 * 60_000)
    assert vip_out.open_barrier is True
    assert vip_out.session is not None
    assert vip_out.session.fee_amount == 0.0

    grace_in = manager.record_entry("34GRC003", timestamp_ms=0)
    assert grace_in.open_barrier is True
    grace_out = manager.record_exit("34GRC003", timestamp_ms=10 * 60_000)
    assert grace_out.session is not None
    assert grace_out.session.fee_amount == 0.0

    paid_in = manager.record_entry("34PAY004", timestamp_ms=0)
    assert paid_in.open_barrier is True
    paid_out = manager.record_exit("34PAY004", timestamp_ms=150 * 60_000)
    assert paid_out.session is not None
    assert paid_out.session.fee_amount == pytest.approx(180.0)
    assert manager.total_revenue() == pytest.approx(180.0)
    assert len(manager.closed_sessions()) == 3
