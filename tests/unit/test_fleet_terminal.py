import pytest

from plate_vision.utils.exceptions import InvalidInputError
from plate_vision.video.fleet import (
    FleetSlaConfig,
    FleetTerminalController,
    TerminalZoneType,
)


def test_fleet_terminal_validation_and_empty_summary() -> None:
    controller = FleetTerminalController()
    empty = controller.summarize()
    assert empty.facility_id == "LOGISTICS_HUB_01"
    assert empty.active_vehicles == 0
    assert empty.completed_visits == 0
    assert empty.sla_compliance_rate == 0.0
    assert empty.visits_by_carrier == {}

    with pytest.raises(InvalidInputError):
        controller.check_in("34ABC123", timestamp_ms=-1)
    with pytest.raises(InvalidInputError):
        controller.check_in("34ABC123", carrier_name="   ", timestamp_ms=0)
    with pytest.raises(InvalidInputError):
        controller.check_in("A", timestamp_ms=0)
    with pytest.raises(InvalidInputError):
        controller.check_out("34ABC123", timestamp_ms=-1)
    with pytest.raises(InvalidInputError):
        controller.check_out("34ABC123", timestamp_ms=1_000)

    controller.check_in("34TRK01", carrier_name="ALPHA", timestamp_ms=10_000)
    with pytest.raises(InvalidInputError):
        controller.check_in("34-TRK-01", carrier_name="ALPHA", timestamp_ms=11_000)
    with pytest.raises(InvalidInputError):
        controller.check_out("34TRK01", timestamp_ms=5_000)


def test_fleet_terminal_sla_demurrage_and_summary() -> None:
    controller = FleetTerminalController(
        config=FleetSlaConfig(
            facility_id="GEBZE_HUB",
            max_turnaround_minutes=60.0,
            demurrage_rate_per_hour=300.0,
            currency="TRY",
        )
    )

    obs1 = controller.check_in(
        "34 log 101",
        country_code="tr",
        carrier_name="MARMARA_FREIGHT",
        assigned_zone=TerminalZoneType.LOADING_DOCK,
        timestamp_ms=0,
    )
    assert obs1.plate_text == "34LOG101"
    assert len(controller.active_vehicles()) == 1

    rec1 = controller.check_out(
        "34-LOG-101",
        country_code="TR",
        timestamp_ms=45 * 60_000,
    )
    assert rec1.visit_id == "visit_000001"
    assert rec1.turnaround_minutes == pytest.approx(45.0)
    assert rec1.sla_breached is False
    assert rec1.overstay_minutes == 0.0
    assert rec1.demurrage_fee == 0.0

    controller.check_in(
        "06CAR202",
        country_code="TR",
        carrier_name="ANADOLU_LOG",
        assigned_zone=TerminalZoneType.WEIGHBRIDGE,
        timestamp_ms=10_000,
    )
    rec2 = controller.check_out(
        "06CAR202",
        country_code="TR",
        timestamp_ms=10_000 + (90 * 60_000),
    )
    assert rec2.visit_id == "visit_000002"
    assert rec2.turnaround_minutes == pytest.approx(90.0)
    assert rec2.sla_breached is True
    assert rec2.overstay_minutes == pytest.approx(30.0)
    assert rec2.demurrage_fee == pytest.approx(150.0)

    controller.check_in(
        "35STG303",
        country_code="TR",
        carrier_name="MARMARA_FREIGHT",
        assigned_zone=TerminalZoneType.YARD,
        timestamp_ms=20_000,
    )
    assert len(controller.active_vehicles()) == 1
    assert len(controller.completed_records()) == 2

    summary = controller.summarize()
    assert summary.facility_id == "GEBZE_HUB"
    assert summary.active_vehicles == 1
    assert summary.completed_visits == 2
    assert summary.sla_breach_count == 1
    assert summary.sla_compliance_rate == pytest.approx(0.5)
    assert summary.mean_turnaround_minutes == pytest.approx(67.5)
    assert summary.total_demurrage_fee == pytest.approx(150.0)
    assert summary.visits_by_carrier == {
        "ANADOLU_LOG": 1,
        "MARMARA_FREIGHT": 1,
    }
