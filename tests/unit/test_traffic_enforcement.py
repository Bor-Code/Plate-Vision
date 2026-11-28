import pytest

from plate_vision.utils.exceptions import ConfigurationError, InvalidInputError
from plate_vision.video import (
    EnforcementZoneRule,
    SignalPhase,
    TrafficEnforcementEngine,
    ViolationObservation,
    ViolationType,
)


def test_traffic_enforcement_red_light_bus_lane_exempt_and_cooldown() -> None:
    engine = TrafficEnforcementEngine(
        rules=[
            EnforcementZoneRule(
                zone_id="junction_01",
                violation_type=ViolationType.RED_LIGHT,
                fine_amount=1500.0,
                red_light_grace_ms=500,
                cooldown_seconds=60.0,
                min_confidence=0.80,
            ),
            EnforcementZoneRule(
                zone_id="bus_lane_02",
                violation_type=ViolationType.BUS_LANE,
                fine_amount=900.0,
                cooldown_seconds=120.0,
                min_confidence=0.75,
            ),
        ]
    )
    assert engine.add_exempt_plate("34 BUS 001", zone_id="bus_lane_02") == "34BUS001"
    assert engine.add_exempt_plate("34 AMB 112") == "34AMB112"

    t0 = 1_700_000_000_000
    assert (
        engine.evaluate(
            ViolationObservation(
                zone_id="junction_01",
                plate_text="34ABC123",
                timestamp_ms=t0,
                confidence=0.92,
                signal_phase=SignalPhase.GREEN,
                signal_elapsed_ms=2000,
            )
        )
        is None
    )
    assert (
        engine.evaluate(
            ViolationObservation(
                zone_id="junction_01",
                plate_text="34ABC123",
                timestamp_ms=t0 + 1000,
                confidence=0.92,
                signal_phase=SignalPhase.RED,
                signal_elapsed_ms=300,
            )
        )
        is None
    )
    assert (
        engine.evaluate(
            ViolationObservation(
                zone_id="junction_01",
                plate_text="34LOW001",
                timestamp_ms=t0 + 2000,
                confidence=0.50,
                signal_phase=SignalPhase.RED,
                signal_elapsed_ms=1200,
            )
        )
        is None
    )

    ticket_red = engine.evaluate(
        ViolationObservation(
            zone_id="junction_01",
            plate_text="34-ABC-123",
            timestamp_ms=t0 + 2500,
            confidence=0.95,
            frame_indices=(12, 13),
            signal_phase=SignalPhase.RED,
            signal_elapsed_ms=1200,
            speed_kmh=58.4,
        )
    )
    assert ticket_red is not None
    assert ticket_red.plate_text == "34ABC123"
    assert ticket_red.violation_type == ViolationType.RED_LIGHT
    assert ticket_red.fine_amount == 1500.0
    assert ticket_red.speed_kmh == 58.4

    assert (
        engine.evaluate(
            ViolationObservation(
                zone_id="junction_01",
                plate_text="34ABC123",
                timestamp_ms=t0 + 10_000,
                confidence=0.95,
                signal_phase=SignalPhase.RED,
                signal_elapsed_ms=3000,
            )
        )
        is None
    )

    assert (
        engine.evaluate(
            ViolationObservation(
                zone_id="bus_lane_02",
                plate_text="34BUS001",
                timestamp_ms=t0 + 12_000,
                confidence=0.90,
            )
        )
        is None
    )
    assert (
        engine.evaluate(
            ViolationObservation(
                zone_id="junction_01",
                plate_text="34AMB112",
                timestamp_ms=t0 + 14_000,
                confidence=0.93,
                signal_phase=SignalPhase.RED,
                signal_elapsed_ms=2000,
            )
        )
        is None
    )

    ticket_bus = engine.evaluate(
        ViolationObservation(
            zone_id="bus_lane_02",
            plate_text="06CAR777",
            timestamp_ms=t0 + 16_000,
            confidence=0.88,
        )
    )
    assert ticket_bus is not None
    assert ticket_bus.fine_amount == 900.0

    assert len(engine.tickets()) == 2
    summary = engine.summary()
    assert summary.total_observations == 8
    assert summary.tickets_issued == 2
    assert summary.exempt_skipped == 2
    assert summary.grace_or_compliant_skipped == 2
    assert summary.low_confidence_skipped == 1
    assert summary.cooldown_skipped == 1
    assert summary.total_fine_amount == 2400.0
    assert summary.by_violation_type == {"red_light": 1, "bus_lane": 1}


def test_traffic_enforcement_validation_errors() -> None:
    with pytest.raises(ConfigurationError):
        TrafficEnforcementEngine(currency="INVALID")

    rule = EnforcementZoneRule(
        zone_id="zone_a",
        violation_type=ViolationType.WRONG_WAY,
        fine_amount=2000.0,
    )
    with pytest.raises(ConfigurationError):
        TrafficEnforcementEngine(rules=[rule, rule])

    engine = TrafficEnforcementEngine(rules=[rule])
    with pytest.raises(ConfigurationError):
        engine.evaluate(
            ViolationObservation(
                zone_id="missing_zone",
                plate_text="34ABC123",
                timestamp_ms=1000,
                confidence=0.9,
            )
        )

    with pytest.raises(InvalidInputError):
        engine.evaluate(
            ViolationObservation(
                zone_id="zone_a",
                plate_text="--",
                timestamp_ms=1000,
                confidence=0.9,
            )
        )
