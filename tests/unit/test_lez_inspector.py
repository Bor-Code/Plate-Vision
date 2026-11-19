import pytest

from plate_vision.utils.exceptions import InvalidInputError
from plate_vision.video.lez import (
    EmissionClass,
    LowEmissionZoneConfig,
    LowEmissionZoneInspector,
    VehicleEmissionProfile,
)


def test_lez_inspector_validation_and_empty_summary() -> None:
    inspector = LowEmissionZoneInspector()
    empty = inspector.summarize()
    assert empty.zone_id == "LEZ_CENTER_01"
    assert empty.total_inspections == 0
    assert empty.compliant_count == 0
    assert empty.violation_count == 0
    assert empty.compliance_rate == 0.0
    assert empty.total_fine_amount == 0.0
    assert empty.violations_by_reason == {}

    with pytest.raises(InvalidInputError):
        inspector.inspect("34ABC123", day_of_month=0)
    with pytest.raises(InvalidInputError):
        inspector.inspect("34ABC123", day_of_month=32)
    with pytest.raises(InvalidInputError):
        inspector.inspect("X", day_of_month=10)


def test_lez_inspector_emission_exempt_odd_even_and_summary() -> None:
    config = LowEmissionZoneConfig(
        zone_id="HISTORIC_CORE",
        allowed_classes=(EmissionClass.ZERO_EMISSION, EmissionClass.EURO_6),
        enforce_odd_even_days=True,
        non_compliant_fine_amount=1800.0,
        unregistered_fine_amount=2200.0,
        odd_even_fine_amount=900.0,
        currency="TRY",
    )
    inspector = LowEmissionZoneInspector(
        config=config,
        registry=[
            VehicleEmissionProfile(
                plate_text="34 EV 101",
                country_code="TR",
                emission_class=EmissionClass.ZERO_EMISSION,
            ),
            VehicleEmissionProfile(
                plate_text="34 AMB 102",
                country_code="TR",
                emission_class=EmissionClass.EURO_4,
                is_exempt=True,
            ),
            VehicleEmissionProfile(
                plate_text="34 E6 204",
                country_code="TR",
                emission_class=EmissionClass.EURO_6,
            ),
            VehicleEmissionProfile(
                plate_text="34 OLD 305",
                country_code="TR",
                emission_class=EmissionClass.EURO_4,
            ),
        ],
    )
    inspector.register_vehicle(
        VehicleEmissionProfile(
            plate_text="ABCDEF",
            country_code="DE",
            emission_class=EmissionClass.EURO_6,
        )
    )

    r_ev = inspector.inspect("34EV101", day_of_month=2)
    assert r_ev.inspection_id == "lez_000001"
    assert r_ev.is_compliant is True
    assert r_ev.fine_amount == 0.0

    r_exempt = inspector.inspect("34-AMB-102", day_of_month=15)
    assert r_exempt.is_compliant is True
    assert r_exempt.is_exempt is True
    assert r_exempt.fine_amount == 0.0

    r_even_ok = inspector.inspect("34E6204", day_of_month=12)
    assert r_even_ok.is_compliant is True

    r_odd_fail = inspector.inspect("34E6204", day_of_month=13)
    assert r_odd_fail.is_compliant is False
    assert r_odd_fail.violation_reason == "odd_even_day_restriction"
    assert r_odd_fail.fine_amount == pytest.approx(900.0)

    r_old = inspector.inspect("34OLD305", day_of_month=13)
    assert r_old.is_compliant is False
    assert r_old.violation_reason == "restricted_emission_class"
    assert r_old.fine_amount == pytest.approx(1800.0)

    r_unreg = inspector.inspect("06NEW999", day_of_month=13)
    assert r_unreg.is_compliant is False
    assert r_unreg.violation_reason == "unregistered_vehicle"
    assert r_unreg.fine_amount == pytest.approx(2200.0)

    r_no_digits = inspector.inspect("ABCDEF", country_code="DE", day_of_month=13)
    assert r_no_digits.is_compliant is True

    assert len(inspector.history()) == 7
    summary = inspector.summarize()
    assert summary.zone_id == "HISTORIC_CORE"
    assert summary.total_inspections == 7
    assert summary.compliant_count == 4
    assert summary.violation_count == 3
    assert summary.total_fine_amount == pytest.approx(4900.0)
    assert summary.violations_by_reason == {
        "odd_even_day_restriction": 1,
        "restricted_emission_class": 1,
        "unregistered_vehicle": 1,
    }
