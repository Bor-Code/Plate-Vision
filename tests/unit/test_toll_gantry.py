import pytest

from plate_vision.utils.exceptions import ConfigurationError, InvalidInputError
from plate_vision.video import (
    TollAccount,
    TollGantryEngine,
    TollTariffConfig,
    TollTransactionStatus,
    VehicleTollClass,
)


def test_toll_gantry_prepaid_peak_postpaid_exempt_violation_and_cooldown() -> None:
    engine = TollGantryEngine(
        tariff=TollTariffConfig(
            gantry_id="bridge_east_01",
            peak_hours_utc=(8,),
            peak_multiplier=1.5,
            violation_surcharge=100.0,
            min_pass_interval_seconds=60.0,
        )
    )

    acc_pre = engine.register_account(
        TollAccount(
            plate_text="34-ABC-123",
            country_code="TR",
            vehicle_class=VehicleTollClass.PASSENGER,
            balance=60.0,
        )
    )
    assert acc_pre.plate_text == "34ABC123"
    topped = engine.top_up_account("34 ABC 123", 40.0, country_code="TR")
    assert topped.balance == 100.0

    engine.register_account(
        TollAccount(
            plate_text="06TRK999",
            country_code="TR",
            vehicle_class=VehicleTollClass.TRUCK,
            balance=0.0,
            postpaid_enabled=True,
        )
    )
    engine.register_account(
        TollAccount(
            plate_text="35AMB112",
            country_code="TR",
            vehicle_class=VehicleTollClass.VAN,
            is_exempt=True,
        )
    )

    off_peak_ts = 10 * 3_600_000
    peak_ts = 8 * 3_600_000

    tx_pre = engine.process_passage("34 ABC 123", timestamp_ms=off_peak_ts)
    assert tx_pre.status == TollTransactionStatus.PAID_PREPAID
    assert tx_pre.is_peak_hour is False
    assert tx_pre.total_fee == 35.0
    assert tx_pre.remaining_balance == 65.0

    tx_dup = engine.process_passage("34ABC123", timestamp_ms=off_peak_ts + 15_000)
    assert tx_dup.status == TollTransactionStatus.DUPLICATE_IGNORED
    assert tx_dup.total_fee == 0.0
    assert tx_dup.remaining_balance == 65.0

    tx_peak_pre = engine.process_passage("34ABC123", timestamp_ms=off_peak_ts + 22 * 3_600_000)
    assert tx_peak_pre.status == TollTransactionStatus.PAID_PREPAID
    assert tx_peak_pre.is_peak_hour is True
    assert tx_peak_pre.total_fee == 52.5
    assert tx_peak_pre.remaining_balance == 12.5

    tx_insufficient = engine.process_passage("34ABC123", timestamp_ms=off_peak_ts + 24 * 3_600_000)
    assert tx_insufficient.status == TollTransactionStatus.UNPAID_VIOLATION
    assert tx_insufficient.total_fee == 135.0
    assert tx_insufficient.remaining_balance == 12.5

    tx_postpaid = engine.process_passage("06TRK999", timestamp_ms=peak_ts)
    assert tx_postpaid.status == TollTransactionStatus.BILLED_POSTPAID
    assert tx_postpaid.total_fee == 142.5

    tx_exempt = engine.process_passage("35AMB112", timestamp_ms=peak_ts)
    assert tx_exempt.status == TollTransactionStatus.EXEMPT
    assert tx_exempt.total_fee == 0.0

    tx_unreg = engine.process_passage(
        "16NEW777",
        timestamp_ms=off_peak_ts,
        vehicle_class=VehicleTollClass.ELECTRIC,
    )
    assert tx_unreg.status == TollTransactionStatus.UNPAID_VIOLATION
    assert tx_unreg.total_fee == 120.0
    assert tx_unreg.remaining_balance is None

    fetched = engine.get_account("34ABC123")
    assert fetched is not None
    assert fetched.balance == 12.5
    assert len(engine.transactions()) == 7

    summary = engine.summary()
    assert summary.gantry_id == "bridge_east_01"
    assert summary.total_passages == 7
    assert summary.paid_prepaid_count == 2
    assert summary.billed_postpaid_count == 1
    assert summary.exempt_count == 1
    assert summary.violation_count == 2
    assert summary.duplicate_ignored_count == 1
    assert summary.total_collected_revenue == 230.0
    assert summary.total_unpaid_amount == 255.0


def test_toll_gantry_validation_errors() -> None:
    with pytest.raises(ConfigurationError):
        TollGantryEngine(tariff=TollTariffConfig(peak_hours_utc=(25,)))

    with pytest.raises(ConfigurationError):
        TollGantryEngine(tariff=TollTariffConfig(base_rates={VehicleTollClass.PASSENGER: -10.0}))

    engine = TollGantryEngine()
    with pytest.raises(InvalidInputError):
        engine.process_passage("34ABC123", timestamp_ms=-1)

    with pytest.raises(InvalidInputError):
        engine.process_passage("A", timestamp_ms=1000)

    with pytest.raises(InvalidInputError):
        engine.top_up_account("34ABC123", 0.0)

    with pytest.raises(InvalidInputError):
        engine.top_up_account("34MISSING1", 50.0)
