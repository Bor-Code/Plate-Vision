from plate_vision.serving import AsgiTestClient, create_app
from plate_vision.video import TollGantryEngine, TollTariffConfig


def test_serving_toll_gantry_endpoints_end_to_end() -> None:
    toll_engine = TollGantryEngine(
        tariff=TollTariffConfig(
            gantry_id="bosphorus_south",
            min_pass_interval_seconds=30.0,
            violation_surcharge=150.0,
        )
    )
    app = create_app(toll_engine=toll_engine)
    client = AsgiTestClient(app)

    res_reg = client.post(
        "/v1/toll/account",
        json={
            "plate_text": "34-ABC-123",
            "country_code": "TR",
            "vehicle_class": "passenger",
            "balance": 60.0,
        },
    )
    assert res_reg.status_code == 200
    assert res_reg.json()["plate_text"] == "34ABC123"
    assert res_reg.json()["balance"] == 60.0

    res_top = client.post(
        "/v1/toll/top-up",
        json={"plate_text": "34 ABC 123", "country_code": "TR", "amount": 40.0},
    )
    assert res_top.status_code == 200
    assert res_top.json()["balance"] == 100.0

    res_unknown_top = client.post(
        "/v1/toll/top-up",
        json={"plate_text": "06ZZZ999", "country_code": "TR", "amount": 25.0},
    )
    assert res_unknown_top.status_code == 422

    base_ts = 11 * 3_600_000
    res_pass = client.post(
        "/v1/toll/passage",
        json={"plate_text": "34ABC123", "country_code": "TR", "timestamp_ms": base_ts},
    )
    assert res_pass.status_code == 200
    assert res_pass.json()["status"] == "paid_prepaid"
    assert res_pass.json()["total_fee"] == 35.0
    assert res_pass.json()["remaining_balance"] == 65.0

    res_dup = client.post(
        "/v1/toll/passage",
        json={
            "plate_text": "34ABC123",
            "country_code": "TR",
            "timestamp_ms": base_ts + 5_000,
        },
    )
    assert res_dup.status_code == 200
    assert res_dup.json()["status"] == "duplicate_ignored"

    res_viol = client.post(
        "/v1/toll/passage",
        json={
            "plate_text": "06UNR555",
            "country_code": "TR",
            "timestamp_ms": base_ts + 40_000,
        },
    )
    assert res_viol.status_code == 200
    assert res_viol.json()["status"] == "unpaid_violation"
    assert res_viol.json()["total_fee"] == 185.0

    res_sum = client.get("/v1/toll/summary")
    assert res_sum.status_code == 200
    summary = res_sum.json()
    assert summary["gantry_id"] == "bosphorus_south"
    assert summary["total_passages"] == 3
    assert summary["paid_prepaid_count"] == 1
    assert summary["duplicate_ignored_count"] == 1
    assert summary["violation_count"] == 1
    assert summary["total_collected_revenue"] == 35.0
    assert summary["total_unpaid_amount"] == 185.0
