import pytest

from plate_vision.config.schema import ServingConfig
from plate_vision.pipeline.watchlist import (
    WatchlistCategory,
    WatchlistEntry,
    WatchlistMatcher,
)
from plate_vision.serving.app import AsgiTestClient, create_app
from plate_vision.video.corridor import CorridorConfig, HighwayCorridorMonitor
from plate_vision.video.parking import ParkingLotManager, ParkingTariffConfig


def test_serving_parking_and_corridor_api_workflows() -> None:
    watchlist = WatchlistMatcher(
        entries=[
            WatchlistEntry(
                plate_text="34BLK001",
                country_code="TR",
                category=WatchlistCategory.BLOCKLIST,
            ),
            WatchlistEntry(
                plate_text="34VIP777",
                country_code="TR",
                category=WatchlistCategory.VIP,
            ),
        ]
    )
    parking = ParkingLotManager(
        tariff=ParkingTariffConfig(
            grace_period_minutes=15.0,
            hourly_rate=50.0,
            daily_max_rate=400.0,
            vip_free_parking=True,
        ),
        watchlist=watchlist,
        capacity=10,
    )
    corridor = HighwayCorridorMonitor(
        config=CorridorConfig(
            corridor_id="TEM_WEST",
            distance_km=12.0,
            speed_limit_kmh=120.0,
            tolerance_percent=10.0,
            toll_rate_per_km=2.5,
            speeding_fine_amount=1500.0,
        )
    )
    app = create_app(
        config=ServingConfig(),
        watchlist=watchlist,
        parking_manager=parking,
        corridor_monitor=corridor,
    )
    client = AsgiTestClient(app)

    blocked = client.post(
        "/v1/parking/entry",
        json={"plate_text": "34 BLK 001", "country_code": "TR", "timestamp_ms": 0},
    )
    assert blocked.status_code == 200
    assert blocked.json()["open_barrier"] is False
    assert blocked.json()["reason"] == "watchlist_blocked"

    vip_in = client.post(
        "/v1/parking/entry",
        json={"plate_text": "34 VIP 777", "country_code": "TR", "timestamp_ms": 1_000},
    )
    assert vip_in.status_code == 200
    assert vip_in.json()["open_barrier"] is True

    car_in = client.post(
        "/v1/parking/entry",
        json={"plate_text": "34 ABC 123", "country_code": "TR", "timestamp_ms": 2_000},
    )
    assert car_in.status_code == 200
    assert car_in.json()["open_barrier"] is True

    status_mid = client.get("/v1/parking/status")
    assert status_mid.status_code == 200
    assert status_mid.json()["occupancy_count"] == 2
    assert status_mid.json()["capacity"] == 10

    car_out = client.post(
        "/v1/parking/exit",
        json={
            "plate_text": "34-ABC-123",
            "country_code": "TR",
            "timestamp_ms": 2_000 + (90 * 60_000),
        },
    )
    assert car_out.status_code == 200
    assert car_out.json()["open_barrier"] is True
    assert car_out.json()["session"]["fee_amount"] == pytest.approx(100.0)

    vip_out = client.post(
        "/v1/parking/exit",
        json={
            "plate_text": "34VIP777",
            "country_code": "TR",
            "timestamp_ms": 1_000 + (120 * 60_000),
        },
    )
    assert vip_out.status_code == 200
    assert vip_out.json()["session"]["fee_amount"] == 0.0

    status_end = client.get("/v1/parking/status")
    assert status_end.status_code == 200
    assert status_end.json()["occupancy_count"] == 0
    assert status_end.json()["total_revenue"] == pytest.approx(100.0)
    assert len(status_end.json()["closed_sessions"]) == 2

    bad_exit = client.post(
        "/v1/corridor/exit",
        json={
            "plate_text": "34ABC123",
            "country_code": "TR",
            "checkpoint_id": "KM_12",
            "timestamp_ms": 50_000,
        },
    )
    assert bad_exit.status_code == 422

    c_in_1 = client.post(
        "/v1/corridor/entry",
        json={
            "plate_text": "34 ABC 123",
            "country_code": "TR",
            "checkpoint_id": "KM_00",
            "timestamp_ms": 0,
        },
    )
    assert c_in_1.status_code == 200
    assert c_in_1.json()["plate_text"] == "34ABC123"

    c_out_1 = client.post(
        "/v1/corridor/exit",
        json={
            "plate_text": "34-ABC-123",
            "country_code": "TR",
            "checkpoint_id": "KM_12",
            "timestamp_ms": 360_000,
        },
    )
    assert c_out_1.status_code == 200
    assert c_out_1.json()["average_speed_kmh"] == pytest.approx(120.0)
    assert c_out_1.json()["is_speed_violation"] is False
    assert c_out_1.json()["total_charge"] == pytest.approx(30.0)

    c_in_2 = client.post(
        "/v1/corridor/entry",
        json={
            "plate_text": "06 XYZ 99",
            "country_code": "TR",
            "checkpoint_id": "KM_00",
            "timestamp_ms": 10_000,
        },
    )
    assert c_in_2.status_code == 200

    c_out_2 = client.post(
        "/v1/corridor/exit",
        json={
            "plate_text": "06XYZ99",
            "country_code": "TR",
            "checkpoint_id": "KM_12",
            "timestamp_ms": 280_000,
        },
    )
    assert c_out_2.status_code == 200
    assert c_out_2.json()["average_speed_kmh"] == pytest.approx(160.0)
    assert c_out_2.json()["is_speed_violation"] is True
    assert c_out_2.json()["total_charge"] == pytest.approx(1530.0)

    summary = client.get("/v1/corridor/summary")
    assert summary.status_code == 200
    data = summary.json()
    assert data["corridor_id"] == "TEM_WEST"
    assert data["total_passages"] == 2
    assert data["violation_count"] == 1
    assert data["total_toll_revenue"] == pytest.approx(60.0)
    assert data["total_fine_revenue"] == pytest.approx(1500.0)
