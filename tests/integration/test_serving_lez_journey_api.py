import pytest

from plate_vision.config.schema import ServingConfig
from plate_vision.serving.app import AsgiTestClient, create_app
from plate_vision.video.journey import CameraNode, MultiCameraJourneyTracker
from plate_vision.video.lez import (
    EmissionClass,
    LowEmissionZoneConfig,
    LowEmissionZoneInspector,
    VehicleEmissionProfile,
)


def test_serving_lez_and_journey_api_workflows() -> None:
    lez = LowEmissionZoneInspector(
        config=LowEmissionZoneConfig(
            zone_id="CBD_01",
            allowed_classes=(EmissionClass.ZERO_EMISSION, EmissionClass.EURO_6),
            enforce_odd_even_days=True,
            non_compliant_fine_amount=1800.0,
            unregistered_fine_amount=2400.0,
            odd_even_fine_amount=900.0,
        ),
        registry=[
            VehicleEmissionProfile(
                plate_text="34 EV 100",
                country_code="TR",
                emission_class=EmissionClass.ZERO_EMISSION,
            ),
            VehicleEmissionProfile(
                plate_text="34 OLD 200",
                country_code="TR",
                emission_class=EmissionClass.EURO_4,
            ),
        ],
    )
    journey = MultiCameraJourneyTracker(
        cameras=(
            CameraNode(camera_id="CAM_W", latitude=41.045, longitude=29.034, zone_name="WEST"),
            CameraNode(camera_id="CAM_E", latitude=41.040, longitude=29.045, zone_name="EAST"),
            CameraNode(camera_id="CAM_A", latitude=39.933, longitude=32.859, zone_name="ANKARA"),
        ),
        cloned_plate_speed_threshold_kmh=250.0,
    )
    app = create_app(
        config=ServingConfig(),
        lez_inspector=lez,
        journey_tracker=journey,
    )
    client = AsgiTestClient(app)

    res_ev = client.post(
        "/v1/lez/inspect",
        json={"plate_text": "34-EV-100", "country_code": "TR", "day_of_month": 5},
    )
    assert res_ev.status_code == 200
    assert res_ev.json()["is_compliant"] is True
    assert res_ev.json()["fine_amount"] == 0.0

    res_old = client.post(
        "/v1/lez/inspect",
        json={"plate_text": "34OLD200", "country_code": "TR", "day_of_month": 5},
    )
    assert res_old.status_code == 200
    assert res_old.json()["is_compliant"] is False
    assert res_old.json()["violation_reason"] == "restricted_emission_class"
    assert res_old.json()["fine_amount"] == pytest.approx(1800.0)

    lez_sum = client.get("/v1/lez/summary")
    assert lez_sum.status_code == 200
    assert lez_sum.json()["total_inspections"] == 2
    assert lez_sum.json()["violation_count"] == 1
    assert lez_sum.json()["total_fine_amount"] == pytest.approx(1800.0)

    bad_cam = client.post(
        "/v1/journey/sighting",
        json={
            "camera_id": "NO_SUCH_CAM",
            "plate_text": "34ABC123",
            "country_code": "TR",
            "timestamp_ms": 1000,
        },
    )
    assert bad_cam.status_code == 422

    s1 = client.post(
        "/v1/journey/sighting",
        json={
            "camera_id": "CAM_W",
            "plate_text": "34 ABC 123",
            "country_code": "TR",
            "timestamp_ms": 0,
            "confidence": 0.96,
        },
    )
    assert s1.status_code == 200
    assert s1.json()["plate_text"] == "34ABC123"

    s2 = client.post(
        "/v1/journey/sighting",
        json={
            "camera_id": "CAM_E",
            "plate_text": "34-ABC-123",
            "country_code": "TR",
            "timestamp_ms": 120_000,
            "confidence": 0.94,
        },
    )
    assert s2.status_code == 200

    j_ok = client.post(
        "/v1/journey/query",
        json={"plate_text": "34ABC123", "country_code": "TR"},
    )
    assert j_ok.status_code == 200
    assert j_ok.json()["visited_zones"] == ["WEST", "EAST"]
    assert j_ok.json()["has_cloned_plate_anomaly"] is False

    client.post(
        "/v1/journey/sighting",
        json={
            "camera_id": "CAM_W",
            "plate_text": "06CLN99",
            "country_code": "TR",
            "timestamp_ms": 10_000,
        },
    )
    client.post(
        "/v1/journey/sighting",
        json={
            "camera_id": "CAM_A",
            "plate_text": "06CLN99",
            "country_code": "TR",
            "timestamp_ms": 20_000,
        },
    )
    j_cloned = client.post(
        "/v1/journey/query",
        json={"plate_text": "06CLN99", "country_code": "TR"},
    )
    assert j_cloned.status_code == 200
    assert j_cloned.json()["has_cloned_plate_anomaly"] is True

    j_missing = client.post(
        "/v1/journey/query",
        json={"plate_text": "99ZZ999", "country_code": "TR"},
    )
    assert j_missing.status_code == 422
