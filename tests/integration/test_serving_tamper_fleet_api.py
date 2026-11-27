import base64

import pytest

from plate_vision.config.schema import ServingConfig
from plate_vision.data.synthetic import encode_grayscale_png
from plate_vision.serving.app import AsgiTestClient, create_app
from plate_vision.video.fleet import FleetSlaConfig, FleetTerminalController
from plate_vision.video.tamper import CameraHealthThresholds, CameraTamperMonitor


def _b64_png(grid: list[list[int]]) -> str:
    return base64.b64encode(encode_grayscale_png(grid)).decode("ascii")


def test_serving_tamper_and_fleet_api_workflows() -> None:
    tamper = CameraTamperMonitor(
        thresholds=CameraHealthThresholds(
            min_blackout_brightness=8.0,
            min_contrast_std=6.0,
            max_saturated_ratio=0.45,
            min_sharpness_score=3.0,
            max_quadrant_shift=45.0,
            consecutive_alert_frames=2,
        )
    )
    fleet = FleetTerminalController(
        config=FleetSlaConfig(
            facility_id="IZMIR_PORT",
            max_turnaround_minutes=30.0,
            demurrage_rate_per_hour=200.0,
            currency="TRY",
        )
    )
    app = create_app(
        config=ServingConfig(),
        tamper_monitor=tamper,
        fleet_controller=fleet,
    )
    client = AsgiTestClient(app)

    healthy_b64 = _b64_png(
        [[25 if (x + y) % 2 == 0 else 185 for x in range(16)] for y in range(16)]
    )
    covered_b64 = _b64_png([[95 for _ in range(16)] for _ in range(16)])

    ok_resp = client.post(
        "/v1/camera/tamper/check",
        json={
            "camera_id": "CAM_01",
            "image_base64": healthy_b64,
            "calibrate_reference": True,
        },
    )
    assert ok_resp.status_code == 200
    assert ok_resp.json()["is_compromised"] is False
    assert ok_resp.json()["signals"] == ["none"]

    bad_1 = client.post(
        "/v1/camera/tamper/check",
        json={"camera_id": "CAM_01", "image_base64": covered_b64},
    )
    assert bad_1.status_code == 200
    assert bad_1.json()["is_compromised"] is True
    assert bad_1.json()["maintenance_alert_triggered"] is False

    bad_2 = client.post(
        "/v1/camera/tamper/check",
        json={"camera_id": "CAM_01", "image_base64": covered_b64},
    )
    assert bad_2.status_code == 200
    assert bad_2.json()["maintenance_alert_triggered"] is True

    alerts = client.get("/v1/camera/tamper/alerts")
    assert alerts.status_code == 200
    assert alerts.json()["count"] == 1
    assert alerts.json()["alerted_cameras"] == ["CAM_01"]

    in_resp = client.post(
        "/v1/fleet/check-in",
        json={
            "plate_text": "35 TRK 77",
            "country_code": "TR",
            "carrier_name": "EGE_CARGO",
            "assigned_zone": "weighbridge",
            "timestamp_ms": 0,
        },
    )
    assert in_resp.status_code == 200
    assert in_resp.json()["plate_text"] == "35TRK77"

    out_resp = client.post(
        "/v1/fleet/check-out",
        json={
            "plate_text": "35-TRK-77",
            "country_code": "TR",
            "timestamp_ms": 60 * 60_000,
        },
    )
    assert out_resp.status_code == 200
    assert out_resp.json()["sla_breached"] is True
    assert out_resp.json()["overstay_minutes"] == pytest.approx(30.0)
    assert out_resp.json()["demurrage_fee"] == pytest.approx(100.0)

    bad_out = client.post(
        "/v1/fleet/check-out",
        json={"plate_text": "35TRK77", "country_code": "TR", "timestamp_ms": 70 * 60_000},
    )
    assert bad_out.status_code == 422

    summary = client.get("/v1/fleet/summary")
    assert summary.status_code == 200
    assert summary.json()["facility_id"] == "IZMIR_PORT"
    assert summary.json()["completed_visits"] == 1
    assert summary.json()["sla_breach_count"] == 1
    assert summary.json()["total_demurrage_fee"] == pytest.approx(100.0)
