import pytest
from pydantic import ValidationError

from plate_vision.serving import (
    CameraTamperAlertsResponse,
    CameraTamperCheckRequest,
    FleetCheckInRequest,
    FleetCheckOutRequest,
)
from plate_vision.video.fleet import TerminalZoneType


def test_serving_tamper_and_fleet_schemas() -> None:
    tamper_req = CameraTamperCheckRequest(
        camera_id="CAM_NORTH",
        image_base64="AAAA",
        mime_type="image/png",
        calibrate_reference=True,
    )
    assert tamper_req.camera_id == "CAM_NORTH"
    assert tamper_req.calibrate_reference is True

    alerts = CameraTamperAlertsResponse(alerted_cameras=("CAM_NORTH",), count=1)
    assert alerts.count == 1
    assert alerts.alerted_cameras == ("CAM_NORTH",)

    check_in = FleetCheckInRequest(
        plate_text="34LOG100",
        country_code="TR",
        carrier_name="MARMARA_FREIGHT",
        assigned_zone=TerminalZoneType.WEIGHBRIDGE,
        timestamp_ms=1_000,
    )
    assert check_in.assigned_zone == TerminalZoneType.WEIGHBRIDGE

    check_out = FleetCheckOutRequest(
        plate_text="34LOG100",
        country_code="TR",
        timestamp_ms=61_000,
    )
    assert check_out.timestamp_ms == 61_000

    with pytest.raises(ValidationError):
        CameraTamperCheckRequest(camera_id="", image_base64="AAAA")
    with pytest.raises(ValidationError):
        CameraTamperCheckRequest(camera_id="CAM", image_base64="")
    with pytest.raises(ValidationError):
        CameraTamperAlertsResponse(alerted_cameras=(), count=-1)
    with pytest.raises(ValidationError):
        FleetCheckInRequest(plate_text="", timestamp_ms=0)
    with pytest.raises(ValidationError):
        FleetCheckOutRequest(plate_text="34ABC123", timestamp_ms=-1)
