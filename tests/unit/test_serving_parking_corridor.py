import pytest
from pydantic import ValidationError

from plate_vision.serving import (
    CorridorCheckpointEventRequest,
    ParkingGateEventRequest,
    ParkingStatusResponse,
)
from plate_vision.video.parking import ParkingSession


def test_serving_parking_and_corridor_schemas() -> None:
    gate_req = ParkingGateEventRequest(
        plate_text="34ABC123",
        country_code="TR",
        timestamp_ms=10_000,
    )
    assert gate_req.plate_text == "34ABC123"
    assert gate_req.country_code == "TR"
    assert gate_req.timestamp_ms == 10_000

    session = ParkingSession(
        session_id="sess_000001",
        plate_text="34ABC123",
        country_code="TR",
        entry_timestamp_ms=10_000,
        is_vip=False,
        is_active=True,
    )
    status = ParkingStatusResponse(
        occupancy_count=1,
        capacity=100,
        total_revenue=0.0,
        active_sessions=(session,),
        closed_sessions=(),
    )
    assert status.occupancy_count == 1
    assert status.capacity == 100
    assert status.active_sessions[0].session_id == "sess_000001"

    corridor_req = CorridorCheckpointEventRequest(
        plate_text="06XYZ99",
        country_code="TR",
        checkpoint_id="KM_00",
        timestamp_ms=5_000,
    )
    assert corridor_req.checkpoint_id == "KM_00"

    with pytest.raises(ValidationError):
        ParkingGateEventRequest(plate_text="", timestamp_ms=0)
    with pytest.raises(ValidationError):
        ParkingGateEventRequest(plate_text="34ABC123", timestamp_ms=-1)
    with pytest.raises(ValidationError):
        ParkingStatusResponse(
            occupancy_count=0,
            capacity=0,
            total_revenue=0.0,
            active_sessions=(),
            closed_sessions=(),
        )
    with pytest.raises(ValidationError):
        CorridorCheckpointEventRequest(plate_text="34ABC123", checkpoint_id="", timestamp_ms=0)
