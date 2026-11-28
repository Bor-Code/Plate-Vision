import pytest
from pydantic import ValidationError

from plate_vision.serving import (
    TollAccountRegisterRequest,
    TollPassageRequest,
    TollTopUpRequest,
)
from plate_vision.video import VehicleTollClass


def test_serving_toll_request_schemas_valid_and_defaults() -> None:
    acc_req = TollAccountRegisterRequest(plate_text="34ABC123", balance=100.0)
    assert acc_req.country_code == "TR"
    assert acc_req.vehicle_class == VehicleTollClass.PASSENGER
    assert acc_req.postpaid_enabled is False
    assert acc_req.is_exempt is False

    top_req = TollTopUpRequest(plate_text="34ABC123", amount=50.0)
    assert top_req.country_code == "TR"
    assert top_req.amount == 50.0

    pass_req = TollPassageRequest(
        plate_text="34ABC123",
        timestamp_ms=10_000,
        vehicle_class=VehicleTollClass.TRUCK,
    )
    assert pass_req.vehicle_class == VehicleTollClass.TRUCK


def test_serving_toll_request_schemas_reject_invalid_fields() -> None:
    with pytest.raises(ValidationError):
        TollTopUpRequest(plate_text="34ABC123", amount=0.0)

    with pytest.raises(ValidationError):
        TollPassageRequest(plate_text="34ABC123", timestamp_ms=-1)

    with pytest.raises(ValidationError):
        TollAccountRegisterRequest.model_validate(
            {
                "plate_text": "34ABC123",
                "extra_field": "disallowed",
            }
        )
