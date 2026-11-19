import pytest
from pydantic import ValidationError

from plate_vision.serving import JourneyQueryRequest, LezInspectRequest


def test_serving_lez_and_journey_schemas() -> None:
    lez_req = LezInspectRequest(
        plate_text="34ABC123",
        country_code="TR",
        day_of_month=15,
    )
    assert lez_req.plate_text == "34ABC123"
    assert lez_req.country_code == "TR"
    assert lez_req.day_of_month == 15

    journey_req = JourneyQueryRequest(plate_text="34ABC123", country_code="TR")
    assert journey_req.plate_text == "34ABC123"
    assert journey_req.country_code == "TR"

    with pytest.raises(ValidationError):
        LezInspectRequest(plate_text="", day_of_month=1)
    with pytest.raises(ValidationError):
        LezInspectRequest(plate_text="34ABC123", day_of_month=0)
    with pytest.raises(ValidationError):
        LezInspectRequest(plate_text="34ABC123", day_of_month=32)
    with pytest.raises(ValidationError):
        JourneyQueryRequest(plate_text="", country_code="TR")
