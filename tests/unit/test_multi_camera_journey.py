import pytest

from plate_vision.utils.exceptions import ConfigurationError, InvalidInputError
from plate_vision.video.journey import (
    CameraNode,
    CameraSighting,
    MultiCameraJourneyTracker,
    haversine_distance_km,
)


def test_haversine_and_journey_validation() -> None:
    zero_dist = haversine_distance_km(41.0082, 28.9784, 41.0082, 28.9784)
    assert zero_dist == pytest.approx(0.0)

    istanbul_ankara = haversine_distance_km(41.0082, 28.9784, 39.9334, 32.8597)
    assert 345.0 < istanbul_ankara < 360.0

    with pytest.raises(ConfigurationError):
        MultiCameraJourneyTracker(cloned_plate_speed_threshold_kmh=0.0)

    tracker = MultiCameraJourneyTracker()
    with pytest.raises(InvalidInputError):
        tracker.register_camera(CameraNode(camera_id="   ", latitude=41.0, longitude=29.0))

    tracker.register_camera(CameraNode(camera_id="CAM_1", latitude=41.0, longitude=29.0))
    with pytest.raises(InvalidInputError):
        tracker.record_sighting(
            CameraSighting(camera_id="MISSING", plate_text="34ABC123", timestamp_ms=1000)
        )
    with pytest.raises(InvalidInputError):
        tracker.record_sighting(
            CameraSighting(camera_id="CAM_1", plate_text="A-", timestamp_ms=1000)
        )
    with pytest.raises(InvalidInputError):
        tracker.build_journey("34ABC123")


def test_multi_camera_journey_reconstruction_and_cloned_plate_detection() -> None:
    cameras = (
        CameraNode(camera_id="CAM_A", latitude=41.0082, longitude=28.9784, zone_name="CENTER"),
        CameraNode(camera_id="CAM_B", latitude=41.0900, longitude=29.0500, zone_name="NORTH"),
        CameraNode(camera_id="CAM_C", latitude=39.9334, longitude=32.8597, zone_name="ANKARA"),
    )
    tracker = MultiCameraJourneyTracker(
        cameras=cameras,
        cloned_plate_speed_threshold_kmh=250.0,
    )

    tracker.record_sighting(
        CameraSighting(camera_id="CAM_B", plate_text="34-ABC-123", timestamp_ms=600_000)
    )
    tracker.record_sighting(
        CameraSighting(camera_id="CAM_A", plate_text="34 abc 123", timestamp_ms=0)
    )
    tracker.record_sighting(
        CameraSighting(camera_id="CAM_B", plate_text="34ABC123", timestamp_ms=600_000)
    )

    normal_report = tracker.build_journey("34ABC123", country_code="TR")
    assert normal_report.plate_text == "34ABC123"
    assert normal_report.sightings_count == 3
    assert normal_report.visited_camera_ids == ("CAM_A", "CAM_B", "CAM_B")
    assert normal_report.visited_zones == ("CENTER", "NORTH")
    assert len(normal_report.legs) == 2
    assert normal_report.legs[0].is_cloned_plate_anomaly is False
    assert normal_report.legs[1].estimated_speed_kmh is None
    assert normal_report.legs[1].is_cloned_plate_anomaly is False
    assert normal_report.has_cloned_plate_anomaly is False

    tracker.record_sighting(
        CameraSighting(camera_id="CAM_A", plate_text="06CLN99", timestamp_ms=10_000)
    )
    tracker.record_sighting(
        CameraSighting(camera_id="CAM_C", plate_text="06CLN99", timestamp_ms=20_000)
    )
    tracker.record_sighting(
        CameraSighting(camera_id="CAM_A", plate_text="35SIM77", timestamp_ms=1_000)
    )
    tracker.record_sighting(
        CameraSighting(camera_id="CAM_C", plate_text="35SIM77", timestamp_ms=1_000)
    )

    cloned_report = tracker.build_journey("06CLN99")
    assert cloned_report.has_cloned_plate_anomaly is True
    assert cloned_report.legs[0].estimated_speed_kmh is not None
    assert cloned_report.legs[0].estimated_speed_kmh > 250.0

    simultaneous_report = tracker.build_journey("35SIM77")
    assert simultaneous_report.has_cloned_plate_anomaly is True
    assert simultaneous_report.legs[0].estimated_speed_kmh is None

    assert tracker.anomalous_vehicles() == (("TR", "06CLN99"), ("TR", "35SIM77"))
