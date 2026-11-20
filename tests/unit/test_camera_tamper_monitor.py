import pytest

from plate_vision.utils.exceptions import InvalidInputError
from plate_vision.video.tamper import (
    CameraHealthThresholds,
    CameraTamperMonitor,
    TamperSignal,
)


def _healthy_grid() -> list[list[int]]:
    return [[20 if (x + y) % 2 == 0 else 180 for x in range(8)] for y in range(8)]


def test_camera_tamper_monitor_grid_validation() -> None:
    monitor = CameraTamperMonitor()
    with pytest.raises(InvalidInputError):
        monitor.evaluate_grid("   ", _healthy_grid())
    with pytest.raises(InvalidInputError):
        monitor.evaluate_grid("CAM_1", [[100, 100]])
    with pytest.raises(InvalidInputError):
        monitor.evaluate_grid("CAM_1", [[100], [100]])
    with pytest.raises(InvalidInputError):
        monitor.evaluate_grid("CAM_1", [[10, 20], [30]])
    with pytest.raises(InvalidInputError):
        monitor.evaluate_grid("CAM_1", [[10, 300], [20, 40]])
    with pytest.raises(InvalidInputError):
        monitor.evaluate_grid("CAM_1", [[10, -1], [20, 40]])


def test_camera_tamper_signals_escalation_and_recovery() -> None:
    monitor = CameraTamperMonitor(
        thresholds=CameraHealthThresholds(
            min_blackout_brightness=8.0,
            min_contrast_std=6.0,
            max_saturated_ratio=0.45,
            min_sharpness_score=3.0,
            max_quadrant_shift=45.0,
            consecutive_alert_frames=2,
        )
    )
    healthy = _healthy_grid()
    ref_means = monitor.calibrate_reference("CAM_A", healthy)
    assert len(ref_means) == 4

    ok = monitor.evaluate_grid("CAM_A", healthy)
    assert ok.signals == (TamperSignal.NONE,)
    assert ok.is_compromised is False
    assert ok.consecutive_compromised_count == 0
    assert ok.maintenance_alert_triggered is False

    blackout = [[2 for _ in range(8)] for _ in range(8)]
    b1 = monitor.evaluate_grid("CAM_A", blackout)
    assert TamperSignal.SIGNAL_BLACKOUT in b1.signals
    assert TamperSignal.LENS_OCCLUSION not in b1.signals
    assert b1.is_compromised is True
    assert b1.consecutive_compromised_count == 1
    assert b1.maintenance_alert_triggered is False

    occluded = [[110 for _ in range(8)] for _ in range(8)]
    o2 = monitor.evaluate_grid("CAM_A", occluded)
    assert TamperSignal.LENS_OCCLUSION in o2.signals
    assert TamperSignal.SEVERE_DEFOCUS in o2.signals
    assert o2.consecutive_compromised_count == 2
    assert o2.maintenance_alert_triggered is True
    assert monitor.alerted_cameras() == ("CAM_A",)

    glare = [[250 if x < 5 else 20 for x in range(8)] for _ in range(8)]
    g3 = monitor.evaluate_grid("CAM_B", glare)
    assert TamperSignal.GLARE_OVEREXPOSURE in g3.signals

    shifted = [
        [200 if y < 4 and x < 4 else (20 if (x + y) % 2 == 0 else 180) for x in range(8)]
        for y in range(8)
    ]
    s_fov = monitor.evaluate_grid("CAM_A", shifted)
    assert TamperSignal.FOV_SHIFT in s_fov.signals
    assert s_fov.quadrant_shift > 45.0

    recovered = monitor.evaluate_grid("CAM_A", healthy)
    assert recovered.signals == (TamperSignal.NONE,)
    assert recovered.is_compromised is False
    assert recovered.consecutive_compromised_count == 0
    assert recovered.maintenance_alert_triggered is False
    assert monitor.alerted_cameras() == ()
