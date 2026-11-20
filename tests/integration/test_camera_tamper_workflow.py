from plate_vision.data.synthetic import encode_grayscale_png
from plate_vision.video.tamper import (
    CameraHealthThresholds,
    CameraTamperMonitor,
    TamperSignal,
)


def test_camera_tamper_image_bytes_workflow() -> None:
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
    textured_frame = [[30 if (x + y) % 2 == 0 else 190 for x in range(32)] for y in range(16)]
    ok_png = encode_grayscale_png(textured_frame)
    ok_res = monitor.evaluate_image_bytes("GATE_CAM_01", ok_png)
    assert ok_res.signals == (TamperSignal.NONE,)
    assert ok_res.is_compromised is False

    covered_frame = [[95 for _ in range(32)] for _ in range(16)]
    bad_png = encode_grayscale_png(covered_frame)
    warn_1 = monitor.evaluate_image_bytes("GATE_CAM_01", bad_png)
    assert warn_1.is_compromised is True
    assert TamperSignal.LENS_OCCLUSION in warn_1.signals
    assert warn_1.maintenance_alert_triggered is False

    warn_2 = monitor.evaluate_image_bytes("GATE_CAM_01", bad_png)
    assert warn_2.is_compromised is True
    assert warn_2.maintenance_alert_triggered is True
    assert monitor.alerted_cameras() == ("GATE_CAM_01",)

    recover_res = monitor.evaluate_image_bytes("GATE_CAM_01", ok_png)
    assert recover_res.is_compromised is False
    assert monitor.alerted_cameras() == ()
