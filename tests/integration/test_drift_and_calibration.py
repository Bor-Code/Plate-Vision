from plate_vision.evaluation.monitoring import (
    DriftDetector,
    TemperatureScaler,
    compute_expected_calibration_error,
)


def test_end_to_end_drift_detection_and_calibration_improvement() -> None:
    baseline_brightness = [120.0 + (i % 15) for i in range(50)]
    baseline_contrast = [40.0 + (i % 10) for i in range(50)]
    baseline_conf = [0.85 + 0.002 * (i % 10) for i in range(50)]

    detector = DriftDetector(
        baseline_brightness=baseline_brightness,
        baseline_contrast=baseline_contrast,
        baseline_confidences=baseline_conf,
        psi_threshold=0.25,
        num_bins=5,
    )

    stable_report = detector.evaluate(
        current_brightness=baseline_brightness,
        current_contrast=baseline_contrast,
        current_confidences=baseline_conf,
    )
    assert stable_report.drift_detected is False
    assert stable_report.status == "stable"
    assert stable_report.max_psi == 0.0

    shifted_brightness = [30.0 + (i % 10) for i in range(50)]
    shifted_conf = [0.45 + 0.002 * (i % 10) for i in range(50)]
    drift_report = detector.evaluate(
        current_brightness=shifted_brightness,
        current_contrast=baseline_contrast,
        current_confidences=shifted_conf,
    )
    assert drift_report.drift_detected is True
    assert drift_report.status == "drift_alert"
    assert drift_report.brightness_psi > 0.25

    raw_confidences = [0.95 for _ in range(20)]
    outcomes = [i % 2 == 0 for i in range(20)]
    before_cal = compute_expected_calibration_error(raw_confidences, outcomes, num_bins=5)

    scaler = TemperatureScaler()
    scaler.fit(raw_confidences, outcomes)
    calibrated_confidences = [scaler.scale_confidence(c) for c in raw_confidences]
    after_cal = compute_expected_calibration_error(calibrated_confidences, outcomes, num_bins=5)

    assert after_cal.expected_calibration_error < before_cal.expected_calibration_error
    assert after_cal.brier_score < before_cal.brier_score
