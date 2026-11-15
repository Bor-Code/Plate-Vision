import pytest

from plate_vision.evaluation.monitoring import (
    DriftDetector,
    TemperatureScaler,
    compute_expected_calibration_error,
    compute_population_stability_index,
)
from plate_vision.utils.exceptions import InvalidInputError


def test_compute_expected_calibration_error_and_bins() -> None:
    empty_report = compute_expected_calibration_error([], [], num_bins=5)
    assert empty_report.expected_calibration_error == 0.0
    assert empty_report.bins == ()

    confidences = [0.9, 0.9, 0.9, 0.9, 0.2, 0.2]
    correctness = [True, True, False, False, False, False]
    report = compute_expected_calibration_error(confidences, correctness, num_bins=5)
    assert len(report.bins) == 5
    assert 0.0 < report.expected_calibration_error < 1.0
    assert 0.0 < report.brier_score < 1.0

    with pytest.raises(InvalidInputError):
        compute_expected_calibration_error([0.5], [True, False])
    with pytest.raises(InvalidInputError):
        compute_expected_calibration_error([0.5], [True], num_bins=0)


def test_temperature_scaler_fit_and_scaling() -> None:
    with pytest.raises(InvalidInputError):
        TemperatureScaler(temperature=0.0)

    scaler = TemperatureScaler(temperature=2.0)
    assert scaler.scale_confidence(0.9) < 0.9
    assert scaler.scale_confidence(0.1) > 0.1

    overconfident = [0.98, 0.97, 0.96, 0.95, 0.94, 0.93]
    labels = [True, False, True, False, True, False]
    fitted_t = scaler.fit(overconfident, labels)
    assert fitted_t > 1.0

    with pytest.raises(InvalidInputError):
        scaler.fit([], [])


def test_psi_and_drift_detector_validation() -> None:
    with pytest.raises(InvalidInputError):
        compute_population_stability_index([], [1.0])
    with pytest.raises(InvalidInputError):
        compute_population_stability_index([1.0], [1.0], num_bins=1)

    assert compute_population_stability_index([5.0, 5.0], [5.0, 5.0]) == 0.0

    with pytest.raises(InvalidInputError):
        DriftDetector([], [1.0], [0.9])
