import math
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from plate_vision.utils.exceptions import InvalidInputError


class ReliabilityBin(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lower_bound: float = Field(ge=0.0, le=1.0)
    upper_bound: float = Field(ge=0.0, le=1.0)
    count: int = Field(ge=0)
    mean_confidence: float = Field(ge=0.0, le=1.0)
    empirical_accuracy: float = Field(ge=0.0, le=1.0)
    calibration_gap: float = Field(ge=0.0, le=1.0)


class CalibrationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    expected_calibration_error: float = Field(ge=0.0, le=1.0)
    maximum_calibration_error: float = Field(ge=0.0, le=1.0)
    brier_score: float = Field(ge=0.0, le=1.0)
    bins: tuple[ReliabilityBin, ...]


def compute_expected_calibration_error(
    confidences: Sequence[float],
    correctness: Sequence[bool],
    *,
    num_bins: int = 10,
) -> CalibrationReport:
    if len(confidences) != len(correctness):
        raise InvalidInputError(
            "confidences and correctness must have identical length",
            details={
                "confidences_len": len(confidences),
                "correctness_len": len(correctness),
            },
        )
    if num_bins < 1:
        raise InvalidInputError(
            "num_bins must be >= 1",
            details={"num_bins": num_bins},
        )
    total_samples = len(confidences)
    if total_samples == 0:
        return CalibrationReport(
            expected_calibration_error=0.0,
            maximum_calibration_error=0.0,
            brier_score=0.0,
            bins=(),
        )

    bin_conf_sums = [0.0 for _ in range(num_bins)]
    bin_acc_sums = [0.0 for _ in range(num_bins)]
    bin_counts = [0 for _ in range(num_bins)]
    brier_sum = 0.0

    for conf, is_correct in zip(confidences, correctness, strict=True):
        clamped = min(1.0, max(0.0, float(conf)))
        target = 1.0 if is_correct else 0.0
        brier_sum += (clamped - target) ** 2
        idx = min(num_bins - 1, int(clamped * num_bins))
        bin_conf_sums[idx] += clamped
        bin_acc_sums[idx] += target
        bin_counts[idx] += 1

    bins: list[ReliabilityBin] = []
    ece = 0.0
    mce = 0.0

    for idx in range(num_bins):
        lower = round(idx / num_bins, 6)
        upper = round((idx + 1) / num_bins, 6)
        count = bin_counts[idx]
        if count == 0:
            bins.append(
                ReliabilityBin(
                    lower_bound=lower,
                    upper_bound=upper,
                    count=0,
                    mean_confidence=0.0,
                    empirical_accuracy=0.0,
                    calibration_gap=0.0,
                )
            )
            continue
        mean_conf = bin_conf_sums[idx] / count
        emp_acc = bin_acc_sums[idx] / count
        gap = abs(mean_conf - emp_acc)
        ece += (count / total_samples) * gap
        mce = max(mce, gap)
        bins.append(
            ReliabilityBin(
                lower_bound=lower,
                upper_bound=upper,
                count=count,
                mean_confidence=round(mean_conf, 6),
                empirical_accuracy=round(emp_acc, 6),
                calibration_gap=round(gap, 6),
            )
        )

    return CalibrationReport(
        expected_calibration_error=round(min(1.0, ece), 6),
        maximum_calibration_error=round(min(1.0, mce), 6),
        brier_score=round(min(1.0, brier_sum / total_samples), 6),
        bins=tuple(bins),
    )


class TemperatureScaler:
    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0.0:
            raise InvalidInputError(
                "temperature must be > 0.0",
                details={"temperature": temperature},
            )
        self.temperature = temperature

    def scale_confidence(self, confidence: float) -> float:
        eps = 1e-6
        clamped = min(1.0 - eps, max(eps, float(confidence)))
        logit = math.log(clamped / (1.0 - clamped))
        scaled_logit = logit / self.temperature
        if scaled_logit >= 0.0:
            z = math.exp(-scaled_logit)
            return round(1.0 / (1.0 + z), 6)
        z = math.exp(scaled_logit)
        return round(z / (1.0 + z), 6)

    def fit(
        self,
        confidences: Sequence[float],
        correctness: Sequence[bool],
        *,
        candidates: Sequence[float] = (
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            2.0,
            2.5,
            3.0,
        ),
    ) -> float:
        if len(confidences) != len(correctness) or not confidences:
            raise InvalidInputError(
                "Non-empty matching confidences and correctness required for fitting"
            )
        best_temp = self.temperature
        best_nll = float("inf")
        eps = 1e-6

        for candidate in candidates:
            if candidate <= 0.0:
                continue
            scaler = TemperatureScaler(temperature=candidate)
            nll = 0.0
            for conf, is_correct in zip(confidences, correctness, strict=True):
                p = min(1.0 - eps, max(eps, scaler.scale_confidence(conf)))
                nll += -math.log(p) if is_correct else -math.log(1.0 - p)
            if nll < best_nll:
                best_nll = nll
                best_temp = candidate

        self.temperature = best_temp
        return self.temperature


class DriftReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    brightness_psi: float = Field(ge=0.0)
    contrast_psi: float = Field(ge=0.0)
    confidence_psi: float = Field(ge=0.0)
    max_psi: float = Field(ge=0.0)
    drift_detected: bool
    status: str


def compute_population_stability_index(
    baseline_values: Sequence[float],
    current_values: Sequence[float],
    *,
    num_bins: int = 10,
) -> float:
    if not baseline_values or not current_values:
        raise InvalidInputError("Baseline and current distributions must be non-empty")
    if num_bins < 2:
        raise InvalidInputError("num_bins must be >= 2", details={"num_bins": num_bins})

    min_val = min(min(baseline_values), min(current_values))
    max_val = max(max(baseline_values), max(current_values))
    if math.isclose(min_val, max_val, abs_tol=1e-9):
        return 0.0

    span = max_val - min_val
    base_counts = [0 for _ in range(num_bins)]
    curr_counts = [0 for _ in range(num_bins)]

    for val in baseline_values:
        idx = min(num_bins - 1, max(0, int(((val - min_val) / span) * num_bins)))
        base_counts[idx] += 1

    for val in current_values:
        idx = min(num_bins - 1, max(0, int(((val - min_val) / span) * num_bins)))
        curr_counts[idx] += 1

    eps = 1e-4
    base_total = float(len(baseline_values))
    curr_total = float(len(current_values))
    psi = 0.0

    for b_cnt, c_cnt in zip(base_counts, curr_counts, strict=True):
        b_prop = max(eps, b_cnt / base_total)
        c_prop = max(eps, c_cnt / curr_total)
        psi += (c_prop - b_prop) * math.log(c_prop / b_prop)

    return round(max(0.0, psi), 6)


class DriftDetector:
    def __init__(
        self,
        baseline_brightness: Sequence[float],
        baseline_contrast: Sequence[float],
        baseline_confidences: Sequence[float],
        *,
        psi_threshold: float = 0.25,
        num_bins: int = 10,
    ) -> None:
        if not baseline_brightness or not baseline_contrast or not baseline_confidences:
            raise InvalidInputError("All baseline distributions must be non-empty")
        self.baseline_brightness = tuple(float(x) for x in baseline_brightness)
        self.baseline_contrast = tuple(float(x) for x in baseline_contrast)
        self.baseline_confidences = tuple(float(x) for x in baseline_confidences)
        self.psi_threshold = psi_threshold
        self.num_bins = num_bins

    def evaluate(
        self,
        current_brightness: Sequence[float],
        current_contrast: Sequence[float],
        current_confidences: Sequence[float],
    ) -> DriftReport:
        b_psi = compute_population_stability_index(
            self.baseline_brightness,
            current_brightness,
            num_bins=self.num_bins,
        )
        c_psi = compute_population_stability_index(
            self.baseline_contrast,
            current_contrast,
            num_bins=self.num_bins,
        )
        conf_psi = compute_population_stability_index(
            self.baseline_confidences,
            current_confidences,
            num_bins=self.num_bins,
        )
        max_psi = max(b_psi, c_psi, conf_psi)
        drift_detected = max_psi >= self.psi_threshold
        if drift_detected:
            status = "drift_alert"
        elif max_psi >= 0.10:
            status = "moderate_shift"
        else:
            status = "stable"

        return DriftReport(
            brightness_psi=b_psi,
            contrast_psi=c_psi,
            confidence_psi=conf_psi,
            max_psi=max_psi,
            drift_detected=drift_detected,
            status=status,
        )


__all__ = [
    "CalibrationReport",
    "DriftDetector",
    "DriftReport",
    "ReliabilityBin",
    "TemperatureScaler",
    "compute_expected_calibration_error",
    "compute_population_stability_index",
]
