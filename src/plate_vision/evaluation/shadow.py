from pydantic import BaseModel, ConfigDict, Field

from plate_vision.pipeline.runner import PipelineInferenceResult, PlateVisionPipeline
from plate_vision.utils.exceptions import ConfigurationError, InvalidInputError


class ShadowComparisonRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str = Field(min_length=1)
    primary_plates: tuple[str, ...]
    candidate_plates: tuple[str, ...]
    exact_agreement: bool
    primary_mean_confidence: float = Field(ge=0.0, le=1.0)
    candidate_mean_confidence: float = Field(ge=0.0, le=1.0)
    primary_latency_ms: float = Field(ge=0.0)
    candidate_latency_ms: float = Field(ge=0.0)


class ShadowEvaluationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    total_samples: int = Field(ge=0)
    agreement_rate: float = Field(ge=0.0, le=1.0)
    mean_confidence_delta: float
    mean_latency_delta_ms: float
    disagreed_sample_ids: tuple[str, ...]
    promote_recommended: bool


def _mean_confidence(result: PipelineInferenceResult) -> float:
    if not result.plates:
        return 0.0
    return sum(p.combined_confidence for p in result.plates) / float(len(result.plates))


class ShadowPipelineRunner:
    def __init__(
        self,
        primary: PlateVisionPipeline,
        candidate: PlateVisionPipeline,
        *,
        min_agreement_rate: float = 0.90,
        max_latency_regression_ms: float = 15.0,
    ) -> None:
        if not 0.0 <= min_agreement_rate <= 1.0:
            raise ConfigurationError(
                "min_agreement_rate must be in [0.0, 1.0]",
                details={"min_agreement_rate": min_agreement_rate},
            )
        if max_latency_regression_ms < 0.0:
            raise ConfigurationError(
                "max_latency_regression_ms must be >= 0.0",
                details={"max_latency_regression_ms": max_latency_regression_ms},
            )
        self.primary = primary
        self.candidate = candidate
        self.min_agreement_rate = min_agreement_rate
        self.max_latency_regression_ms = max_latency_regression_ms
        self._records: list[ShadowComparisonRecord] = []

    def predict_and_compare(
        self,
        sample_id: str,
        image_bytes: bytes,
    ) -> tuple[PipelineInferenceResult, ShadowComparisonRecord]:
        clean_id = sample_id.strip()
        if not clean_id:
            raise InvalidInputError("sample_id must not be empty")

        primary_res = self.primary.predict_bytes(image_bytes)
        candidate_res = self.candidate.predict_bytes(image_bytes)

        primary_plates = tuple(p.plate_text for p in primary_res.plates)
        candidate_plates = tuple(p.plate_text for p in candidate_res.plates)
        record = ShadowComparisonRecord(
            sample_id=clean_id,
            primary_plates=primary_plates,
            candidate_plates=candidate_plates,
            exact_agreement=(primary_plates == candidate_plates),
            primary_mean_confidence=round(_mean_confidence(primary_res), 6),
            candidate_mean_confidence=round(_mean_confidence(candidate_res), 6),
            primary_latency_ms=primary_res.latency_ms,
            candidate_latency_ms=candidate_res.latency_ms,
        )
        self._records.append(record)
        return primary_res, record

    def summarize(self) -> ShadowEvaluationSummary:
        total = len(self._records)
        if total == 0:
            return ShadowEvaluationSummary(
                total_samples=0,
                agreement_rate=0.0,
                mean_confidence_delta=0.0,
                mean_latency_delta_ms=0.0,
                disagreed_sample_ids=(),
                promote_recommended=False,
            )

        agreed = sum(1 for r in self._records if r.exact_agreement)
        agreement_rate = round(agreed / float(total), 6)
        conf_delta = round(
            sum(r.candidate_mean_confidence - r.primary_mean_confidence for r in self._records)
            / float(total),
            6,
        )
        latency_delta = round(
            sum(r.candidate_latency_ms - r.primary_latency_ms for r in self._records)
            / float(total),
            4,
        )
        disagreed_ids = tuple(r.sample_id for r in self._records if not r.exact_agreement)
        promote = (
            agreement_rate >= self.min_agreement_rate
            and latency_delta <= self.max_latency_regression_ms
        )
        return ShadowEvaluationSummary(
            total_samples=total,
            agreement_rate=agreement_rate,
            mean_confidence_delta=conf_delta,
            mean_latency_delta_ms=latency_delta,
            disagreed_sample_ids=disagreed_ids,
            promote_recommended=promote,
        )

    def records(self) -> tuple[ShadowComparisonRecord, ...]:
        return tuple(self._records)

    def clear(self) -> None:
        self._records.clear()


__all__ = [
    "ShadowComparisonRecord",
    "ShadowEvaluationSummary",
    "ShadowPipelineRunner",
]
