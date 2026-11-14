from pydantic import BaseModel, ConfigDict, Field

from plate_vision.detection.models import DetectorMetrics
from plate_vision.recognition.trainer import RecognizerEpochMetrics


class ConfusionPair(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_char: str = Field(min_length=1, max_length=1)
    predicted_char: str = Field(min_length=1, max_length=1)
    count: int = Field(ge=1)


class WorstSampleEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str = Field(min_length=1)
    target_text: str
    predicted_text: str
    edit_distance: int = Field(ge=0)
    character_error_rate: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)


class EndToEndMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plate_level_accuracy: float = Field(ge=0.0, le=1.0)
    latency_p50_ms: float = Field(ge=0.0)
    latency_p95_ms: float = Field(ge=0.0)
    throughput_fps: float = Field(ge=0.0)


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    detector: DetectorMetrics
    recognizer: RecognizerEpochMetrics
    end_to_end: EndToEndMetrics
    confusion_pairs: tuple[ConfusionPair, ...]
    worst_samples: tuple[WorstSampleEntry, ...]


__all__ = [
    "ConfusionPair",
    "EndToEndMetrics",
    "EvaluationReport",
    "WorstSampleEntry",
]
