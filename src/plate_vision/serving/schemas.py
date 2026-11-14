from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from plate_vision.pipeline.runner import PipelineInferenceResult


class SinglePredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    image_base64: str = Field(min_length=1)
    mime_type: str = Field(default="image/png", min_length=1)


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    items: tuple[SinglePredictRequest, ...] = Field(min_length=1)


class BatchPredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    results: tuple[PipelineInferenceResult, ...]
    total_plates: int = Field(ge=0)
    batch_latency_ms: float = Field(ge=0.0)


class HealthStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str
    service: str
    version: str


class ReadinessStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ready: bool
    detector_loaded: bool
    recognizer_loaded: bool


class StructuredErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    error_code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "BatchPredictRequest",
    "BatchPredictResponse",
    "HealthStatusResponse",
    "ReadinessStatusResponse",
    "SinglePredictRequest",
    "StructuredErrorResponse",
]
