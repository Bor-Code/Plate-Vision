from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from plate_vision.pipeline.runner import PipelineInferenceResult
from plate_vision.pipeline.watchlist import WatchlistMatch
from plate_vision.utils.storage import PredictionAuditRecord
from plate_vision.video.fleet import TerminalZoneType
from plate_vision.video.parking import ParkingSession


class SinglePredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    image_base64: str = Field(min_length=1)
    mime_type: str = Field(default="image/png", min_length=1)
    country_code: str | None = Field(default=None)


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    items: tuple[SinglePredictRequest, ...] = Field(min_length=1)


class VideoPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    frames: tuple[SinglePredictRequest, ...] = Field(min_length=1)
    fps: float = Field(default=30.0, gt=0.0)


class BatchPredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    results: tuple[PipelineInferenceResult, ...]
    total_plates: int = Field(ge=0)
    batch_latency_ms: float = Field(ge=0.0)


class AuditLogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    records: tuple[PredictionAuditRecord, ...]
    count: int = Field(ge=0)


class WatchlistCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plate_text: str = Field(min_length=1)
    country_code: str = Field(default="TR", min_length=2)


class WatchlistCheckResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    matches: tuple[WatchlistMatch, ...]
    matched: bool


class DriftEvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    brightness: tuple[float, ...] = Field(min_length=1)
    contrast: tuple[float, ...] = Field(min_length=1)
    confidences: tuple[float, ...] = Field(min_length=1)


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


class ParkingGateEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plate_text: str = Field(min_length=1)
    country_code: str = Field(default="TR", min_length=2)
    timestamp_ms: int = Field(ge=0)


class ParkingStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    occupancy_count: int = Field(ge=0)
    capacity: int = Field(ge=1)
    total_revenue: float = Field(ge=0.0)
    active_sessions: tuple[ParkingSession, ...]
    closed_sessions: tuple[ParkingSession, ...]


class CorridorCheckpointEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plate_text: str = Field(min_length=1)
    country_code: str = Field(default="TR", min_length=2)
    checkpoint_id: str = Field(default="CP_01", min_length=1)
    timestamp_ms: int = Field(ge=0)


class LezInspectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plate_text: str = Field(min_length=1)
    country_code: str = Field(default="TR", min_length=2)
    day_of_month: int = Field(default=1, ge=1, le=31)


class JourneyQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plate_text: str = Field(min_length=1)
    country_code: str = Field(default="TR", min_length=2)


class CameraTamperCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    camera_id: str = Field(min_length=1)
    image_base64: str = Field(min_length=1)
    mime_type: str = Field(default="image/png", min_length=1)
    calibrate_reference: bool = False


class CameraTamperAlertsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    alerted_cameras: tuple[str, ...]
    count: int = Field(ge=0)


class FleetCheckInRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plate_text: str = Field(min_length=1)
    country_code: str = Field(default="TR", min_length=2)
    carrier_name: str = Field(default="DEFAULT_CARRIER", min_length=1)
    assigned_zone: TerminalZoneType = TerminalZoneType.LOADING_DOCK
    timestamp_ms: int = Field(ge=0)


class FleetCheckOutRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plate_text: str = Field(min_length=1)
    country_code: str = Field(default="TR", min_length=2)
    timestamp_ms: int = Field(ge=0)


__all__ = [
    "AuditLogResponse",
    "BatchPredictRequest",
    "BatchPredictResponse",
    "CameraTamperAlertsResponse",
    "CameraTamperCheckRequest",
    "CorridorCheckpointEventRequest",
    "DriftEvaluationRequest",
    "FleetCheckInRequest",
    "FleetCheckOutRequest",
    "HealthStatusResponse",
    "JourneyQueryRequest",
    "LezInspectRequest",
    "ParkingGateEventRequest",
    "ParkingStatusResponse",
    "ReadinessStatusResponse",
    "SinglePredictRequest",
    "StructuredErrorResponse",
    "VideoPredictRequest",
    "WatchlistCheckRequest",
    "WatchlistCheckResponse",
]
