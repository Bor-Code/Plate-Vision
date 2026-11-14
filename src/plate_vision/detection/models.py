from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from plate_vision.config.schema import DetectorConfig
from plate_vision.data.models import BoundingBox


class DetectionPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    bbox: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    class_id: int = Field(default=0, ge=0)


class DetectorMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    map50: float = Field(ge=0.0, le=1.0)
    map50_95: float = Field(ge=0.0, le=1.0)

    @property
    def fitness(self) -> float:
        return 0.1 * self.map50 + 0.9 * self.map50_95


class DetectorCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    epoch: int = Field(ge=1)
    best_fitness: float = Field(ge=0.0)
    metrics: DetectorMetrics
    model_architecture: str
    input_size: int = Field(ge=128)
    state: dict[str, Any] = Field(default_factory=dict)


class DetectorTrainingSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    epochs_completed: int = Field(ge=1)
    best_epoch: int = Field(ge=1)
    stopped_early: bool
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    best_metrics: DetectorMetrics
    test_metrics: DetectorMetrics
    history: tuple[DetectorMetrics, ...]


class PlateDetectorProtocol(Protocol):
    def detect(
        self,
        image_bytes: bytes,
        image_width: int,
        image_height: int,
    ) -> list[DetectionPrediction]: ...


class YoloBackendProtocol(Protocol):
    def train_epoch(
        self,
        epoch: int,
        learning_rate: float,
        dataset_yaml: Path,
        config: DetectorConfig,
    ) -> DetectorMetrics: ...

    def evaluate(
        self,
        dataset_yaml: Path,
        split: str,
        config: DetectorConfig,
    ) -> DetectorMetrics: ...

    def get_state(self) -> dict[str, Any]: ...

    def load_state(self, state: dict[str, Any]) -> None: ...


__all__ = [
    "DetectionPrediction",
    "DetectorCheckpoint",
    "DetectorMetrics",
    "DetectorTrainingSummary",
    "PlateDetectorProtocol",
    "YoloBackendProtocol",
]
