from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvironmentType(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    data_dir: Path = Field(default=Path("data"))
    raw_data_dir: Path = Field(default=Path("data/raw"))
    processed_data_dir: Path = Field(default=Path("data/processed"))
    artifacts_dir: Path = Field(default=Path("artifacts"))
    checkpoints_dir: Path = Field(default=Path("checkpoints"))
    logs_dir: Path = Field(default=Path("logs"))


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    level: LogLevel = Field(default=LogLevel.INFO)
    json_format: bool = Field(default=True)
    service_name: str = Field(default="plate-vision", min_length=1)
    include_timestamp: bool = Field(default=True)


class DatasetSplitRatios(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    train: float = Field(default=0.70, gt=0.0, lt=1.0)
    val: float = Field(default=0.15, gt=0.0, lt=1.0)
    test: float = Field(default=0.15, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_total_ratio(self) -> Self:
        total = self.train + self.val + self.test
        tolerance = 1e-6
        if abs(total - 1.0) > tolerance:
            msg = f"Split ratios must sum to 1.0, received {total:.6f}"
            raise ValueError(msg)
        return self


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = Field(default=42, ge=0)
    country_profile: str = Field(default="TR", min_length=2)
    split_ratios: DatasetSplitRatios = Field(default_factory=DatasetSplitRatios)
    min_image_width: int = Field(default=32, ge=8)
    min_image_height: int = Field(default=16, ge=8)
    max_image_bytes: int = Field(default=10_485_760, ge=1024)
    num_workers: int = Field(default=4, ge=0)


class DetectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_architecture: str = Field(default="yolo11n.pt", min_length=1)
    input_size: int = Field(default=640, ge=128)
    confidence_threshold: float = Field(default=0.25, gt=0.0, lt=1.0)
    iou_threshold: float = Field(default=0.45, gt=0.0, lt=1.0)
    max_detections: int = Field(default=10, ge=1)
    epochs: int = Field(default=50, ge=1)
    batch_size: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=0.001, gt=0.0)
    weight_decay: float = Field(default=0.0005, ge=0.0)
    warmup_epochs: int = Field(default=3, ge=0)
    early_stopping_patience: int = Field(default=10, ge=1)
    mixed_precision: bool = Field(default=True)
    device: str = Field(default="cpu", min_length=1)

    @model_validator(mode="after")
    def validate_input_size_multiple(self) -> Self:
        stride = 32
        if self.input_size % stride != 0:
            msg = f"Detector input_size must be divisible by {stride}, received {self.input_size}"
            raise ValueError(msg)
        return self


class RecognizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    image_height: int = Field(default=32, ge=16)
    image_width: int = Field(default=128, ge=32)
    num_channels: int = Field(default=1, ge=1, le=3)
    hidden_size: int = Field(default=256, ge=32)
    num_lstm_layers: int = Field(default=2, ge=1)
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    beam_width: int = Field(default=5, ge=1)
    epochs: int = Field(default=60, ge=1)
    batch_size: int = Field(default=64, ge=1)
    learning_rate: float = Field(default=0.001, gt=0.0)
    weight_decay: float = Field(default=0.0001, ge=0.0)
    grad_clip_norm: float = Field(default=5.0, gt=0.0)
    warmup_epochs: int = Field(default=5, ge=0)
    early_stopping_patience: int = Field(default=10, ge=1)
    mixed_precision: bool = Field(default=True)
    device: str = Field(default="cpu", min_length=1)


class ServingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    host: str = Field(default="0.0.0.0", min_length=1)
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1)
    timeout_seconds: float = Field(default=5.0, gt=0.0)
    max_payload_bytes: int = Field(default=10_485_760, ge=1024)
    max_batch_size: int = Field(default=16, ge=1)
    allowed_mime_types: tuple[str, ...] = Field(
        default=("image/jpeg", "image/png", "image/webp"),
        min_length=1,
    )
    detector_onnx_path: Path = Field(default=Path("artifacts/models/detector.onnx"))
    recognizer_onnx_path: Path = Field(default=Path("artifacts/models/recognizer.onnx"))
    enable_metrics: bool = Field(default=True)
    rate_limit_requests_per_minute: int = Field(default=600, ge=1)
    api_keys: tuple[str, ...] = Field(default=())


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PLATE_VISION_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    environment: EnvironmentType = Field(default=EnvironmentType.DEVELOPMENT)
    seed: int = Field(default=42, ge=0)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    detection: DetectorConfig = Field(default_factory=DetectorConfig)
    recognition: RecognizerConfig = Field(default_factory=RecognizerConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)


__all__ = [
    "AppSettings",
    "DataConfig",
    "DatasetSplitRatios",
    "DetectorConfig",
    "EnvironmentType",
    "LogLevel",
    "LoggingConfig",
    "PathsConfig",
    "RecognizerConfig",
    "ServingConfig",
]
