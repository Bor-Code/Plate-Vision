from enum import StrEnum
from pathlib import Path
from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class BoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    x_min: float = Field(ge=0.0)
    y_min: float = Field(ge=0.0)
    x_max: float = Field(gt=0.0)
    y_max: float = Field(gt=0.0)
    class_id: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_coordinates(self) -> Self:
        if self.x_max <= self.x_min:
            msg = f"x_max ({self.x_max}) must be greater than x_min ({self.x_min})"
            raise ValueError(msg)
        if self.y_max <= self.y_min:
            msg = f"y_max ({self.y_max}) must be greater than y_min ({self.y_min})"
            raise ValueError(msg)
        return self

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    def to_yolo(
        self,
        image_width: int,
        image_height: int,
    ) -> tuple[int, float, float, float, float]:
        if image_width <= 0 or image_height <= 0:
            msg = "image_width and image_height must be positive integers"
            raise ValueError(msg)
        x_center = ((self.x_min + self.x_max) / 2.0) / float(image_width)
        y_center = ((self.y_min + self.y_max) / 2.0) / float(image_height)
        norm_width = self.width / float(image_width)
        norm_height = self.height / float(image_height)
        return (self.class_id, x_center, y_center, norm_width, norm_height)

    def format_yolo_line(self, image_width: int, image_height: int) -> str:
        class_id, x_center, y_center, norm_w, norm_h = self.to_yolo(
            image_width=image_width,
            image_height=image_height,
        )
        return f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"


class PlateAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plate_text: str = Field(min_length=1)
    bbox: BoundingBox
    country_code: str = Field(default="TR", min_length=2)
    corners: (
        tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ]
        | None
    ) = None

    @model_validator(mode="after")
    def validate_plate_text(self) -> Self:
        normalized = self.plate_text.strip()
        if not normalized:
            msg = "plate_text must not be blank"
            raise ValueError(msg)
        return self


class ImageMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    size_bytes: int = Field(ge=1)
    sha256: str = Field(min_length=64, max_length=64)
    mime_type: str = Field(min_length=1)


class DatasetRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str = Field(min_length=1)
    image: ImageMetadata
    annotations: tuple[PlateAnnotation, ...] = Field(default=())
    source_dataset: str = Field(default="default", min_length=1)
    stratum_key: str = Field(default="default", min_length=1)


class SplitDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    train: tuple[DatasetRecord, ...] = Field(default=())
    val: tuple[DatasetRecord, ...] = Field(default=())
    test: tuple[DatasetRecord, ...] = Field(default=())

    def for_split(self, split: DatasetSplit) -> tuple[DatasetRecord, ...]:
        if split is DatasetSplit.TRAIN:
            return self.train
        if split is DatasetSplit.VAL:
            return self.val
        return self.test


class DeduplicationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    unique_records: tuple[DatasetRecord, ...]
    duplicate_record_ids: tuple[str, ...]
    removed_count: int = Field(ge=0)


class DatasetStorageProtocol(Protocol):
    def read_bytes(self, path: Path) -> bytes: ...

    def write_text(self, path: Path, content: str) -> None: ...


__all__ = [
    "BoundingBox",
    "DatasetRecord",
    "DatasetSplit",
    "DatasetStorageProtocol",
    "DeduplicationResult",
    "ImageMetadata",
    "PlateAnnotation",
    "SplitDataset",
]
