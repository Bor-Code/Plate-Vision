import hashlib
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Generic, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from plate_vision.utils.exceptions import ConfigurationError, DataIOError

T = TypeVar("T")


def compute_payload_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class PredictionAuditRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    payload_sha256: str = Field(min_length=64, max_length=64)
    country_code: str = Field(min_length=2)
    plate_texts: tuple[str, ...]
    confidences: tuple[float, ...]
    latency_ms: float = Field(ge=0.0)
    cached: bool
    timestamp_epoch_ms: int = Field(ge=0)


@runtime_checkable
class PredictionStorageProtocol(Protocol):
    def save(self, record: PredictionAuditRecord) -> None: ...

    def list_records(self, limit: int | None = None) -> tuple[PredictionAuditRecord, ...]: ...


class InMemoryPredictionStore:
    def __init__(self, max_records: int = 10_000) -> None:
        if max_records < 1:
            raise ConfigurationError(
                "max_records must be >= 1",
                details={"max_records": max_records},
            )
        self.max_records = max_records
        self._records: list[PredictionAuditRecord] = []

    def save(self, record: PredictionAuditRecord) -> None:
        if len(self._records) >= self.max_records:
            self._records.pop(0)
        self._records.append(record)

    def list_records(self, limit: int | None = None) -> tuple[PredictionAuditRecord, ...]:
        if limit is None:
            return tuple(self._records)
        if limit <= 0:
            return ()
        return tuple(self._records[-limit:])


class LocalJsonlPredictionStore:
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path

    def save(self, record: PredictionAuditRecord) -> None:
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            serialized = json.dumps(record.model_dump(mode="json"), sort_keys=True)
            with self.file_path.open("a", encoding="utf-8") as handle:
                handle.write(serialized + "\n")
        except OSError as exc:
            raise DataIOError(
                f"Failed to write prediction audit record to {self.file_path}",
                details={"path": str(self.file_path), "reason": str(exc)},
            ) from exc

    def list_records(self, limit: int | None = None) -> tuple[PredictionAuditRecord, ...]:
        if not self.file_path.exists():
            return ()
        try:
            lines = [
                line.strip()
                for line in self.file_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except OSError as exc:
            raise DataIOError(
                f"Failed to read prediction audit records from {self.file_path}",
                details={"path": str(self.file_path), "reason": str(exc)},
            ) from exc
        records = [PredictionAuditRecord.model_validate(json.loads(line)) for line in lines]
        if limit is None:
            return tuple(records)
        if limit <= 0:
            return ()
        return tuple(records[-limit:])


class PredictionCache(Generic[T]):
    def __init__(self, max_entries: int = 256) -> None:
        if max_entries < 1:
            raise ConfigurationError(
                "max_entries must be >= 1",
                details={"max_entries": max_entries},
            )
        self.max_entries = max_entries
        self._entries: OrderedDict[str, T] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> T | None:
        if key not in self._entries:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return self._entries[key]

    def put(self, key: str, value: T) -> None:
        if key in self._entries:
            self._entries.move_to_end(key)
        self._entries[key] = value
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()
        self.hits = 0
        self.misses = 0

    @property
    def size(self) -> int:
        return len(self._entries)


def build_audit_record(
    payload_bytes: bytes,
    *,
    country_code: str,
    plate_texts: tuple[str, ...],
    confidences: tuple[float, ...],
    latency_ms: float,
    cached: bool,
) -> PredictionAuditRecord:
    return PredictionAuditRecord(
        payload_sha256=compute_payload_sha256(payload_bytes),
        country_code=country_code,
        plate_texts=plate_texts,
        confidences=confidences,
        latency_ms=round(max(0.0, latency_ms), 4),
        cached=cached,
        timestamp_epoch_ms=int(time.time() * 1000),
    )


__all__ = [
    "InMemoryPredictionStore",
    "LocalJsonlPredictionStore",
    "PredictionAuditRecord",
    "PredictionCache",
    "PredictionStorageProtocol",
    "build_audit_record",
    "compute_payload_sha256",
]
