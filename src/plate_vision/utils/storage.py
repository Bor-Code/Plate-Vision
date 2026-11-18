import hashlib
import json
import sqlite3
import time
from collections import OrderedDict
from contextlib import closing
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


class SqlitePredictionStore:
    def __init__(self, db_path: Path, max_records: int = 100_000) -> None:
        if max_records < 1:
            raise ConfigurationError(
                "max_records must be >= 1",
                details={"max_records": max_records},
            )
        self.db_path = db_path
        self.max_records = max_records
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self.db_path)) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS prediction_audit ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "payload_sha256 TEXT NOT NULL, "
                    "country_code TEXT NOT NULL, "
                    "plate_texts_json TEXT NOT NULL, "
                    "confidences_json TEXT NOT NULL, "
                    "latency_ms REAL NOT NULL, "
                    "cached INTEGER NOT NULL, "
                    "timestamp_epoch_ms INTEGER NOT NULL"
                    ");"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_prediction_audit_country "
                    "ON prediction_audit(country_code, id);"
                )
                conn.commit()
        except (OSError, sqlite3.Error) as exc:
            raise DataIOError(
                f"Failed to initialize SQLite audit store at {self.db_path}",
                details={"path": str(self.db_path), "reason": str(exc)},
            ) from exc

    @staticmethod
    def _row_to_record(row: tuple[object, ...]) -> PredictionAuditRecord:
        return PredictionAuditRecord(
            payload_sha256=str(row[0]),
            country_code=str(row[1]),
            plate_texts=tuple(str(item) for item in json.loads(str(row[2]))),
            confidences=tuple(float(item) for item in json.loads(str(row[3]))),
            latency_ms=float(str(row[4])),
            cached=bool(int(str(row[5]))),
            timestamp_epoch_ms=int(str(row[6])),
        )

    def save(self, record: PredictionAuditRecord) -> None:
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                conn.execute(
                    "INSERT INTO prediction_audit ("
                    "payload_sha256, country_code, plate_texts_json, confidences_json, "
                    "latency_ms, cached, timestamp_epoch_ms"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?);",
                    (
                        record.payload_sha256,
                        record.country_code,
                        json.dumps(list(record.plate_texts)),
                        json.dumps(list(record.confidences)),
                        record.latency_ms,
                        int(record.cached),
                        record.timestamp_epoch_ms,
                    ),
                )
                count_row = conn.execute("SELECT COUNT(*) FROM prediction_audit;").fetchone()
                total = int(count_row[0]) if count_row is not None else 0
                if total > self.max_records:
                    excess = total - self.max_records
                    conn.execute(
                        "DELETE FROM prediction_audit WHERE id IN ("
                        "SELECT id FROM prediction_audit ORDER BY id ASC LIMIT ?"
                        ");",
                        (excess,),
                    )
                conn.commit()
        except sqlite3.Error as exc:
            raise DataIOError(
                f"Failed to save prediction audit record to {self.db_path}",
                details={"path": str(self.db_path), "reason": str(exc)},
            ) from exc

    def list_records(self, limit: int | None = None) -> tuple[PredictionAuditRecord, ...]:
        if limit is not None and limit <= 0:
            return ()
        base_cols = (
            "payload_sha256, country_code, plate_texts_json, confidences_json, "
            "latency_ms, cached, timestamp_epoch_ms"
        )
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                if limit is None:
                    rows = conn.execute(
                        f"SELECT {base_cols} FROM prediction_audit ORDER BY id ASC;"
                    ).fetchall()
                else:
                    rows = list(
                        reversed(
                            conn.execute(
                                f"SELECT {base_cols} FROM prediction_audit "
                                "ORDER BY id DESC LIMIT ?;",
                                (limit,),
                            ).fetchall()
                        )
                    )
        except sqlite3.Error as exc:
            raise DataIOError(
                f"Failed to query prediction audit records from {self.db_path}",
                details={"path": str(self.db_path), "reason": str(exc)},
            ) from exc
        return tuple(self._row_to_record(row) for row in rows)

    def query_by_country(
        self,
        country_code: str,
        limit: int = 50,
    ) -> tuple[PredictionAuditRecord, ...]:
        if limit <= 0:
            return ()
        base_cols = (
            "payload_sha256, country_code, plate_texts_json, confidences_json, "
            "latency_ms, cached, timestamp_epoch_ms"
        )
        target_code = country_code.strip().upper()
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                rows = list(
                    reversed(
                        conn.execute(
                            f"SELECT {base_cols} FROM prediction_audit "
                            "WHERE country_code = ? ORDER BY id DESC LIMIT ?;",
                            (target_code, limit),
                        ).fetchall()
                    )
                )
        except sqlite3.Error as exc:
            raise DataIOError(
                f"Failed to query prediction audit records by country from {self.db_path}",
                details={
                    "path": str(self.db_path),
                    "country_code": target_code,
                    "reason": str(exc),
                },
            ) from exc
        return tuple(self._row_to_record(row) for row in rows)

    def count(self) -> int:
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                row = conn.execute("SELECT COUNT(*) FROM prediction_audit;").fetchone()
                return int(row[0]) if row is not None else 0
        except sqlite3.Error as exc:
            raise DataIOError(
                f"Failed to count prediction audit records in {self.db_path}",
                details={"path": str(self.db_path), "reason": str(exc)},
            ) from exc


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
    "SqlitePredictionStore",
    "build_audit_record",
    "compute_payload_sha256",
]
