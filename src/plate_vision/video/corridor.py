from pydantic import BaseModel, ConfigDict, Field

from plate_vision.data.profiles import get_country_profile
from plate_vision.utils.exceptions import InvalidInputError


class CorridorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    corridor_id: str = Field(default="CORRIDOR_01", min_length=1)
    distance_km: float = Field(default=10.0, gt=0.0)
    speed_limit_kmh: float = Field(default=120.0, gt=0.0)
    tolerance_percent: float = Field(default=10.0, ge=0.0, le=100.0)
    toll_rate_per_km: float = Field(default=2.5, ge=0.0)
    speeding_fine_amount: float = Field(default=1500.0, ge=0.0)
    currency: str = Field(default="TRY", min_length=3, max_length=3)


class CorridorEntryObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    corridor_id: str = Field(min_length=1)
    plate_text: str = Field(min_length=2)
    country_code: str = Field(min_length=2)
    entry_checkpoint_id: str = Field(min_length=1)
    entry_timestamp_ms: int = Field(ge=0)


class CorridorPassageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passage_id: str = Field(min_length=1)
    corridor_id: str = Field(min_length=1)
    plate_text: str = Field(min_length=2)
    country_code: str = Field(min_length=2)
    entry_checkpoint_id: str = Field(min_length=1)
    exit_checkpoint_id: str = Field(min_length=1)
    entry_timestamp_ms: int = Field(ge=0)
    exit_timestamp_ms: int = Field(ge=0)
    travel_time_seconds: float = Field(gt=0.0)
    average_speed_kmh: float = Field(ge=0.0)
    effective_speed_limit_kmh: float = Field(gt=0.0)
    is_speed_violation: bool
    toll_amount: float = Field(ge=0.0)
    fine_amount: float = Field(ge=0.0)
    total_charge: float = Field(ge=0.0)
    currency: str = Field(min_length=3, max_length=3)


class CorridorAnalyticsSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    corridor_id: str = Field(min_length=1)
    total_passages: int = Field(ge=0)
    active_in_corridor: int = Field(ge=0)
    violation_count: int = Field(ge=0)
    violation_rate: float = Field(ge=0.0, le=1.0)
    mean_speed_kmh: float = Field(ge=0.0)
    max_speed_kmh: float = Field(ge=0.0)
    total_toll_revenue: float = Field(ge=0.0)
    total_fine_revenue: float = Field(ge=0.0)
    currency: str = Field(min_length=3, max_length=3)


class HighwayCorridorMonitor:
    def __init__(self, config: CorridorConfig | None = None) -> None:
        self.config = config if config is not None else CorridorConfig()
        self._active_entries: dict[str, CorridorEntryObservation] = {}
        self._passages: list[CorridorPassageRecord] = []
        self._next_seq = 1

    @staticmethod
    def _normalize_plate(plate_text: str, country_code: str) -> tuple[str, str]:
        code = country_code.strip().upper()
        profile = get_country_profile(code)
        norm = profile.normalize(plate_text)
        if len(norm) < 2:
            raise InvalidInputError(
                "Normalized plate_text must contain at least 2 characters",
                details={"plate_text": plate_text, "country_code": code},
            )
        return code, norm

    @staticmethod
    def _validate_checkpoint_and_timestamp(checkpoint_id: str, timestamp_ms: int) -> str:
        clean_cp = checkpoint_id.strip()
        if not clean_cp:
            raise InvalidInputError("checkpoint_id must not be empty")
        if timestamp_ms < 0:
            raise InvalidInputError(
                "timestamp_ms must be >= 0",
                details={"timestamp_ms": timestamp_ms},
            )
        return clean_cp

    def record_entry(
        self,
        plate_text: str,
        *,
        country_code: str = "TR",
        checkpoint_id: str = "CP_ENTRY",
        timestamp_ms: int,
    ) -> CorridorEntryObservation:
        clean_cp = self._validate_checkpoint_and_timestamp(checkpoint_id, timestamp_ms)
        code, norm = self._normalize_plate(plate_text, country_code)
        key = f"{code}:{norm}"
        observation = CorridorEntryObservation(
            corridor_id=self.config.corridor_id,
            plate_text=norm,
            country_code=code,
            entry_checkpoint_id=clean_cp,
            entry_timestamp_ms=timestamp_ms,
        )
        self._active_entries[key] = observation
        return observation

    def record_exit(
        self,
        plate_text: str,
        *,
        country_code: str = "TR",
        checkpoint_id: str = "CP_EXIT",
        timestamp_ms: int,
    ) -> CorridorPassageRecord:
        clean_cp = self._validate_checkpoint_and_timestamp(checkpoint_id, timestamp_ms)
        code, norm = self._normalize_plate(plate_text, country_code)
        key = f"{code}:{norm}"
        entry = self._active_entries.get(key)
        if entry is None:
            raise InvalidInputError(
                "No active corridor entry found for plate",
                details={"plate_text": norm, "country_code": code},
            )
        if timestamp_ms <= entry.entry_timestamp_ms:
            raise InvalidInputError(
                "Exit timestamp must exceed entry timestamp",
                details={
                    "entry_timestamp_ms": entry.entry_timestamp_ms,
                    "exit_timestamp_ms": timestamp_ms,
                },
            )

        del self._active_entries[key]
        travel_time_seconds = round((timestamp_ms - entry.entry_timestamp_ms) / 1000.0, 4)
        raw_travel_seconds = (timestamp_ms - entry.entry_timestamp_ms) / 1000.0
        average_speed_kmh = round((self.config.distance_km / raw_travel_seconds) * 3600.0, 4)
        effective_limit_kmh = round(
            self.config.speed_limit_kmh * (1.0 + (self.config.tolerance_percent / 100.0)),
            4,
        )
        is_violation = average_speed_kmh > effective_limit_kmh
        toll_amount = round(self.config.distance_km * self.config.toll_rate_per_km, 2)
        fine_amount = round(self.config.speeding_fine_amount, 2) if is_violation else 0.0
        total_charge = round(toll_amount + fine_amount, 2)

        record = CorridorPassageRecord(
            passage_id=f"pass_{self._next_seq:06d}",
            corridor_id=self.config.corridor_id,
            plate_text=norm,
            country_code=code,
            entry_checkpoint_id=entry.entry_checkpoint_id,
            exit_checkpoint_id=clean_cp,
            entry_timestamp_ms=entry.entry_timestamp_ms,
            exit_timestamp_ms=timestamp_ms,
            travel_time_seconds=travel_time_seconds,
            average_speed_kmh=average_speed_kmh,
            effective_speed_limit_kmh=effective_limit_kmh,
            is_speed_violation=is_violation,
            toll_amount=toll_amount,
            fine_amount=fine_amount,
            total_charge=total_charge,
            currency=self.config.currency,
        )
        self._next_seq += 1
        self._passages.append(record)
        return record

    def active_entries(self) -> tuple[CorridorEntryObservation, ...]:
        return tuple(self._active_entries.values())

    def passages(self) -> tuple[CorridorPassageRecord, ...]:
        return tuple(self._passages)

    def summarize(self) -> CorridorAnalyticsSummary:
        total = len(self._passages)
        if total == 0:
            return CorridorAnalyticsSummary(
                corridor_id=self.config.corridor_id,
                total_passages=0,
                active_in_corridor=len(self._active_entries),
                violation_count=0,
                violation_rate=0.0,
                mean_speed_kmh=0.0,
                max_speed_kmh=0.0,
                total_toll_revenue=0.0,
                total_fine_revenue=0.0,
                currency=self.config.currency,
            )

        violation_count = sum(1 for item in self._passages if item.is_speed_violation)
        violation_rate = round(violation_count / float(total), 6)
        mean_speed = round(sum(item.average_speed_kmh for item in self._passages) / float(total), 4)
        max_speed = max(item.average_speed_kmh for item in self._passages)
        total_toll = round(sum(item.toll_amount for item in self._passages), 2)
        total_fine = round(sum(item.fine_amount for item in self._passages), 2)
        return CorridorAnalyticsSummary(
            corridor_id=self.config.corridor_id,
            total_passages=total,
            active_in_corridor=len(self._active_entries),
            violation_count=violation_count,
            violation_rate=violation_rate,
            mean_speed_kmh=mean_speed,
            max_speed_kmh=max_speed,
            total_toll_revenue=total_toll,
            total_fine_revenue=total_fine,
            currency=self.config.currency,
        )


__all__ = [
    "CorridorAnalyticsSummary",
    "CorridorConfig",
    "CorridorEntryObservation",
    "CorridorPassageRecord",
    "HighwayCorridorMonitor",
]
