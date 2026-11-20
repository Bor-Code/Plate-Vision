from collections import Counter
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from plate_vision.data.profiles import get_country_profile
from plate_vision.utils.exceptions import InvalidInputError


class TerminalZoneType(StrEnum):
    YARD = "yard"
    LOADING_DOCK = "loading_dock"
    WEIGHBRIDGE = "weighbridge"
    STAGING = "staging"


class FleetSlaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    facility_id: str = Field(default="LOGISTICS_HUB_01", min_length=1)
    max_turnaround_minutes: float = Field(default=60.0, gt=0.0)
    demurrage_rate_per_hour: float = Field(default=250.0, ge=0.0)
    currency: str = Field(default="TRY", min_length=3, max_length=3)


class FleetCheckInObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    facility_id: str = Field(min_length=1)
    plate_text: str = Field(min_length=2)
    country_code: str = Field(min_length=2)
    carrier_name: str = Field(min_length=1)
    assigned_zone: TerminalZoneType
    arrival_timestamp_ms: int = Field(ge=0)


class FleetTurnaroundRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    visit_id: str = Field(min_length=1)
    facility_id: str = Field(min_length=1)
    plate_text: str = Field(min_length=2)
    country_code: str = Field(min_length=2)
    carrier_name: str = Field(min_length=1)
    assigned_zone: TerminalZoneType
    arrival_timestamp_ms: int = Field(ge=0)
    departure_timestamp_ms: int = Field(ge=0)
    turnaround_minutes: float = Field(ge=0.0)
    sla_breached: bool
    overstay_minutes: float = Field(ge=0.0)
    demurrage_fee: float = Field(ge=0.0)
    currency: str = Field(min_length=3, max_length=3)


class FleetOperationsSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    facility_id: str = Field(min_length=1)
    active_vehicles: int = Field(ge=0)
    completed_visits: int = Field(ge=0)
    sla_breach_count: int = Field(ge=0)
    sla_compliance_rate: float = Field(ge=0.0, le=1.0)
    mean_turnaround_minutes: float = Field(ge=0.0)
    total_demurrage_fee: float = Field(ge=0.0)
    visits_by_carrier: dict[str, int]
    currency: str = Field(min_length=3, max_length=3)


class FleetTerminalController:
    def __init__(self, config: FleetSlaConfig | None = None) -> None:
        self.config = config if config is not None else FleetSlaConfig()
        self._active: dict[str, FleetCheckInObservation] = {}
        self._completed: list[FleetTurnaroundRecord] = []
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

    def check_in(
        self,
        plate_text: str,
        *,
        country_code: str = "TR",
        carrier_name: str = "DEFAULT_CARRIER",
        assigned_zone: TerminalZoneType = TerminalZoneType.LOADING_DOCK,
        timestamp_ms: int,
    ) -> FleetCheckInObservation:
        if timestamp_ms < 0:
            raise InvalidInputError(
                "timestamp_ms must be >= 0",
                details={"timestamp_ms": timestamp_ms},
            )
        clean_carrier = carrier_name.strip()
        if not clean_carrier:
            raise InvalidInputError("carrier_name must not be empty")
        code, norm = self._normalize_plate(plate_text, country_code)
        key = f"{code}:{norm}"
        if key in self._active:
            raise InvalidInputError(
                "Vehicle is already checked in at terminal",
                details={"plate_text": norm, "country_code": code},
            )
        observation = FleetCheckInObservation(
            facility_id=self.config.facility_id,
            plate_text=norm,
            country_code=code,
            carrier_name=clean_carrier,
            assigned_zone=assigned_zone,
            arrival_timestamp_ms=timestamp_ms,
        )
        self._active[key] = observation
        return observation

    def check_out(
        self,
        plate_text: str,
        *,
        country_code: str = "TR",
        timestamp_ms: int,
    ) -> FleetTurnaroundRecord:
        if timestamp_ms < 0:
            raise InvalidInputError(
                "timestamp_ms must be >= 0",
                details={"timestamp_ms": timestamp_ms},
            )
        code, norm = self._normalize_plate(plate_text, country_code)
        key = f"{code}:{norm}"
        entry = self._active.get(key)
        if entry is None:
            raise InvalidInputError(
                "No active fleet check-in found for vehicle",
                details={"plate_text": norm, "country_code": code},
            )
        if timestamp_ms < entry.arrival_timestamp_ms:
            raise InvalidInputError(
                "Departure timestamp cannot precede arrival timestamp",
                details={
                    "arrival_timestamp_ms": entry.arrival_timestamp_ms,
                    "departure_timestamp_ms": timestamp_ms,
                },
            )

        del self._active[key]
        raw_minutes = (timestamp_ms - entry.arrival_timestamp_ms) / 60_000.0
        turnaround_minutes = round(raw_minutes, 4)
        sla_breached = raw_minutes > self.config.max_turnaround_minutes
        raw_overstay = max(0.0, raw_minutes - self.config.max_turnaround_minutes)
        overstay_minutes = round(raw_overstay, 4)
        demurrage_fee = (
            round((raw_overstay / 60.0) * self.config.demurrage_rate_per_hour, 2)
            if sla_breached
            else 0.0
        )

        record = FleetTurnaroundRecord(
            visit_id=f"visit_{self._next_seq:06d}",
            facility_id=self.config.facility_id,
            plate_text=norm,
            country_code=code,
            carrier_name=entry.carrier_name,
            assigned_zone=entry.assigned_zone,
            arrival_timestamp_ms=entry.arrival_timestamp_ms,
            departure_timestamp_ms=timestamp_ms,
            turnaround_minutes=turnaround_minutes,
            sla_breached=sla_breached,
            overstay_minutes=overstay_minutes,
            demurrage_fee=demurrage_fee,
            currency=self.config.currency,
        )
        self._next_seq += 1
        self._completed.append(record)
        return record

    def active_vehicles(self) -> tuple[FleetCheckInObservation, ...]:
        return tuple(self._active.values())

    def completed_records(self) -> tuple[FleetTurnaroundRecord, ...]:
        return tuple(self._completed)

    def summarize(self) -> FleetOperationsSummary:
        total = len(self._completed)
        if total == 0:
            return FleetOperationsSummary(
                facility_id=self.config.facility_id,
                active_vehicles=len(self._active),
                completed_visits=0,
                sla_breach_count=0,
                sla_compliance_rate=0.0,
                mean_turnaround_minutes=0.0,
                total_demurrage_fee=0.0,
                visits_by_carrier={},
                currency=self.config.currency,
            )

        breaches = sum(1 for item in self._completed if item.sla_breached)
        compliance_rate = round((total - breaches) / float(total), 6)
        mean_turnaround = round(
            sum(item.turnaround_minutes for item in self._completed) / float(total),
            4,
        )
        total_demurrage = round(sum(item.demurrage_fee for item in self._completed), 2)
        carrier_counts: Counter[str] = Counter(item.carrier_name for item in self._completed)
        return FleetOperationsSummary(
            facility_id=self.config.facility_id,
            active_vehicles=len(self._active),
            completed_visits=total,
            sla_breach_count=breaches,
            sla_compliance_rate=compliance_rate,
            mean_turnaround_minutes=mean_turnaround,
            total_demurrage_fee=total_demurrage,
            visits_by_carrier=dict(sorted(carrier_counts.items())),
            currency=self.config.currency,
        )


__all__ = [
    "FleetCheckInObservation",
    "FleetOperationsSummary",
    "FleetSlaConfig",
    "FleetTerminalController",
    "FleetTurnaroundRecord",
    "TerminalZoneType",
]
