import math
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from plate_vision.data.profiles import get_country_profile
from plate_vision.pipeline.watchlist import WatchlistCategory, WatchlistMatcher
from plate_vision.utils.exceptions import ConfigurationError, InvalidInputError


class GateDirection(StrEnum):
    ENTRY = "entry"
    EXIT = "exit"


class ParkingTariffConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    grace_period_minutes: float = Field(default=15.0, ge=0.0)
    hourly_rate: float = Field(default=50.0, ge=0.0)
    daily_max_rate: float = Field(default=400.0, ge=0.0)
    vip_free_parking: bool = Field(default=True)
    currency: str = Field(default="TRY", min_length=3, max_length=3)


class ParkingSession(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str = Field(min_length=1)
    plate_text: str = Field(min_length=2)
    country_code: str = Field(min_length=2)
    entry_timestamp_ms: int = Field(ge=0)
    exit_timestamp_ms: int | None = None
    duration_minutes: float = Field(default=0.0, ge=0.0)
    fee_amount: float = Field(default=0.0, ge=0.0)
    currency: str = Field(default="TRY", min_length=3, max_length=3)
    is_vip: bool = False
    is_active: bool = True


class GateAccessDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    open_barrier: bool
    reason: str
    session: ParkingSession | None = None


class ParkingLotManager:
    def __init__(
        self,
        tariff: ParkingTariffConfig | None = None,
        watchlist: WatchlistMatcher | None = None,
        *,
        capacity: int = 500,
    ) -> None:
        if capacity < 1:
            raise ConfigurationError(
                "capacity must be >= 1",
                details={"capacity": capacity},
            )
        self.tariff = tariff if tariff is not None else ParkingTariffConfig()
        self.watchlist = watchlist
        self.capacity = capacity
        self._active_sessions: dict[str, ParkingSession] = {}
        self._closed_sessions: list[ParkingSession] = []
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

    def record_entry(
        self,
        plate_text: str,
        *,
        country_code: str = "TR",
        timestamp_ms: int,
    ) -> GateAccessDecision:
        if timestamp_ms < 0:
            raise InvalidInputError(
                "timestamp_ms must be >= 0",
                details={"timestamp_ms": timestamp_ms},
            )
        code, norm = self._normalize_plate(plate_text, country_code)
        is_vip = False

        if self.watchlist is not None:
            matches = self.watchlist.check(norm, country_code=code)
            categories = {m.entry.category for m in matches}
            if WatchlistCategory.BLOCKLIST in categories or WatchlistCategory.STOLEN in categories:
                return GateAccessDecision(
                    open_barrier=False,
                    reason="watchlist_blocked",
                    session=None,
                )
            if WatchlistCategory.VIP in categories:
                is_vip = True

        key = f"{code}:{norm}"
        existing = self._active_sessions.get(key)
        if existing is not None:
            return GateAccessDecision(
                open_barrier=False,
                reason="already_parked",
                session=existing,
            )

        if len(self._active_sessions) >= self.capacity:
            return GateAccessDecision(
                open_barrier=False,
                reason="lot_full",
                session=None,
            )

        session = ParkingSession(
            session_id=f"sess_{self._next_seq:06d}",
            plate_text=norm,
            country_code=code,
            entry_timestamp_ms=timestamp_ms,
            currency=self.tariff.currency,
            is_vip=is_vip,
            is_active=True,
        )
        self._next_seq += 1
        self._active_sessions[key] = session
        return GateAccessDecision(
            open_barrier=True,
            reason="entry_granted",
            session=session,
        )

    def _compute_fee(self, duration_minutes: float, is_vip: bool) -> float:
        if is_vip and self.tariff.vip_free_parking:
            return 0.0
        if duration_minutes <= self.tariff.grace_period_minutes:
            return 0.0
        billable_hours = math.ceil(duration_minutes / 60.0)
        raw_fee = billable_hours * self.tariff.hourly_rate
        days = max(1, math.ceil(duration_minutes / 1440.0))
        capped = min(raw_fee, days * self.tariff.daily_max_rate)
        return round(capped, 2)

    def record_exit(
        self,
        plate_text: str,
        *,
        country_code: str = "TR",
        timestamp_ms: int,
    ) -> GateAccessDecision:
        if timestamp_ms < 0:
            raise InvalidInputError(
                "timestamp_ms must be >= 0",
                details={"timestamp_ms": timestamp_ms},
            )
        code, norm = self._normalize_plate(plate_text, country_code)
        key = f"{code}:{norm}"
        active = self._active_sessions.get(key)
        if active is None:
            return GateAccessDecision(
                open_barrier=False,
                reason="no_active_session",
                session=None,
            )
        if timestamp_ms < active.entry_timestamp_ms:
            raise InvalidInputError(
                "Exit timestamp cannot precede entry timestamp",
                details={
                    "entry_timestamp_ms": active.entry_timestamp_ms,
                    "exit_timestamp_ms": timestamp_ms,
                },
            )

        duration_minutes = (timestamp_ms - active.entry_timestamp_ms) / 60_000.0
        fee = self._compute_fee(duration_minutes, active.is_vip)
        closed = ParkingSession(
            session_id=active.session_id,
            plate_text=active.plate_text,
            country_code=active.country_code,
            entry_timestamp_ms=active.entry_timestamp_ms,
            exit_timestamp_ms=timestamp_ms,
            duration_minutes=round(duration_minutes, 4),
            fee_amount=fee,
            currency=active.currency,
            is_vip=active.is_vip,
            is_active=False,
        )
        del self._active_sessions[key]
        self._closed_sessions.append(closed)
        return GateAccessDecision(
            open_barrier=True,
            reason="exit_granted",
            session=closed,
        )

    @property
    def occupancy_count(self) -> int:
        return len(self._active_sessions)

    def total_revenue(self) -> float:
        return round(sum(s.fee_amount for s in self._closed_sessions), 2)

    def active_sessions(self) -> tuple[ParkingSession, ...]:
        return tuple(self._active_sessions.values())

    def closed_sessions(self) -> tuple[ParkingSession, ...]:
        return tuple(self._closed_sessions)


__all__ = [
    "GateAccessDecision",
    "GateDirection",
    "ParkingLotManager",
    "ParkingSession",
    "ParkingTariffConfig",
]
