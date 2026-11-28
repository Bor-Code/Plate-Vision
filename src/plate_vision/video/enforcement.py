from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from plate_vision.data.profiles import get_country_profile
from plate_vision.utils.exceptions import ConfigurationError, InvalidInputError


class ViolationType(StrEnum):
    RED_LIGHT = "red_light"
    BUS_LANE = "bus_lane"
    RESTRICTED_ZONE = "restricted_zone"
    WRONG_WAY = "wrong_way"
    SHOULDER_DRIVING = "shoulder_driving"


class SignalPhase(StrEnum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class EnforcementZoneRule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    zone_id: str = Field(min_length=1)
    violation_type: ViolationType
    fine_amount: float = Field(ge=0.0)
    currency: str = Field(default="TRY", min_length=3, max_length=3)
    red_light_grace_ms: int = Field(default=500, ge=0)
    cooldown_seconds: float = Field(default=60.0, ge=0.0)
    min_confidence: float = Field(default=0.75, ge=0.0, le=1.0)


class ViolationObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    zone_id: str = Field(min_length=1)
    plate_text: str = Field(min_length=2)
    country_code: str = Field(default="TR", min_length=2)
    timestamp_ms: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    frame_indices: tuple[int, ...] = (0,)
    signal_phase: SignalPhase | None = None
    signal_elapsed_ms: int = Field(default=0, ge=0)
    speed_kmh: float | None = Field(default=None, ge=0.0)


class TrafficViolationTicket(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ticket_id: str = Field(min_length=1)
    zone_id: str = Field(min_length=1)
    violation_type: ViolationType
    plate_text: str = Field(min_length=2)
    country_code: str = Field(min_length=2)
    timestamp_ms: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    frame_indices: tuple[int, ...]
    fine_amount: float = Field(ge=0.0)
    currency: str = Field(min_length=3, max_length=3)
    speed_kmh: float | None = None


class EnforcementSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    total_observations: int = Field(ge=0)
    tickets_issued: int = Field(ge=0)
    exempt_skipped: int = Field(ge=0)
    grace_or_compliant_skipped: int = Field(ge=0)
    low_confidence_skipped: int = Field(ge=0)
    cooldown_skipped: int = Field(ge=0)
    total_fine_amount: float = Field(ge=0.0)
    by_violation_type: dict[str, int]
    currency: str = Field(min_length=3, max_length=3)


class TrafficEnforcementEngine:
    def __init__(
        self,
        rules: Sequence[EnforcementZoneRule] | None = None,
        *,
        currency: str = "TRY",
    ) -> None:
        self.currency = currency.strip().upper()
        if len(self.currency) != 3:
            raise ConfigurationError(
                "currency must be a 3-letter code",
                details={"currency": currency},
            )
        self._rules: dict[str, EnforcementZoneRule] = {}
        self._exempt_plates: set[tuple[str, str, str | None]] = set()
        self._last_ticket_ms: dict[tuple[str, str, str, ViolationType], int] = {}
        self._tickets: list[TrafficViolationTicket] = []
        self._total_observations = 0
        self._exempt_skipped = 0
        self._grace_or_compliant_skipped = 0
        self._low_confidence_skipped = 0
        self._cooldown_skipped = 0
        self._next_seq = 1
        if rules is not None:
            for rule in rules:
                self.register_rule(rule)

    def register_rule(self, rule: EnforcementZoneRule) -> None:
        if rule.zone_id in self._rules:
            raise ConfigurationError(
                f"Duplicate enforcement zone_id: {rule.zone_id}",
                details={"zone_id": rule.zone_id},
            )
        self._rules[rule.zone_id] = rule

    @staticmethod
    def _normalize_plate(plate_text: str, country_code: str) -> tuple[str, str]:
        code = country_code.strip().upper()
        profile = get_country_profile(code)
        normalized = profile.normalize(plate_text)
        if len(normalized) < 2:
            raise InvalidInputError(
                "Normalized plate_text must contain at least 2 characters",
                details={"plate_text": plate_text, "country_code": code},
            )
        return code, normalized

    def add_exempt_plate(
        self,
        plate_text: str,
        *,
        country_code: str = "TR",
        zone_id: str | None = None,
    ) -> str:
        code, norm = self._normalize_plate(plate_text, country_code)
        self._exempt_plates.add((code, norm, zone_id))
        return norm

    def evaluate(self, observation: ViolationObservation) -> TrafficViolationTicket | None:
        rule = self._rules.get(observation.zone_id)
        if rule is None:
            raise ConfigurationError(
                f"Unknown enforcement zone_id: {observation.zone_id}",
                details={"zone_id": observation.zone_id},
            )
        code, norm = self._normalize_plate(observation.plate_text, observation.country_code)
        self._total_observations += 1

        if observation.confidence < rule.min_confidence:
            self._low_confidence_skipped += 1
            return None

        if (code, norm, None) in self._exempt_plates or (
            code,
            norm,
            observation.zone_id,
        ) in self._exempt_plates:
            self._exempt_skipped += 1
            return None

        if rule.violation_type == ViolationType.RED_LIGHT and (
            observation.signal_phase != SignalPhase.RED
            or observation.signal_elapsed_ms < rule.red_light_grace_ms
        ):
            self._grace_or_compliant_skipped += 1
            return None

        cooldown_key = (code, norm, observation.zone_id, rule.violation_type)
        prev_ms = self._last_ticket_ms.get(cooldown_key)
        cooldown_ms = int(rule.cooldown_seconds * 1000.0)
        if prev_ms is not None and 0 <= (observation.timestamp_ms - prev_ms) < cooldown_ms:
            self._cooldown_skipped += 1
            return None

        self._last_ticket_ms[cooldown_key] = observation.timestamp_ms
        ticket = TrafficViolationTicket(
            ticket_id=f"{observation.zone_id}_tkt_{self._next_seq:06d}",
            zone_id=observation.zone_id,
            violation_type=rule.violation_type,
            plate_text=norm,
            country_code=code,
            timestamp_ms=observation.timestamp_ms,
            confidence=observation.confidence,
            frame_indices=observation.frame_indices,
            fine_amount=round(rule.fine_amount, 2),
            currency=rule.currency,
            speed_kmh=observation.speed_kmh,
        )
        self._next_seq += 1
        self._tickets.append(ticket)
        return ticket

    def tickets(self) -> list[TrafficViolationTicket]:
        return list(self._tickets)

    def summary(self) -> EnforcementSummary:
        counts: dict[str, int] = {}
        total_fines = 0.0
        for ticket in self._tickets:
            vtype = ticket.violation_type.value
            counts[vtype] = counts.get(vtype, 0) + 1
            total_fines += ticket.fine_amount

        return EnforcementSummary(
            total_observations=self._total_observations,
            tickets_issued=len(self._tickets),
            exempt_skipped=self._exempt_skipped,
            grace_or_compliant_skipped=self._grace_or_compliant_skipped,
            low_confidence_skipped=self._low_confidence_skipped,
            cooldown_skipped=self._cooldown_skipped,
            total_fine_amount=round(total_fines, 2),
            by_violation_type=counts,
            currency=self.currency,
        )


__all__ = [
    "EnforcementSummary",
    "EnforcementZoneRule",
    "SignalPhase",
    "TrafficEnforcementEngine",
    "TrafficViolationTicket",
    "ViolationObservation",
    "ViolationType",
]
