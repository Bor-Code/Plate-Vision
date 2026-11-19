from collections import Counter
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from plate_vision.data.profiles import get_country_profile
from plate_vision.utils.exceptions import InvalidInputError


class EmissionClass(StrEnum):
    ZERO_EMISSION = "zero_emission"
    EURO_6 = "euro_6"
    EURO_5 = "euro_5"
    EURO_4 = "euro_4"
    EURO_3_OR_LOWER = "euro_3_or_lower"
    UNREGISTERED = "unregistered"


class VehicleEmissionProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plate_text: str = Field(min_length=2)
    country_code: str = Field(default="TR", min_length=2)
    emission_class: EmissionClass
    is_exempt: bool = False


class LowEmissionZoneConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    zone_id: str = Field(default="LEZ_CENTER_01", min_length=1)
    allowed_classes: tuple[EmissionClass, ...] = (
        EmissionClass.ZERO_EMISSION,
        EmissionClass.EURO_6,
        EmissionClass.EURO_5,
    )
    enforce_odd_even_days: bool = False
    non_compliant_fine_amount: float = Field(default=2000.0, ge=0.0)
    unregistered_fine_amount: float = Field(default=2500.0, ge=0.0)
    odd_even_fine_amount: float = Field(default=1000.0, ge=0.0)
    currency: str = Field(default="TRY", min_length=3, max_length=3)


class LezInspectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    inspection_id: str = Field(min_length=1)
    zone_id: str = Field(min_length=1)
    plate_text: str = Field(min_length=2)
    country_code: str = Field(min_length=2)
    day_of_month: int = Field(ge=1, le=31)
    emission_class: EmissionClass
    is_exempt: bool
    is_compliant: bool
    violation_reason: str | None = None
    fine_amount: float = Field(ge=0.0)
    currency: str = Field(min_length=3, max_length=3)


class LezEnforcementSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    zone_id: str = Field(min_length=1)
    total_inspections: int = Field(ge=0)
    compliant_count: int = Field(ge=0)
    violation_count: int = Field(ge=0)
    compliance_rate: float = Field(ge=0.0, le=1.0)
    total_fine_amount: float = Field(ge=0.0)
    violations_by_reason: dict[str, int]
    currency: str = Field(min_length=3, max_length=3)


def _find_last_digit(normalized_plate: str) -> int | None:
    for ch in reversed(normalized_plate):
        if ch.isascii() and ch.isdigit():
            return int(ch)
    return None


class LowEmissionZoneInspector:
    def __init__(
        self,
        config: LowEmissionZoneConfig | None = None,
        registry: Sequence[VehicleEmissionProfile] = (),
    ) -> None:
        self.config = config if config is not None else LowEmissionZoneConfig()
        self._registry: dict[str, VehicleEmissionProfile] = {}
        for item in registry:
            self.register_vehicle(item)
        self._history: list[LezInspectionResult] = []
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

    def register_vehicle(self, profile: VehicleEmissionProfile) -> None:
        code, norm = self._normalize_plate(profile.plate_text, profile.country_code)
        normalized_profile = VehicleEmissionProfile(
            plate_text=norm,
            country_code=code,
            emission_class=profile.emission_class,
            is_exempt=profile.is_exempt,
        )
        self._registry[f"{code}:{norm}"] = normalized_profile

    def _evaluate_compliance(
        self,
        normalized_plate: str,
        emission_class: EmissionClass,
        is_exempt: bool,
        day_of_month: int,
    ) -> tuple[bool, str | None, float]:
        if is_exempt or emission_class == EmissionClass.ZERO_EMISSION:
            return True, None, 0.0
        if (
            emission_class == EmissionClass.UNREGISTERED
            and EmissionClass.UNREGISTERED not in self.config.allowed_classes
        ):
            return (
                False,
                "unregistered_vehicle",
                round(self.config.unregistered_fine_amount, 2),
            )
        if emission_class not in self.config.allowed_classes:
            return (
                False,
                "restricted_emission_class",
                round(self.config.non_compliant_fine_amount, 2),
            )
        if self.config.enforce_odd_even_days:
            last_digit = _find_last_digit(normalized_plate)
            if last_digit is not None and (last_digit % 2) != (day_of_month % 2):
                return (
                    False,
                    "odd_even_day_restriction",
                    round(self.config.odd_even_fine_amount, 2),
                )
        return True, None, 0.0

    def inspect(
        self,
        plate_text: str,
        *,
        country_code: str = "TR",
        day_of_month: int = 1,
    ) -> LezInspectionResult:
        if day_of_month < 1 or day_of_month > 31:
            raise InvalidInputError(
                "day_of_month must be in [1, 31]",
                details={"day_of_month": day_of_month},
            )
        code, norm = self._normalize_plate(plate_text, country_code)
        registered = self._registry.get(f"{code}:{norm}")
        emission_class = (
            registered.emission_class if registered is not None else EmissionClass.UNREGISTERED
        )
        is_exempt = registered.is_exempt if registered is not None else False
        is_compliant, reason, fine_amount = self._evaluate_compliance(
            norm,
            emission_class,
            is_exempt,
            day_of_month,
        )

        result = LezInspectionResult(
            inspection_id=f"lez_{self._next_seq:06d}",
            zone_id=self.config.zone_id,
            plate_text=norm,
            country_code=code,
            day_of_month=day_of_month,
            emission_class=emission_class,
            is_exempt=is_exempt,
            is_compliant=is_compliant,
            violation_reason=reason,
            fine_amount=fine_amount,
            currency=self.config.currency,
        )
        self._next_seq += 1
        self._history.append(result)
        return result

    def history(self) -> tuple[LezInspectionResult, ...]:
        return tuple(self._history)

    def summarize(self) -> LezEnforcementSummary:
        total = len(self._history)
        if total == 0:
            return LezEnforcementSummary(
                zone_id=self.config.zone_id,
                total_inspections=0,
                compliant_count=0,
                violation_count=0,
                compliance_rate=0.0,
                total_fine_amount=0.0,
                violations_by_reason={},
                currency=self.config.currency,
            )

        compliant_count = sum(1 for item in self._history if item.is_compliant)
        violation_count = total - compliant_count
        compliance_rate = round(compliant_count / float(total), 6)
        total_fine = round(sum(item.fine_amount for item in self._history), 2)
        reason_counts: Counter[str] = Counter(
            item.violation_reason for item in self._history if item.violation_reason is not None
        )
        return LezEnforcementSummary(
            zone_id=self.config.zone_id,
            total_inspections=total,
            compliant_count=compliant_count,
            violation_count=violation_count,
            compliance_rate=compliance_rate,
            total_fine_amount=total_fine,
            violations_by_reason=dict(sorted(reason_counts.items())),
            currency=self.config.currency,
        )


__all__ = [
    "EmissionClass",
    "LezEnforcementSummary",
    "LezInspectionResult",
    "LowEmissionZoneConfig",
    "LowEmissionZoneInspector",
    "VehicleEmissionProfile",
]
