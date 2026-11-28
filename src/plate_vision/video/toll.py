from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from plate_vision.data.profiles import get_country_profile
from plate_vision.utils.exceptions import ConfigurationError, InvalidInputError


class VehicleTollClass(StrEnum):
    MOTORCYCLE = "motorcycle"
    PASSENGER = "passenger"
    VAN = "van"
    TRUCK = "truck"
    ELECTRIC = "electric"


class TollTransactionStatus(StrEnum):
    PAID_PREPAID = "paid_prepaid"
    BILLED_POSTPAID = "billed_postpaid"
    EXEMPT = "exempt"
    UNPAID_VIOLATION = "unpaid_violation"
    DUPLICATE_IGNORED = "duplicate_ignored"


def _default_base_rates() -> dict[VehicleTollClass, float]:
    return {
        VehicleTollClass.MOTORCYCLE: 15.0,
        VehicleTollClass.PASSENGER: 35.0,
        VehicleTollClass.VAN: 55.0,
        VehicleTollClass.TRUCK: 95.0,
        VehicleTollClass.ELECTRIC: 20.0,
    }


class TollAccount(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plate_text: str = Field(min_length=2)
    country_code: str = Field(default="TR", min_length=2)
    vehicle_class: VehicleTollClass = VehicleTollClass.PASSENGER
    balance: float = Field(default=0.0, ge=0.0)
    postpaid_enabled: bool = False
    is_exempt: bool = False


class TollTariffConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gantry_id: str = Field(default="gantry_01", min_length=1)
    base_rates: dict[VehicleTollClass, float] = Field(default_factory=_default_base_rates)
    peak_hours_utc: tuple[int, ...] = (7, 8, 9, 17, 18, 19)
    peak_multiplier: float = Field(default=1.25, ge=1.0)
    violation_surcharge: float = Field(default=150.0, ge=0.0)
    min_pass_interval_seconds: float = Field(default=30.0, ge=0.0)
    currency: str = Field(default="TRY", min_length=3, max_length=3)


class TollTransaction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    transaction_id: str = Field(min_length=1)
    gantry_id: str = Field(min_length=1)
    plate_text: str = Field(min_length=2)
    country_code: str = Field(min_length=2)
    vehicle_class: VehicleTollClass
    timestamp_ms: int = Field(ge=0)
    is_peak_hour: bool
    base_fee: float = Field(ge=0.0)
    total_fee: float = Field(ge=0.0)
    remaining_balance: float | None = None
    status: TollTransactionStatus
    currency: str = Field(min_length=3, max_length=3)


class TollCollectionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gantry_id: str = Field(min_length=1)
    total_passages: int = Field(ge=0)
    paid_prepaid_count: int = Field(ge=0)
    billed_postpaid_count: int = Field(ge=0)
    exempt_count: int = Field(ge=0)
    violation_count: int = Field(ge=0)
    duplicate_ignored_count: int = Field(ge=0)
    total_collected_revenue: float = Field(ge=0.0)
    total_unpaid_amount: float = Field(ge=0.0)
    currency: str = Field(min_length=3, max_length=3)


class TollGantryEngine:
    def __init__(self, tariff: TollTariffConfig | None = None) -> None:
        active_tariff = tariff if tariff is not None else TollTariffConfig()
        for hour in active_tariff.peak_hours_utc:
            if not (0 <= hour <= 23):
                raise ConfigurationError(
                    f"peak_hours_utc entries must be in 0..23: {hour}",
                    details={"hour": hour},
                )
        for vehicle_cls, rate in active_tariff.base_rates.items():
            if rate < 0.0:
                raise ConfigurationError(
                    f"Toll base rate must be >= 0 for {vehicle_cls.value}",
                    details={"vehicle_class": vehicle_cls.value, "rate": rate},
                )
        self.tariff = active_tariff
        self._accounts: dict[tuple[str, str], TollAccount] = {}
        self._last_charged_ms: dict[tuple[str, str], int] = {}
        self._transactions: list[TollTransaction] = []
        self._next_seq = 1

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

    def register_account(self, account: TollAccount) -> TollAccount:
        code, norm = self._normalize_plate(account.plate_text, account.country_code)
        normalized_account = account.model_copy(
            update={"plate_text": norm, "country_code": code, "balance": round(account.balance, 2)}
        )
        self._accounts[(code, norm)] = normalized_account
        return normalized_account

    def get_account(self, plate_text: str, *, country_code: str = "TR") -> TollAccount | None:
        code, norm = self._normalize_plate(plate_text, country_code)
        return self._accounts.get((code, norm))

    def top_up_account(
        self,
        plate_text: str,
        amount: float,
        *,
        country_code: str = "TR",
    ) -> TollAccount:
        if amount <= 0.0:
            raise InvalidInputError(
                "Top-up amount must be > 0",
                details={"amount": amount},
            )
        code, norm = self._normalize_plate(plate_text, country_code)
        key = (code, norm)
        existing = self._accounts.get(key)
        if existing is None:
            raise InvalidInputError(
                f"No toll account registered for {norm}",
                details={"plate_text": norm, "country_code": code},
            )
        updated = existing.model_copy(update={"balance": round(existing.balance + amount, 2)})
        self._accounts[key] = updated
        return updated

    def _next_transaction_id(self) -> str:
        tx_id = f"{self.tariff.gantry_id}_tx_{self._next_seq:06d}"
        self._next_seq += 1
        return tx_id

    def process_passage(
        self,
        plate_text: str,
        *,
        timestamp_ms: int,
        country_code: str = "TR",
        vehicle_class: VehicleTollClass | None = None,
    ) -> TollTransaction:
        if timestamp_ms < 0:
            raise InvalidInputError(
                "timestamp_ms must be >= 0",
                details={"timestamp_ms": timestamp_ms},
            )
        code, norm = self._normalize_plate(plate_text, country_code)
        key = (code, norm)
        account = self._accounts.get(key)
        active_class = (
            vehicle_class
            if vehicle_class is not None
            else (account.vehicle_class if account is not None else VehicleTollClass.PASSENGER)
        )

        utc_hour = (timestamp_ms // 3_600_000) % 24
        is_peak = utc_hour in self.tariff.peak_hours_utc
        base_rate = float(self.tariff.base_rates.get(active_class, 35.0))
        adjusted_fee = round(base_rate * (self.tariff.peak_multiplier if is_peak else 1.0), 2)

        prev_ts = self._last_charged_ms.get(key)
        cooldown_ms = int(self.tariff.min_pass_interval_seconds * 1000.0)
        if prev_ts is not None and 0 <= (timestamp_ms - prev_ts) < cooldown_ms:
            tx = TollTransaction(
                transaction_id=self._next_transaction_id(),
                gantry_id=self.tariff.gantry_id,
                plate_text=norm,
                country_code=code,
                vehicle_class=active_class,
                timestamp_ms=timestamp_ms,
                is_peak_hour=is_peak,
                base_fee=base_rate,
                total_fee=0.0,
                remaining_balance=account.balance if account is not None else None,
                status=TollTransactionStatus.DUPLICATE_IGNORED,
                currency=self.tariff.currency,
            )
            self._transactions.append(tx)
            return tx

        self._last_charged_ms[key] = timestamp_ms

        if account is not None and account.is_exempt:
            status = TollTransactionStatus.EXEMPT
            total_fee = 0.0
            remaining_balance: float | None = account.balance
        elif account is not None and account.balance >= adjusted_fee:
            new_balance = round(account.balance - adjusted_fee, 2)
            self._accounts[key] = account.model_copy(update={"balance": new_balance})
            status = TollTransactionStatus.PAID_PREPAID
            total_fee = adjusted_fee
            remaining_balance = new_balance
        elif account is not None and account.postpaid_enabled:
            status = TollTransactionStatus.BILLED_POSTPAID
            total_fee = adjusted_fee
            remaining_balance = account.balance
        else:
            status = TollTransactionStatus.UNPAID_VIOLATION
            total_fee = round(adjusted_fee + self.tariff.violation_surcharge, 2)
            remaining_balance = account.balance if account is not None else None

        tx = TollTransaction(
            transaction_id=self._next_transaction_id(),
            gantry_id=self.tariff.gantry_id,
            plate_text=norm,
            country_code=code,
            vehicle_class=active_class,
            timestamp_ms=timestamp_ms,
            is_peak_hour=is_peak,
            base_fee=base_rate,
            total_fee=total_fee,
            remaining_balance=remaining_balance,
            status=status,
            currency=self.tariff.currency,
        )
        self._transactions.append(tx)
        return tx

    def transactions(self) -> list[TollTransaction]:
        return list(self._transactions)

    def summary(self) -> TollCollectionSummary:
        paid_prepaid = 0
        billed_postpaid = 0
        exempt = 0
        violations = 0
        duplicates = 0
        collected = 0.0
        unpaid = 0.0

        for tx in self._transactions:
            if tx.status == TollTransactionStatus.PAID_PREPAID:
                paid_prepaid += 1
                collected += tx.total_fee
            elif tx.status == TollTransactionStatus.BILLED_POSTPAID:
                billed_postpaid += 1
                collected += tx.total_fee
            elif tx.status == TollTransactionStatus.EXEMPT:
                exempt += 1
            elif tx.status == TollTransactionStatus.UNPAID_VIOLATION:
                violations += 1
                unpaid += tx.total_fee
            else:
                duplicates += 1

        return TollCollectionSummary(
            gantry_id=self.tariff.gantry_id,
            total_passages=len(self._transactions),
            paid_prepaid_count=paid_prepaid,
            billed_postpaid_count=billed_postpaid,
            exempt_count=exempt,
            violation_count=violations,
            duplicate_ignored_count=duplicates,
            total_collected_revenue=round(collected, 2),
            total_unpaid_amount=round(unpaid, 2),
            currency=self.tariff.currency,
        )


__all__ = [
    "TollAccount",
    "TollCollectionSummary",
    "TollGantryEngine",
    "TollTariffConfig",
    "TollTransaction",
    "TollTransactionStatus",
    "VehicleTollClass",
]
