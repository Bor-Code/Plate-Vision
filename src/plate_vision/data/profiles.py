import random
import re
from collections.abc import Mapping
from re import Pattern
from typing import Final, Protocol

from plate_vision.utils.exceptions import ConfigurationError

TURKEY_CHARSET: Final[str] = "0123456789ABCDEFGHIJKLMNOPRSTUVYZ"
TURKEY_LETTERS: Final[str] = "ABCDEFGHIJKLMNOPRSTUVYZ"
TURKEY_DIGITS: Final[str] = "0123456789"

TURKEY_DIGIT_TO_ALPHA: Final[Mapping[str, str]] = {
    "0": "O",
    "1": "I",
    "8": "B",
    "5": "S",
    "2": "Z",
}

TURKEY_ALPHA_TO_DIGIT: Final[Mapping[str, str]] = {
    "O": "0",
    "I": "1",
    "B": "8",
    "S": "5",
    "Z": "2",
    "D": "0",
}

TURKEY_PATTERNS: Final[tuple[Pattern[str], ...]] = (
    re.compile(r"^(0[1-9]|[1-7][0-9]|8[01])([A-PR-VYZ]{1})([0-9]{4})$"),
    re.compile(r"^(0[1-9]|[1-7][0-9]|8[01])([A-PR-VYZ]{2})([0-9]{3,4})$"),
    re.compile(r"^(0[1-9]|[1-7][0-9]|8[01])([A-PR-VYZ]{3})([0-9]{2,3})$"),
)

STANDARD_LATIN_CHARSET: Final[str] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
STANDARD_LETTERS: Final[str] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

GERMANY_PATTERNS: Final[tuple[Pattern[str], ...]] = (
    re.compile(r"^([A-Z]{2,5})([1-9][0-9]{0,3})$"),
)

UK_PATTERNS: Final[tuple[Pattern[str], ...]] = (re.compile(r"^([A-Z]{2})([0-9]{2})([A-Z]{3})$"),)

AZERBAIJAN_PATTERNS: Final[tuple[Pattern[str], ...]] = (
    re.compile(r"^(0[1-9]|[1-9][0-9])([A-Z]{2})([0-9]{3})$"),
)

FRANCE_LETTERS: Final[str] = "ABCDEFGHJKLMNPQRSTVWXYZ"
FRANCE_PATTERNS: Final[tuple[Pattern[str], ...]] = (
    re.compile(r"^([A-HJ-NP-TV-Z]{2})([0-9]{3})([A-HJ-NP-TV-Z]{2})$"),
)

ITALY_LETTERS: Final[str] = "ABCDEFGHJKLMNPRSTVWXYZ"
ITALY_PATTERNS: Final[tuple[Pattern[str], ...]] = (
    re.compile(r"^([A-HJ-NPR-TV-Z]{2})([0-9]{3})([A-HJ-NPR-TV-Z]{2})$"),
)

SPAIN_LETTERS: Final[str] = "BCDFGHJKLMNPRSTVWXYZ"
SPAIN_PATTERNS: Final[tuple[Pattern[str], ...]] = (
    re.compile(r"^([0-9]{4})([BCDFGHJKLMNPRSTVWXYZ]{3})$"),
)

NETHERLANDS_LETTERS: Final[str] = "BCDFGHJKLMNPQRSTVWXZ"
NETHERLANDS_PATTERNS: Final[tuple[Pattern[str], ...]] = (
    re.compile(r"^([0-9]{2})([BCDFGHJKLMNPQRSTVWXZ]{3})([0-9]{1})$"),
    re.compile(r"^([0-9]{1})([BCDFGHJKLMNPQRSTVWXZ]{3})([0-9]{2})$"),
)

US_PATTERNS: Final[tuple[Pattern[str], ...]] = (
    re.compile(r"^([A-Z]{3})([0-9]{4})$"),
    re.compile(r"^([0-9]{1})([A-Z]{3})([0-9]{3})$"),
)

CANADA_PATTERNS: Final[tuple[Pattern[str], ...]] = (
    re.compile(r"^([A-Z]{4})([0-9]{3})$"),
    re.compile(r"^([A-Z]{3})([0-9]{3})$"),
)

JAPAN_PATTERNS: Final[tuple[Pattern[str], ...]] = (re.compile(r"^([0-9]{3})([A-Z]{1})([0-9]{4})$"),)

SOUTH_KOREA_PATTERNS: Final[tuple[Pattern[str], ...]] = (
    re.compile(r"^([0-9]{2,3})([A-Z]{1})([0-9]{4})$"),
)

BRAZIL_PATTERNS: Final[tuple[Pattern[str], ...]] = (
    re.compile(r"^([A-Z]{3})([0-9]{1})([A-Z]{1})([0-9]{2})$"),
)

MEXICO_PATTERNS: Final[tuple[Pattern[str], ...]] = (
    re.compile(r"^([A-Z]{3})([0-9]{3})([A-Z]{1})$"),
)

UAE_PATTERNS: Final[tuple[Pattern[str], ...]] = (re.compile(r"^([A-Z]{1,2})([1-9][0-9]{3,4})$"),)

SAUDI_ARABIA_PATTERNS: Final[tuple[Pattern[str], ...]] = (re.compile(r"^([0-9]{4})([A-Z]{3})$"),)

INDIA_PATTERNS: Final[tuple[Pattern[str], ...]] = (
    re.compile(r"^([A-Z]{2})([0-9]{2})([A-Z]{2})([0-9]{4})$"),
)

AUSTRALIA_PATTERNS: Final[tuple[Pattern[str], ...]] = (
    re.compile(r"^([A-Z]{2})([0-9]{2})([A-Z]{2})$"),
)

POLAND_PATTERNS: Final[tuple[Pattern[str], ...]] = (
    re.compile(r"^([A-Z]{2})([0-9]{4})([A-Z]{1})$"),
)

SWEDEN_PATTERNS: Final[tuple[Pattern[str], ...]] = (
    re.compile(r"^([A-Z]{3})([0-9]{2})([A-Z]{1})$"),
)


class CountryProfileProtocol(Protocol):
    @property
    def country_code(self) -> str: ...

    @property
    def charset(self) -> str: ...

    @property
    def regex_patterns(self) -> tuple[Pattern[str], ...]: ...

    @property
    def digit_to_alpha_map(self) -> Mapping[str, str]: ...

    @property
    def alpha_to_digit_map(self) -> Mapping[str, str]: ...

    def normalize(self, raw_text: str) -> str: ...

    def is_valid(self, plate_text: str) -> bool: ...

    def correct_confusions(self, raw_text: str) -> str: ...

    def sample_plate(self, rng: random.Random) -> str: ...


class TurkeyCountryProfile:
    @property
    def country_code(self) -> str:
        return "TR"

    @property
    def charset(self) -> str:
        return TURKEY_CHARSET

    @property
    def regex_patterns(self) -> tuple[Pattern[str], ...]:
        return TURKEY_PATTERNS

    @property
    def digit_to_alpha_map(self) -> Mapping[str, str]:
        return TURKEY_DIGIT_TO_ALPHA

    @property
    def alpha_to_digit_map(self) -> Mapping[str, str]:
        return TURKEY_ALPHA_TO_DIGIT

    def normalize(self, raw_text: str) -> str:
        cleaned = re.sub(r"[\s\-_.]+", "", raw_text.upper())
        return "".join(ch for ch in cleaned if ch.isalnum())

    def is_valid(self, plate_text: str) -> bool:
        normalized = self.normalize(plate_text)
        return any(pattern.fullmatch(normalized) is not None for pattern in self.regex_patterns)

    def _map_to_digits(self, segment: str) -> tuple[str, int]:
        chars: list[str] = []
        edits = 0
        for ch in segment:
            if ch.isdigit():
                chars.append(ch)
            elif ch in self.alpha_to_digit_map:
                chars.append(self.alpha_to_digit_map[ch])
                edits += 1
            else:
                chars.append(ch)
        return "".join(chars), edits

    def _map_to_letters(self, segment: str) -> tuple[str, int]:
        chars: list[str] = []
        edits = 0
        for ch in segment:
            if ch in TURKEY_LETTERS:
                chars.append(ch)
            elif ch in self.digit_to_alpha_map:
                chars.append(self.digit_to_alpha_map[ch])
                edits += 1
            else:
                chars.append(ch)
        return "".join(chars), edits

    def correct_confusions(self, raw_text: str) -> str:
        normalized = self.normalize(raw_text)
        if self.is_valid(normalized):
            return normalized

        length = len(normalized)
        if length < 6 or length > 8:
            return normalized

        best_candidate: str | None = None
        best_edits = length + 1

        for letter_len in (1, 2, 3):
            digit_len = length - 2 - letter_len
            if digit_len < 2 or digit_len > 4:
                continue
            prov_seg = normalized[:2]
            letter_seg = normalized[2 : 2 + letter_len]
            digit_seg = normalized[2 + letter_len :]

            fixed_prov, e1 = self._map_to_digits(prov_seg)
            fixed_letters, e2 = self._map_to_letters(letter_seg)
            fixed_digits, e3 = self._map_to_digits(digit_seg)

            candidate = f"{fixed_prov}{fixed_letters}{fixed_digits}"
            total_edits = e1 + e2 + e3
            if self.is_valid(candidate) and total_edits < best_edits:
                best_candidate = candidate
                best_edits = total_edits

        return best_candidate if best_candidate is not None else normalized

    def sample_plate(self, rng: random.Random) -> str:
        province = f"{rng.randint(1, 81):02d}"
        letter_count = rng.choice((1, 2, 3))
        letters = "".join(rng.choice(TURKEY_LETTERS) for _ in range(letter_count))
        if letter_count == 1:
            digit_count = 4
        elif letter_count == 2:
            digit_count = rng.choice((3, 4))
        else:
            digit_count = rng.choice((2, 3))
        low = 10 ** (digit_count - 1)
        high = (10**digit_count) - 1
        digits = f"{rng.randint(low, high)}"
        return f"{province}{letters}{digits}"


class GermanyCountryProfile:
    @property
    def country_code(self) -> str:
        return "DE"

    @property
    def charset(self) -> str:
        return STANDARD_LATIN_CHARSET

    @property
    def regex_patterns(self) -> tuple[Pattern[str], ...]:
        return GERMANY_PATTERNS

    @property
    def digit_to_alpha_map(self) -> Mapping[str, str]:
        return TURKEY_DIGIT_TO_ALPHA

    @property
    def alpha_to_digit_map(self) -> Mapping[str, str]:
        return TURKEY_ALPHA_TO_DIGIT

    def normalize(self, raw_text: str) -> str:
        cleaned = re.sub(r"[\s\-_.]+", "", raw_text.upper())
        return "".join(ch for ch in cleaned if ch.isalnum())

    def is_valid(self, plate_text: str) -> bool:
        normalized = self.normalize(plate_text)
        if len(normalized) < 3 or len(normalized) > 8:
            return False
        return any(p.fullmatch(normalized) is not None for p in self.regex_patterns)

    def correct_confusions(self, raw_text: str) -> str:
        normalized = self.normalize(raw_text)
        if self.is_valid(normalized):
            return normalized
        length = len(normalized)
        if length < 3 or length > 8:
            return normalized

        best_candidate: str | None = None
        best_edits = length + 1
        for alpha_len in range(2, min(5, length - 1) + 1):
            digit_len = length - alpha_len
            if digit_len < 1 or digit_len > 4:
                continue
            alpha_part = normalized[:alpha_len]
            digit_part = normalized[alpha_len:]
            fixed_a: list[str] = []
            edits = 0
            for ch in alpha_part:
                if ch in STANDARD_LETTERS:
                    fixed_a.append(ch)
                elif ch in self.digit_to_alpha_map:
                    fixed_a.append(self.digit_to_alpha_map[ch])
                    edits += 1
                else:
                    fixed_a.append(ch)
            fixed_d: list[str] = []
            for ch in digit_part:
                if ch.isdigit():
                    fixed_d.append(ch)
                elif ch in self.alpha_to_digit_map:
                    fixed_d.append(self.alpha_to_digit_map[ch])
                    edits += 1
                else:
                    fixed_d.append(ch)
            candidate = "".join(fixed_a) + "".join(fixed_d)
            if self.is_valid(candidate) and edits < best_edits:
                best_candidate = candidate
                best_edits = edits
        return best_candidate if best_candidate is not None else normalized

    def sample_plate(self, rng: random.Random) -> str:
        city_len = rng.choice((1, 2, 3))
        mid_len = rng.choice((1, 2))
        letters = "".join(rng.choice(STANDARD_LETTERS) for _ in range(city_len + mid_len))
        num = rng.randint(1, 9999)
        return f"{letters}{num}"


class UnitedKingdomCountryProfile:
    @property
    def country_code(self) -> str:
        return "GB"

    @property
    def charset(self) -> str:
        return STANDARD_LATIN_CHARSET

    @property
    def regex_patterns(self) -> tuple[Pattern[str], ...]:
        return UK_PATTERNS

    @property
    def digit_to_alpha_map(self) -> Mapping[str, str]:
        return TURKEY_DIGIT_TO_ALPHA

    @property
    def alpha_to_digit_map(self) -> Mapping[str, str]:
        return TURKEY_ALPHA_TO_DIGIT

    def normalize(self, raw_text: str) -> str:
        cleaned = re.sub(r"[\s\-_.]+", "", raw_text.upper())
        return "".join(ch for ch in cleaned if ch.isalnum())

    def is_valid(self, plate_text: str) -> bool:
        normalized = self.normalize(plate_text)
        return any(p.fullmatch(normalized) is not None for p in self.regex_patterns)

    def correct_confusions(self, raw_text: str) -> str:
        normalized = self.normalize(raw_text)
        if self.is_valid(normalized) or len(normalized) != 7:
            return normalized
        chars = list(normalized)
        for idx in (0, 1, 4, 5, 6):
            if chars[idx] in self.digit_to_alpha_map:
                chars[idx] = self.digit_to_alpha_map[chars[idx]]
        for idx in (2, 3):
            if chars[idx] in self.alpha_to_digit_map:
                chars[idx] = self.alpha_to_digit_map[chars[idx]]
        candidate = "".join(chars)
        return candidate if self.is_valid(candidate) else normalized

    def sample_plate(self, rng: random.Random) -> str:
        area = "".join(rng.choice(STANDARD_LETTERS) for _ in range(2))
        age = f"{rng.randint(10, 74):02d}"
        suffix = "".join(rng.choice(STANDARD_LETTERS) for _ in range(3))
        return f"{area}{age}{suffix}"


class AzerbaijanCountryProfile:
    @property
    def country_code(self) -> str:
        return "AZ"

    @property
    def charset(self) -> str:
        return STANDARD_LATIN_CHARSET

    @property
    def regex_patterns(self) -> tuple[Pattern[str], ...]:
        return AZERBAIJAN_PATTERNS

    @property
    def digit_to_alpha_map(self) -> Mapping[str, str]:
        return TURKEY_DIGIT_TO_ALPHA

    @property
    def alpha_to_digit_map(self) -> Mapping[str, str]:
        return TURKEY_ALPHA_TO_DIGIT

    def normalize(self, raw_text: str) -> str:
        cleaned = re.sub(r"[\s\-_.]+", "", raw_text.upper())
        return "".join(ch for ch in cleaned if ch.isalnum())

    def is_valid(self, plate_text: str) -> bool:
        normalized = self.normalize(plate_text)
        return any(p.fullmatch(normalized) is not None for p in self.regex_patterns)

    def correct_confusions(self, raw_text: str) -> str:
        normalized = self.normalize(raw_text)
        if self.is_valid(normalized) or len(normalized) != 7:
            return normalized
        chars = list(normalized)
        for idx in (0, 1, 4, 5, 6):
            if chars[idx] in self.alpha_to_digit_map:
                chars[idx] = self.alpha_to_digit_map[chars[idx]]
        for idx in (2, 3):
            if chars[idx] in self.digit_to_alpha_map:
                chars[idx] = self.digit_to_alpha_map[chars[idx]]
        candidate = "".join(chars)
        return candidate if self.is_valid(candidate) else normalized

    def sample_plate(self, rng: random.Random) -> str:
        region = f"{rng.randint(1, 99):02d}"
        letters = "".join(rng.choice(STANDARD_LETTERS) for _ in range(2))
        digits = f"{rng.randint(100, 999)}"
        return f"{region}{letters}{digits}"


class FranceCountryProfile:
    @property
    def country_code(self) -> str:
        return "FR"

    @property
    def charset(self) -> str:
        return STANDARD_LATIN_CHARSET

    @property
    def regex_patterns(self) -> tuple[Pattern[str], ...]:
        return FRANCE_PATTERNS

    @property
    def digit_to_alpha_map(self) -> Mapping[str, str]:
        return {"8": "B", "5": "S", "2": "Z"}

    @property
    def alpha_to_digit_map(self) -> Mapping[str, str]:
        return TURKEY_ALPHA_TO_DIGIT

    def normalize(self, raw_text: str) -> str:
        cleaned = re.sub(r"[\s\-_.]+", "", raw_text.upper())
        return "".join(ch for ch in cleaned if ch.isalnum())

    def is_valid(self, plate_text: str) -> bool:
        normalized = self.normalize(plate_text)
        return any(p.fullmatch(normalized) is not None for p in self.regex_patterns)

    def correct_confusions(self, raw_text: str) -> str:
        normalized = self.normalize(raw_text)
        if self.is_valid(normalized) or len(normalized) != 7:
            return normalized
        chars = list(normalized)
        for idx in (0, 1, 5, 6):
            if chars[idx] in self.digit_to_alpha_map:
                chars[idx] = self.digit_to_alpha_map[chars[idx]]
        for idx in (2, 3, 4):
            if chars[idx] in self.alpha_to_digit_map:
                chars[idx] = self.alpha_to_digit_map[chars[idx]]
        candidate = "".join(chars)
        return candidate if self.is_valid(candidate) else normalized

    def sample_plate(self, rng: random.Random) -> str:
        prefix = "".join(rng.choice(FRANCE_LETTERS) for _ in range(2))
        digits = f"{rng.randint(1, 999):03d}"
        suffix = "".join(rng.choice(FRANCE_LETTERS) for _ in range(2))
        return f"{prefix}{digits}{suffix}"


class ItalyCountryProfile:
    @property
    def country_code(self) -> str:
        return "IT"

    @property
    def charset(self) -> str:
        return STANDARD_LATIN_CHARSET

    @property
    def regex_patterns(self) -> tuple[Pattern[str], ...]:
        return ITALY_PATTERNS

    @property
    def digit_to_alpha_map(self) -> Mapping[str, str]:
        return {"8": "B", "5": "S", "2": "Z"}

    @property
    def alpha_to_digit_map(self) -> Mapping[str, str]:
        return TURKEY_ALPHA_TO_DIGIT

    def normalize(self, raw_text: str) -> str:
        cleaned = re.sub(r"[\s\-_.]+", "", raw_text.upper())
        return "".join(ch for ch in cleaned if ch.isalnum())

    def is_valid(self, plate_text: str) -> bool:
        normalized = self.normalize(plate_text)
        return any(p.fullmatch(normalized) is not None for p in self.regex_patterns)

    def correct_confusions(self, raw_text: str) -> str:
        normalized = self.normalize(raw_text)
        if self.is_valid(normalized) or len(normalized) != 7:
            return normalized
        chars = list(normalized)
        for idx in (0, 1, 5, 6):
            if chars[idx] in self.digit_to_alpha_map:
                chars[idx] = self.digit_to_alpha_map[chars[idx]]
        for idx in (2, 3, 4):
            if chars[idx] in self.alpha_to_digit_map:
                chars[idx] = self.alpha_to_digit_map[chars[idx]]
        candidate = "".join(chars)
        return candidate if self.is_valid(candidate) else normalized

    def sample_plate(self, rng: random.Random) -> str:
        prefix = "".join(rng.choice(ITALY_LETTERS) for _ in range(2))
        digits = f"{rng.randint(1, 999):03d}"
        suffix = "".join(rng.choice(ITALY_LETTERS) for _ in range(2))
        return f"{prefix}{digits}{suffix}"


class SpainCountryProfile:
    @property
    def country_code(self) -> str:
        return "ES"

    @property
    def charset(self) -> str:
        return STANDARD_LATIN_CHARSET

    @property
    def regex_patterns(self) -> tuple[Pattern[str], ...]:
        return SPAIN_PATTERNS

    @property
    def digit_to_alpha_map(self) -> Mapping[str, str]:
        return {"8": "B", "5": "S", "2": "Z"}

    @property
    def alpha_to_digit_map(self) -> Mapping[str, str]:
        return TURKEY_ALPHA_TO_DIGIT

    def normalize(self, raw_text: str) -> str:
        cleaned = re.sub(r"[\s\-_.]+", "", raw_text.upper())
        return "".join(ch for ch in cleaned if ch.isalnum())

    def is_valid(self, plate_text: str) -> bool:
        normalized = self.normalize(plate_text)
        return any(p.fullmatch(normalized) is not None for p in self.regex_patterns)

    def correct_confusions(self, raw_text: str) -> str:
        normalized = self.normalize(raw_text)
        if self.is_valid(normalized) or len(normalized) != 7:
            return normalized
        chars = list(normalized)
        for idx in (0, 1, 2, 3):
            if chars[idx] in self.alpha_to_digit_map:
                chars[idx] = self.alpha_to_digit_map[chars[idx]]
        for idx in (4, 5, 6):
            if chars[idx] in self.digit_to_alpha_map:
                chars[idx] = self.digit_to_alpha_map[chars[idx]]
        candidate = "".join(chars)
        return candidate if self.is_valid(candidate) else normalized

    def sample_plate(self, rng: random.Random) -> str:
        digits = f"{rng.randint(0, 9999):04d}"
        letters = "".join(rng.choice(SPAIN_LETTERS) for _ in range(3))
        return f"{digits}{letters}"


class NetherlandsCountryProfile:
    @property
    def country_code(self) -> str:
        return "NL"

    @property
    def charset(self) -> str:
        return STANDARD_LATIN_CHARSET

    @property
    def regex_patterns(self) -> tuple[Pattern[str], ...]:
        return NETHERLANDS_PATTERNS

    @property
    def digit_to_alpha_map(self) -> Mapping[str, str]:
        return {"8": "B", "5": "S", "2": "Z"}

    @property
    def alpha_to_digit_map(self) -> Mapping[str, str]:
        return TURKEY_ALPHA_TO_DIGIT

    def normalize(self, raw_text: str) -> str:
        cleaned = re.sub(r"[\s\-_.]+", "", raw_text.upper())
        return "".join(ch for ch in cleaned if ch.isalnum())

    def is_valid(self, plate_text: str) -> bool:
        normalized = self.normalize(plate_text)
        return any(p.fullmatch(normalized) is not None for p in self.regex_patterns)

    def correct_confusions(self, raw_text: str) -> str:
        normalized = self.normalize(raw_text)
        if self.is_valid(normalized) or len(normalized) != 6:
            return normalized
        layouts = (
            ((0, 1, 5), (2, 3, 4)),
            ((0, 4, 5), (1, 2, 3)),
        )
        for digit_indices, alpha_indices in layouts:
            chars = list(normalized)
            for idx in digit_indices:
                if chars[idx] in self.alpha_to_digit_map:
                    chars[idx] = self.alpha_to_digit_map[chars[idx]]
            for idx in alpha_indices:
                if chars[idx] in self.digit_to_alpha_map:
                    chars[idx] = self.digit_to_alpha_map[chars[idx]]
            candidate = "".join(chars)
            if self.is_valid(candidate):
                return candidate
        return normalized

    def sample_plate(self, rng: random.Random) -> str:
        letters = "".join(rng.choice(NETHERLANDS_LETTERS) for _ in range(3))
        if rng.choice((True, False)):
            return f"{rng.randint(10, 99)}{letters}{rng.randint(1, 9)}"
        return f"{rng.randint(1, 9)}{letters}{rng.randint(10, 99)}"


def _normalize_alnum(raw_text: str) -> str:
    cleaned = re.sub(r"[\s\-_.]+", "", raw_text.upper())
    return "".join(ch for ch in cleaned if ch.isalnum())


def _apply_positional_layouts(
    normalized: str,
    layouts: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    alpha_to_digit: Mapping[str, str],
    digit_to_alpha: Mapping[str, str],
    patterns: tuple[Pattern[str], ...],
) -> str:
    for digit_indices, alpha_indices in layouts:
        chars = list(normalized)
        for idx in digit_indices:
            if chars[idx] in alpha_to_digit:
                chars[idx] = alpha_to_digit[chars[idx]]
        for idx in alpha_indices:
            if chars[idx] in digit_to_alpha:
                chars[idx] = digit_to_alpha[chars[idx]]
        candidate = "".join(chars)
        if any(p.fullmatch(candidate) is not None for p in patterns):
            return candidate
    return normalized


class UnitedStatesCountryProfile:
    @property
    def country_code(self) -> str:
        return "US"

    @property
    def charset(self) -> str:
        return STANDARD_LATIN_CHARSET

    @property
    def regex_patterns(self) -> tuple[Pattern[str], ...]:
        return US_PATTERNS

    @property
    def digit_to_alpha_map(self) -> Mapping[str, str]:
        return TURKEY_DIGIT_TO_ALPHA

    @property
    def alpha_to_digit_map(self) -> Mapping[str, str]:
        return TURKEY_ALPHA_TO_DIGIT

    def normalize(self, raw_text: str) -> str:
        return _normalize_alnum(raw_text)

    def is_valid(self, plate_text: str) -> bool:
        normalized = self.normalize(plate_text)
        return any(p.fullmatch(normalized) is not None for p in self.regex_patterns)

    def correct_confusions(self, raw_text: str) -> str:
        normalized = self.normalize(raw_text)
        if self.is_valid(normalized) or len(normalized) != 7:
            return normalized
        layouts = (
            ((3, 4, 5, 6), (0, 1, 2)),
            ((0, 4, 5, 6), (1, 2, 3)),
        )
        return _apply_positional_layouts(
            normalized,
            layouts,
            self.alpha_to_digit_map,
            self.digit_to_alpha_map,
            self.regex_patterns,
        )

    def sample_plate(self, rng: random.Random) -> str:
        letters = "".join(rng.choice(STANDARD_LETTERS) for _ in range(3))
        if rng.choice((True, False)):
            return f"{letters}{rng.randint(1000, 9999)}"
        return f"{rng.randint(1, 9)}{letters}{rng.randint(100, 999)}"


class CanadaCountryProfile:
    @property
    def country_code(self) -> str:
        return "CA"

    @property
    def charset(self) -> str:
        return STANDARD_LATIN_CHARSET

    @property
    def regex_patterns(self) -> tuple[Pattern[str], ...]:
        return CANADA_PATTERNS

    @property
    def digit_to_alpha_map(self) -> Mapping[str, str]:
        return TURKEY_DIGIT_TO_ALPHA

    @property
    def alpha_to_digit_map(self) -> Mapping[str, str]:
        return TURKEY_ALPHA_TO_DIGIT

    def normalize(self, raw_text: str) -> str:
        return _normalize_alnum(raw_text)

    def is_valid(self, plate_text: str) -> bool:
        normalized = self.normalize(plate_text)
        return any(p.fullmatch(normalized) is not None for p in self.regex_patterns)

    def correct_confusions(self, raw_text: str) -> str:
        normalized = self.normalize(raw_text)
        if self.is_valid(normalized):
            return normalized
        if len(normalized) == 7:
            layouts: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = (
                ((4, 5, 6), (0, 1, 2, 3)),
            )
        elif len(normalized) == 6:
            layouts = (((3, 4, 5), (0, 1, 2)),)
        else:
            return normalized
        return _apply_positional_layouts(
            normalized,
            layouts,
            self.alpha_to_digit_map,
            self.digit_to_alpha_map,
            self.regex_patterns,
        )

    def sample_plate(self, rng: random.Random) -> str:
        letter_count = rng.choice((3, 4))
        letters = "".join(rng.choice(STANDARD_LETTERS) for _ in range(letter_count))
        return f"{letters}{rng.randint(100, 999)}"


class JapanCountryProfile:
    @property
    def country_code(self) -> str:
        return "JP"

    @property
    def charset(self) -> str:
        return STANDARD_LATIN_CHARSET

    @property
    def regex_patterns(self) -> tuple[Pattern[str], ...]:
        return JAPAN_PATTERNS

    @property
    def digit_to_alpha_map(self) -> Mapping[str, str]:
        return TURKEY_DIGIT_TO_ALPHA

    @property
    def alpha_to_digit_map(self) -> Mapping[str, str]:
        return TURKEY_ALPHA_TO_DIGIT

    def normalize(self, raw_text: str) -> str:
        return _normalize_alnum(raw_text)

    def is_valid(self, plate_text: str) -> bool:
        normalized = self.normalize(plate_text)
        return any(p.fullmatch(normalized) is not None for p in self.regex_patterns)

    def correct_confusions(self, raw_text: str) -> str:
        normalized = self.normalize(raw_text)
        if self.is_valid(normalized) or len(normalized) != 8:
            return normalized
        layouts = (((0, 1, 2, 4, 5, 6, 7), (3,)),)
        return _apply_positional_layouts(
            normalized,
            layouts,
            self.alpha_to_digit_map,
            self.digit_to_alpha_map,
            self.regex_patterns,
        )

    def sample_plate(self, rng: random.Random) -> str:
        prefix = f"{rng.randint(100, 999)}"
        code_char = rng.choice(STANDARD_LETTERS)
        suffix = f"{rng.randint(1000, 9999)}"
        return f"{prefix}{code_char}{suffix}"


class SouthKoreaCountryProfile:
    @property
    def country_code(self) -> str:
        return "KR"

    @property
    def charset(self) -> str:
        return STANDARD_LATIN_CHARSET

    @property
    def regex_patterns(self) -> tuple[Pattern[str], ...]:
        return SOUTH_KOREA_PATTERNS

    @property
    def digit_to_alpha_map(self) -> Mapping[str, str]:
        return TURKEY_DIGIT_TO_ALPHA

    @property
    def alpha_to_digit_map(self) -> Mapping[str, str]:
        return TURKEY_ALPHA_TO_DIGIT

    def normalize(self, raw_text: str) -> str:
        return _normalize_alnum(raw_text)

    def is_valid(self, plate_text: str) -> bool:
        normalized = self.normalize(plate_text)
        return any(p.fullmatch(normalized) is not None for p in self.regex_patterns)

    def correct_confusions(self, raw_text: str) -> str:
        normalized = self.normalize(raw_text)
        if self.is_valid(normalized):
            return normalized
        if len(normalized) == 7:
            layouts: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = (
                ((0, 1, 3, 4, 5, 6), (2,)),
            )
        elif len(normalized) == 8:
            layouts = (((0, 1, 2, 4, 5, 6, 7), (3,)),)
        else:
            return normalized
        return _apply_positional_layouts(
            normalized,
            layouts,
            self.alpha_to_digit_map,
            self.digit_to_alpha_map,
            self.regex_patterns,
        )

    def sample_plate(self, rng: random.Random) -> str:
        prefix_len = rng.choice((2, 3))
        low = 10 ** (prefix_len - 1)
        high = (10**prefix_len) - 1
        prefix = f"{rng.randint(low, high)}"
        code_char = rng.choice(STANDARD_LETTERS)
        suffix = f"{rng.randint(1000, 9999)}"
        return f"{prefix}{code_char}{suffix}"


class BrazilCountryProfile:
    @property
    def country_code(self) -> str:
        return "BR"

    @property
    def charset(self) -> str:
        return STANDARD_LATIN_CHARSET

    @property
    def regex_patterns(self) -> tuple[Pattern[str], ...]:
        return BRAZIL_PATTERNS

    @property
    def digit_to_alpha_map(self) -> Mapping[str, str]:
        return TURKEY_DIGIT_TO_ALPHA

    @property
    def alpha_to_digit_map(self) -> Mapping[str, str]:
        return TURKEY_ALPHA_TO_DIGIT

    def normalize(self, raw_text: str) -> str:
        return _normalize_alnum(raw_text)

    def is_valid(self, plate_text: str) -> bool:
        normalized = self.normalize(plate_text)
        return any(p.fullmatch(normalized) is not None for p in self.regex_patterns)

    def correct_confusions(self, raw_text: str) -> str:
        normalized = self.normalize(raw_text)
        if self.is_valid(normalized) or len(normalized) != 7:
            return normalized
        layouts = (((3, 5, 6), (0, 1, 2, 4)),)
        return _apply_positional_layouts(
            normalized,
            layouts,
            self.alpha_to_digit_map,
            self.digit_to_alpha_map,
            self.regex_patterns,
        )

    def sample_plate(self, rng: random.Random) -> str:
        prefix = "".join(rng.choice(STANDARD_LETTERS) for _ in range(3))
        d1 = rng.randint(0, 9)
        mid = rng.choice(STANDARD_LETTERS)
        d2 = f"{rng.randint(0, 99):02d}"
        return f"{prefix}{d1}{mid}{d2}"


class MexicoCountryProfile:
    @property
    def country_code(self) -> str:
        return "MX"

    @property
    def charset(self) -> str:
        return STANDARD_LATIN_CHARSET

    @property
    def regex_patterns(self) -> tuple[Pattern[str], ...]:
        return MEXICO_PATTERNS

    @property
    def digit_to_alpha_map(self) -> Mapping[str, str]:
        return TURKEY_DIGIT_TO_ALPHA

    @property
    def alpha_to_digit_map(self) -> Mapping[str, str]:
        return TURKEY_ALPHA_TO_DIGIT

    def normalize(self, raw_text: str) -> str:
        return _normalize_alnum(raw_text)

    def is_valid(self, plate_text: str) -> bool:
        normalized = self.normalize(plate_text)
        return any(p.fullmatch(normalized) is not None for p in self.regex_patterns)

    def correct_confusions(self, raw_text: str) -> str:
        normalized = self.normalize(raw_text)
        if self.is_valid(normalized) or len(normalized) != 7:
            return normalized
        layouts = (((3, 4, 5), (0, 1, 2, 6)),)
        return _apply_positional_layouts(
            normalized,
            layouts,
            self.alpha_to_digit_map,
            self.digit_to_alpha_map,
            self.regex_patterns,
        )

    def sample_plate(self, rng: random.Random) -> str:
        prefix = "".join(rng.choice(STANDARD_LETTERS) for _ in range(3))
        digits = f"{rng.randint(100, 999)}"
        suffix = rng.choice(STANDARD_LETTERS)
        return f"{prefix}{digits}{suffix}"


class UnitedArabEmiratesCountryProfile:
    @property
    def country_code(self) -> str:
        return "AE"

    @property
    def charset(self) -> str:
        return STANDARD_LATIN_CHARSET

    @property
    def regex_patterns(self) -> tuple[Pattern[str], ...]:
        return UAE_PATTERNS

    @property
    def digit_to_alpha_map(self) -> Mapping[str, str]:
        return TURKEY_DIGIT_TO_ALPHA

    @property
    def alpha_to_digit_map(self) -> Mapping[str, str]:
        return TURKEY_ALPHA_TO_DIGIT

    def normalize(self, raw_text: str) -> str:
        return _normalize_alnum(raw_text)

    def is_valid(self, plate_text: str) -> bool:
        normalized = self.normalize(plate_text)
        return any(p.fullmatch(normalized) is not None for p in self.regex_patterns)

    def correct_confusions(self, raw_text: str) -> str:
        normalized = self.normalize(raw_text)
        if self.is_valid(normalized) or len(normalized) < 5 or len(normalized) > 7:
            return normalized
        layouts: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for alpha_len in (1, 2):
            digit_len = len(normalized) - alpha_len
            if 4 <= digit_len <= 5:
                layouts.append(
                    (
                        tuple(range(alpha_len, len(normalized))),
                        tuple(range(alpha_len)),
                    )
                )
        return _apply_positional_layouts(
            normalized,
            tuple(layouts),
            self.alpha_to_digit_map,
            self.digit_to_alpha_map,
            self.regex_patterns,
        )

    def sample_plate(self, rng: random.Random) -> str:
        alpha_len = rng.choice((1, 2))
        prefix = "".join(rng.choice(STANDARD_LETTERS) for _ in range(alpha_len))
        digit_len = rng.choice((4, 5))
        low = 10 ** (digit_len - 1)
        high = (10**digit_len) - 1
        return f"{prefix}{rng.randint(low, high)}"


class SaudiArabiaCountryProfile:
    @property
    def country_code(self) -> str:
        return "SA"

    @property
    def charset(self) -> str:
        return STANDARD_LATIN_CHARSET

    @property
    def regex_patterns(self) -> tuple[Pattern[str], ...]:
        return SAUDI_ARABIA_PATTERNS

    @property
    def digit_to_alpha_map(self) -> Mapping[str, str]:
        return TURKEY_DIGIT_TO_ALPHA

    @property
    def alpha_to_digit_map(self) -> Mapping[str, str]:
        return TURKEY_ALPHA_TO_DIGIT

    def normalize(self, raw_text: str) -> str:
        return _normalize_alnum(raw_text)

    def is_valid(self, plate_text: str) -> bool:
        normalized = self.normalize(plate_text)
        return any(p.fullmatch(normalized) is not None for p in self.regex_patterns)

    def correct_confusions(self, raw_text: str) -> str:
        normalized = self.normalize(raw_text)
        if self.is_valid(normalized) or len(normalized) != 7:
            return normalized
        layouts = (((0, 1, 2, 3), (4, 5, 6)),)
        return _apply_positional_layouts(
            normalized,
            layouts,
            self.alpha_to_digit_map,
            self.digit_to_alpha_map,
            self.regex_patterns,
        )

    def sample_plate(self, rng: random.Random) -> str:
        digits = f"{rng.randint(0, 9999):04d}"
        letters = "".join(rng.choice(STANDARD_LETTERS) for _ in range(3))
        return f"{digits}{letters}"


class _FixedLayoutCountryProfile:
    _code: str = ""
    _patterns: tuple[Pattern[str], ...] = ()
    _expected_length: int = 0
    _layouts: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = ()

    @property
    def country_code(self) -> str:
        return self._code

    @property
    def charset(self) -> str:
        return STANDARD_LATIN_CHARSET

    @property
    def regex_patterns(self) -> tuple[Pattern[str], ...]:
        return self._patterns

    @property
    def digit_to_alpha_map(self) -> Mapping[str, str]:
        return TURKEY_DIGIT_TO_ALPHA

    @property
    def alpha_to_digit_map(self) -> Mapping[str, str]:
        return TURKEY_ALPHA_TO_DIGIT

    def normalize(self, raw_text: str) -> str:
        return _normalize_alnum(raw_text)

    def is_valid(self, plate_text: str) -> bool:
        normalized = self.normalize(plate_text)
        return any(p.fullmatch(normalized) is not None for p in self._patterns)

    def correct_confusions(self, raw_text: str) -> str:
        normalized = self.normalize(raw_text)
        if self.is_valid(normalized) or len(normalized) != self._expected_length:
            return normalized
        return _apply_positional_layouts(
            normalized,
            self._layouts,
            self.alpha_to_digit_map,
            self.digit_to_alpha_map,
            self._patterns,
        )


class IndiaCountryProfile(_FixedLayoutCountryProfile):
    _code = "IN"
    _patterns = INDIA_PATTERNS
    _expected_length = 10
    _layouts = (((2, 3, 6, 7, 8, 9), (0, 1, 4, 5)),)

    def sample_plate(self, rng: random.Random) -> str:
        state = "".join(rng.choice(STANDARD_LETTERS) for _ in range(2))
        rto = f"{rng.randint(1, 99):02d}"
        series = "".join(rng.choice(STANDARD_LETTERS) for _ in range(2))
        number = f"{rng.randint(1, 9999):04d}"
        return f"{state}{rto}{series}{number}"


class AustraliaCountryProfile(_FixedLayoutCountryProfile):
    _code = "AU"
    _patterns = AUSTRALIA_PATTERNS
    _expected_length = 6
    _layouts = (((2, 3), (0, 1, 4, 5)),)

    def sample_plate(self, rng: random.Random) -> str:
        prefix = "".join(rng.choice(STANDARD_LETTERS) for _ in range(2))
        digits = f"{rng.randint(0, 99):02d}"
        suffix = "".join(rng.choice(STANDARD_LETTERS) for _ in range(2))
        return f"{prefix}{digits}{suffix}"


class PolandCountryProfile(_FixedLayoutCountryProfile):
    _code = "PL"
    _patterns = POLAND_PATTERNS
    _expected_length = 7
    _layouts = (((2, 3, 4, 5), (0, 1, 6)),)

    def sample_plate(self, rng: random.Random) -> str:
        prefix = "".join(rng.choice(STANDARD_LETTERS) for _ in range(2))
        digits = f"{rng.randint(0, 9999):04d}"
        suffix = rng.choice(STANDARD_LETTERS)
        return f"{prefix}{digits}{suffix}"


class SwedenCountryProfile(_FixedLayoutCountryProfile):
    _code = "SE"
    _patterns = SWEDEN_PATTERNS
    _expected_length = 6
    _layouts = (((3, 4), (0, 1, 2, 5)),)

    def sample_plate(self, rng: random.Random) -> str:
        prefix = "".join(rng.choice(STANDARD_LETTERS) for _ in range(3))
        digits = f"{rng.randint(0, 99):02d}"
        suffix = rng.choice(STANDARD_LETTERS)
        return f"{prefix}{digits}{suffix}"


class CountryProfileRegistry:
    def __init__(self) -> None:
        self._profiles: dict[str, CountryProfileProtocol] = {}

    def register(self, profile: CountryProfileProtocol) -> None:
        code = profile.country_code.strip().upper()
        if not code:
            raise ConfigurationError("Country profile code must not be empty")
        self._profiles[code] = profile

    def get(self, country_code: str) -> CountryProfileProtocol:
        code = country_code.strip().upper()
        profile = self._profiles.get(code)
        if profile is None:
            raise ConfigurationError(
                f"Unregistered country profile: {country_code}",
                details={"country_code": country_code, "available": list(self.available_codes())},
            )
        return profile

    def available_codes(self) -> tuple[str, ...]:
        return tuple(sorted(self._profiles.keys()))


DEFAULT_PROFILE_REGISTRY: Final[CountryProfileRegistry] = CountryProfileRegistry()
DEFAULT_PROFILE_REGISTRY.register(TurkeyCountryProfile())
DEFAULT_PROFILE_REGISTRY.register(GermanyCountryProfile())
DEFAULT_PROFILE_REGISTRY.register(UnitedKingdomCountryProfile())
DEFAULT_PROFILE_REGISTRY.register(AzerbaijanCountryProfile())
DEFAULT_PROFILE_REGISTRY.register(FranceCountryProfile())
DEFAULT_PROFILE_REGISTRY.register(ItalyCountryProfile())
DEFAULT_PROFILE_REGISTRY.register(SpainCountryProfile())
DEFAULT_PROFILE_REGISTRY.register(NetherlandsCountryProfile())
DEFAULT_PROFILE_REGISTRY.register(UnitedStatesCountryProfile())
DEFAULT_PROFILE_REGISTRY.register(CanadaCountryProfile())
DEFAULT_PROFILE_REGISTRY.register(JapanCountryProfile())
DEFAULT_PROFILE_REGISTRY.register(SouthKoreaCountryProfile())
DEFAULT_PROFILE_REGISTRY.register(BrazilCountryProfile())
DEFAULT_PROFILE_REGISTRY.register(MexicoCountryProfile())
DEFAULT_PROFILE_REGISTRY.register(UnitedArabEmiratesCountryProfile())
DEFAULT_PROFILE_REGISTRY.register(SaudiArabiaCountryProfile())
DEFAULT_PROFILE_REGISTRY.register(IndiaCountryProfile())
DEFAULT_PROFILE_REGISTRY.register(AustraliaCountryProfile())
DEFAULT_PROFILE_REGISTRY.register(PolandCountryProfile())
DEFAULT_PROFILE_REGISTRY.register(SwedenCountryProfile())


def register_country_profile(profile: CountryProfileProtocol) -> None:
    DEFAULT_PROFILE_REGISTRY.register(profile)


def get_country_profile(country_code: str) -> CountryProfileProtocol:
    return DEFAULT_PROFILE_REGISTRY.get(country_code)


__all__ = [
    "AUSTRALIA_PATTERNS",
    "AZERBAIJAN_PATTERNS",
    "BRAZIL_PATTERNS",
    "CANADA_PATTERNS",
    "FRANCE_PATTERNS",
    "GERMANY_PATTERNS",
    "INDIA_PATTERNS",
    "ITALY_PATTERNS",
    "JAPAN_PATTERNS",
    "MEXICO_PATTERNS",
    "NETHERLANDS_PATTERNS",
    "POLAND_PATTERNS",
    "SAUDI_ARABIA_PATTERNS",
    "SOUTH_KOREA_PATTERNS",
    "SPAIN_PATTERNS",
    "STANDARD_LATIN_CHARSET",
    "SWEDEN_PATTERNS",
    "TURKEY_ALPHA_TO_DIGIT",
    "TURKEY_CHARSET",
    "TURKEY_DIGIT_TO_ALPHA",
    "TURKEY_PATTERNS",
    "UAE_PATTERNS",
    "UK_PATTERNS",
    "US_PATTERNS",
    "AustraliaCountryProfile",
    "AzerbaijanCountryProfile",
    "BrazilCountryProfile",
    "CanadaCountryProfile",
    "CountryProfileProtocol",
    "CountryProfileRegistry",
    "FranceCountryProfile",
    "GermanyCountryProfile",
    "IndiaCountryProfile",
    "ItalyCountryProfile",
    "JapanCountryProfile",
    "MexicoCountryProfile",
    "NetherlandsCountryProfile",
    "PolandCountryProfile",
    "SaudiArabiaCountryProfile",
    "SouthKoreaCountryProfile",
    "SpainCountryProfile",
    "SwedenCountryProfile",
    "TurkeyCountryProfile",
    "UnitedArabEmiratesCountryProfile",
    "UnitedKingdomCountryProfile",
    "UnitedStatesCountryProfile",
    "get_country_profile",
    "register_country_profile",
]
