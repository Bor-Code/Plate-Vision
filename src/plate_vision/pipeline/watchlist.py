from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from plate_vision.data.profiles import get_country_profile
from plate_vision.utils.exceptions import ConfigurationError


class WatchlistCategory(StrEnum):
    ALLOWLIST = "allowlist"
    BLOCKLIST = "blocklist"
    VIP = "vip"
    STOLEN = "stolen"


class WatchlistEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plate_text: str = Field(min_length=2)
    country_code: str = Field(default="TR", min_length=2)
    category: WatchlistCategory
    note: str = Field(default="")


class WatchlistMatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entry: WatchlistEntry
    matched_plate_text: str
    edit_distance: int = Field(ge=0)
    exact_match: bool


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev_row = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, start=1):
        curr_row = [i]
        for j, ch_b in enumerate(b, start=1):
            cost = 0 if ch_a == ch_b else 1
            curr_row.append(
                min(
                    curr_row[j - 1] + 1,
                    prev_row[j] + 1,
                    prev_row[j - 1] + cost,
                )
            )
        prev_row = curr_row
    return prev_row[-1]


class WatchlistMatcher:
    def __init__(
        self,
        entries: Sequence[WatchlistEntry] = (),
        max_edit_distance: int = 0,
    ) -> None:
        if max_edit_distance < 0:
            raise ConfigurationError(
                "max_edit_distance must be >= 0",
                details={"max_edit_distance": max_edit_distance},
            )
        self.max_edit_distance = max_edit_distance
        self._normalized_entries: list[tuple[str, str, WatchlistEntry]] = []
        for entry in entries:
            code = entry.country_code.strip().upper()
            profile = get_country_profile(code)
            normalized_plate = profile.normalize(entry.plate_text)
            self._normalized_entries.append((code, normalized_plate, entry))

    def check(
        self,
        plate_text: str,
        country_code: str = "TR",
    ) -> tuple[WatchlistMatch, ...]:
        target_code = country_code.strip().upper()
        profile = get_country_profile(target_code)
        query_norm = profile.normalize(plate_text)
        matches: list[WatchlistMatch] = []

        for entry_code, entry_norm, entry in self._normalized_entries:
            if entry_code != target_code:
                continue
            dist = _levenshtein_distance(query_norm, entry_norm)
            if dist <= self.max_edit_distance:
                matches.append(
                    WatchlistMatch(
                        entry=entry,
                        matched_plate_text=entry_norm,
                        edit_distance=dist,
                        exact_match=(dist == 0),
                    )
                )

        matches.sort(key=lambda item: (item.edit_distance, item.entry.plate_text))
        return tuple(matches)


__all__ = [
    "WatchlistCategory",
    "WatchlistEntry",
    "WatchlistMatch",
    "WatchlistMatcher",
]
