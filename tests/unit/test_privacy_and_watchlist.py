import pytest

from plate_vision.data.models import BoundingBox
from plate_vision.pipeline.privacy import (
    RedactionConfig,
    RedactionMode,
    redact_regions,
)
from plate_vision.pipeline.watchlist import (
    WatchlistCategory,
    WatchlistEntry,
    WatchlistMatcher,
)
from plate_vision.utils.exceptions import ConfigurationError, InvalidInputError


def test_redact_regions_all_modes_and_validation() -> None:
    with pytest.raises(InvalidInputError):
        redact_regions([], [])
    with pytest.raises(InvalidInputError):
        redact_regions([[1, 2], [3]], [])

    grid = [[100 if (r + c) % 2 == 0 else 200 for c in range(8)] for r in range(8)]
    bbox = BoundingBox(x_min=2.0, y_min=2.0, x_max=6.0, y_max=6.0)

    solid = redact_regions(
        grid,
        [bbox],
        RedactionConfig(mode=RedactionMode.SOLID_FILL, fill_value=0),
    )
    assert solid[0][0] == 100
    assert solid[3][3] == 0
    assert grid[3][3] == 100

    pixelated = redact_regions(
        grid,
        [bbox],
        RedactionConfig(mode=RedactionMode.PIXELATE, block_size=2),
    )
    assert pixelated[2][2] == pixelated[2][3] == pixelated[3][2] == pixelated[3][3]
    assert pixelated[2][2] == 150

    blurred = redact_regions(
        grid,
        [bbox],
        RedactionConfig(mode=RedactionMode.BLUR),
    )
    assert blurred[0][0] == 100
    assert 130 <= blurred[3][3] <= 170


def test_watchlist_matcher_exact_fuzzy_and_errors() -> None:
    with pytest.raises(ConfigurationError):
        WatchlistMatcher(max_edit_distance=-1)

    entries = [
        WatchlistEntry(
            plate_text="34-ABC-123",
            country_code="TR",
            category=WatchlistCategory.STOLEN,
            note="reported stolen",
        ),
        WatchlistEntry(
            plate_text="34-ABC-999",
            country_code="TR",
            category=WatchlistCategory.VIP,
        ),
        WatchlistEntry(
            plate_text="B-AB-1234",
            country_code="DE",
            category=WatchlistCategory.BLOCKLIST,
        ),
    ]
    exact_matcher = WatchlistMatcher(entries=entries, max_edit_distance=0)
    exact_hits = exact_matcher.check("34 ABC 123", country_code="TR")
    assert len(exact_hits) == 1
    assert exact_hits[0].exact_match is True
    assert exact_hits[0].entry.category == WatchlistCategory.STOLEN
    assert exact_matcher.check("34 ABD 123", country_code="TR") == ()

    fuzzy_matcher = WatchlistMatcher(entries=entries, max_edit_distance=1)
    fuzzy_hits = fuzzy_matcher.check("34ABD123", country_code="TR")
    assert len(fuzzy_hits) == 1
    assert fuzzy_hits[0].exact_match is False
    assert fuzzy_hits[0].edit_distance == 1
    assert fuzzy_matcher.check("34ABC123", country_code="DE") == ()
