from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from plate_vision.data.profiles import (
    CountryProfileProtocol,
    CountryProfileRegistry,
    get_country_profile,
)
from plate_vision.utils.exceptions import ConfigurationError


class PostProcessedPlate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_text: str
    normalized_text: str
    corrected_text: str
    country_code: str = Field(min_length=2)
    is_valid_format: bool
    was_corrected: bool


def postprocess_plate_text(
    raw_text: str,
    *,
    country_code: str = "TR",
    profile: CountryProfileProtocol | None = None,
) -> PostProcessedPlate:
    active_profile = profile if profile is not None else get_country_profile(country_code)
    normalized = active_profile.normalize(raw_text)
    corrected = active_profile.correct_confusions(normalized)
    is_valid = active_profile.is_valid(corrected)
    return PostProcessedPlate(
        raw_text=raw_text,
        normalized_text=normalized,
        corrected_text=corrected,
        country_code=active_profile.country_code,
        is_valid_format=is_valid,
        was_corrected=(corrected != normalized),
    )


class MultiCountryPostProcessor:
    def __init__(
        self,
        country_codes: Sequence[str] = ("TR", "DE", "GB", "AZ", "FR", "IT", "ES", "NL"),
        registry: CountryProfileRegistry | None = None,
    ) -> None:
        if not country_codes:
            raise ConfigurationError("MultiCountryPostProcessor requires at least one country code")
        self._profiles: list[CountryProfileProtocol] = []
        for code in country_codes:
            resolved = registry.get(code) if registry is not None else get_country_profile(code)
            self._profiles.append(resolved)

    @property
    def country_codes(self) -> tuple[str, ...]:
        return tuple(p.country_code for p in self._profiles)

    def resolve(self, raw_text: str) -> PostProcessedPlate:
        for profile in self._profiles:
            normalized = profile.normalize(raw_text)
            if profile.is_valid(normalized):
                return PostProcessedPlate(
                    raw_text=raw_text,
                    normalized_text=normalized,
                    corrected_text=normalized,
                    country_code=profile.country_code,
                    is_valid_format=True,
                    was_corrected=False,
                )

        for profile in self._profiles:
            candidate = postprocess_plate_text(raw_text, profile=profile)
            if candidate.is_valid_format:
                return candidate

        return postprocess_plate_text(raw_text, profile=self._profiles[0])


__all__ = [
    "MultiCountryPostProcessor",
    "PostProcessedPlate",
    "postprocess_plate_text",
]
