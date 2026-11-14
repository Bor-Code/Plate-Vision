from pydantic import BaseModel, ConfigDict, Field

from plate_vision.data.profiles import CountryProfileProtocol, get_country_profile


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


__all__ = [
    "PostProcessedPlate",
    "postprocess_plate_text",
]
