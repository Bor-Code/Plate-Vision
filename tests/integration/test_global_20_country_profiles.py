import random

from plate_vision.data.profiles import get_country_profile
from plate_vision.data.synthetic import SyntheticPlateConfig, generate_synthetic_sample
from plate_vision.pipeline.postprocess import MultiCountryPostProcessor


def test_global_20_country_profiles_resolution_and_synthesis() -> None:
    processor = MultiCountryPostProcessor()
    assert len(processor.country_codes) == 28
    assert {
        "TR",
        "DE",
        "GB",
        "AZ",
        "FR",
        "IT",
        "ES",
        "NL",
        "US",
        "CA",
        "JP",
        "KR",
        "BR",
        "MX",
        "AE",
        "SA",
        "IN",
        "AU",
        "PL",
        "SE",
    }.issubset(set(processor.country_codes))

    in_res = processor.resolve("MH-I2-AB-1234")
    assert in_res.country_code == "IN"
    assert in_res.corrected_text == "MH12AB1234"
    assert in_res.is_valid_format is True
    assert in_res.was_corrected is True

    au_res = processor.resolve("A8-I2-CK")
    assert au_res.country_code == "AU"
    assert au_res.corrected_text == "AB12CK"
    assert au_res.is_valid_format is True
    assert au_res.was_corrected is True

    pl_res = processor.resolve("WA-1234-K")
    assert pl_res.country_code == "PL"
    assert pl_res.corrected_text == "WA1234K"
    assert pl_res.is_valid_format is True

    se_res = processor.resolve("ABC-12-X")
    assert se_res.country_code == "SE"
    assert se_res.corrected_text == "ABC12X"
    assert se_res.is_valid_format is True

    rng = random.Random(707)
    cfg = SyntheticPlateConfig()
    for code in ("IN", "AU", "PL", "SE"):
        profile = get_country_profile(code)
        plate = profile.sample_plate(rng)
        assert profile.is_valid(plate) is True
        grid, png_bytes, bbox = generate_synthetic_sample(plate, cfg, rng)
        assert len(grid) == 32
        assert len(png_bytes) > 64
        assert bbox.width > 0
