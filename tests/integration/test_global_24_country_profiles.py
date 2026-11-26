import random

from plate_vision.data.profiles import get_country_profile
from plate_vision.data.synthetic import SyntheticPlateConfig, generate_synthetic_sample
from plate_vision.pipeline.postprocess import MultiCountryPostProcessor


def test_global_24_country_profiles_resolution_and_synthesis() -> None:
    processor = MultiCountryPostProcessor()
    assert len(processor.country_codes) == 24
    assert set(processor.country_codes) == {
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
        "CH",
        "AT",
        "BE",
        "PT",
    }

    ch_res = processor.resolve("ZH-123456")
    assert ch_res.country_code == "CH"
    assert ch_res.corrected_text == "ZH123456"
    assert ch_res.is_valid_format is True

    at_res = processor.resolve("WZ-123-K")
    assert at_res.country_code == "AT"
    assert at_res.corrected_text == "WZ123K"
    assert at_res.is_valid_format is True

    be_res = processor.resolve("123-ABC")
    assert be_res.country_code == "BE"
    assert be_res.corrected_text == "123ABC"
    assert be_res.is_valid_format is True

    pt_res = processor.resolve("12-AB-34")
    assert pt_res.country_code == "PT"
    assert pt_res.corrected_text == "12AB34"
    assert pt_res.is_valid_format is True

    regional = MultiCountryPostProcessor(country_codes=("CH", "AT", "BE", "PT"))
    ch_corr = regional.resolve("2H-I23456")
    assert ch_corr.country_code == "CH"
    assert ch_corr.corrected_text == "ZH123456"
    assert ch_corr.was_corrected is True

    at_corr = regional.resolve("W2-I23-K")
    assert at_corr.country_code == "AT"
    assert at_corr.corrected_text == "WZ123K"
    assert at_corr.was_corrected is True

    be_corr = regional.resolve("I23-A8C")
    assert be_corr.country_code == "BE"
    assert be_corr.corrected_text == "123ABC"
    assert be_corr.was_corrected is True

    pt_corr = regional.resolve("I2-A8-34")
    assert pt_corr.country_code == "PT"
    assert pt_corr.corrected_text == "12AB34"
    assert pt_corr.was_corrected is True

    rng = random.Random(1010)
    cfg = SyntheticPlateConfig()
    for code in ("CH", "AT", "BE", "PT"):
        profile = get_country_profile(code)
        plate = profile.sample_plate(rng)
        assert profile.is_valid(plate) is True
        grid, png_bytes, bbox = generate_synthetic_sample(plate, cfg, rng)
        assert len(grid) == 32
        assert len(png_bytes) > 64
        assert bbox.width > 0
