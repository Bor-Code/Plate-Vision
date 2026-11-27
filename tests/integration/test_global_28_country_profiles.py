import random

from plate_vision.data.profiles import get_country_profile
from plate_vision.data.synthetic import SyntheticPlateConfig, generate_synthetic_sample
from plate_vision.pipeline.postprocess import MultiCountryPostProcessor


def test_global_28_country_profiles_resolution_and_synthesis() -> None:
    processor = MultiCountryPostProcessor()
    assert len(processor.country_codes) == 28
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
        "CN",
        "RU",
        "CZ",
        "SG",
    }

    cn_res = processor.resolve("AH-AY-339S")
    assert cn_res.country_code == "CN"
    assert cn_res.corrected_text == "AHAY339S"
    assert cn_res.is_valid_format is True

    ru_res = processor.resolve("A-123-BC-77")
    assert ru_res.country_code == "RU"
    assert ru_res.corrected_text == "A123BC77"
    assert ru_res.is_valid_format is True

    cz_res = processor.resolve("1A-23456")
    assert cz_res.country_code == "CZ"
    assert cz_res.corrected_text == "1A23456"
    assert cz_res.is_valid_format is True

    sg_res = processor.resolve("SBA-1234-K")
    assert sg_res.country_code == "SG"
    assert sg_res.corrected_text == "SBA1234K"
    assert sg_res.is_valid_format is True

    regional = MultiCountryPostProcessor(country_codes=("CN", "RU", "CZ", "SG"))
    cn_corr = regional.resolve("AHAY3395")
    assert cn_corr.country_code == "CN"
    assert cn_corr.corrected_text == "AHAY339S"
    assert cn_corr.was_corrected is True

    ru_corr = regional.resolve("AI238C77")
    assert ru_corr.country_code == "RU"
    assert ru_corr.corrected_text == "A123BC77"
    assert ru_corr.was_corrected is True

    cz_corr = regional.resolve("IA23456")
    assert cz_corr.country_code == "CZ"
    assert cz_corr.corrected_text == "1A23456"
    assert cz_corr.was_corrected is True

    sg_corr = regional.resolve("S8A4234K")
    assert sg_corr.country_code == "SG"
    assert sg_corr.corrected_text == "SBA4234K"
    assert sg_corr.was_corrected is True

    rng = random.Random(2028)
    cfg = SyntheticPlateConfig()
    for code in ("CN", "RU", "CZ", "SG"):
        profile = get_country_profile(code)
        plate = profile.sample_plate(rng)
        assert profile.is_valid(plate) is True
        grid, png_bytes, bbox = generate_synthetic_sample(plate, cfg, rng)
        assert len(grid) == 32
        assert len(png_bytes) > 64
        assert bbox.width > 0
