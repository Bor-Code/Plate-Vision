import random

from plate_vision.data.profiles import get_country_profile
from plate_vision.data.synthetic import SyntheticPlateConfig, generate_synthetic_sample
from plate_vision.pipeline.postprocess import MultiCountryPostProcessor


def test_global_12_country_profiles_resolution_and_synthesis() -> None:
    processor = MultiCountryPostProcessor()
    assert len(processor.country_codes) == 24
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
    }.issubset(set(processor.country_codes))

    us_res = processor.resolve("7XYZ 4S6")
    assert us_res.country_code == "US"
    assert us_res.corrected_text == "7XYZ456"
    assert us_res.is_valid_format is True
    assert us_res.was_corrected is True

    ca_res = processor.resolve("ABCD-012")
    assert ca_res.country_code == "CA"
    assert ca_res.corrected_text == "ABCD012"
    assert ca_res.is_valid_format is True

    jp_res = processor.resolve("300-K-1234")
    assert jp_res.country_code == "JP"
    assert jp_res.corrected_text == "300K1234"
    assert jp_res.is_valid_format is True

    kr_res = processor.resolve("90-M-3456")
    assert kr_res.country_code == "KR"
    assert kr_res.corrected_text == "90M3456"
    assert kr_res.is_valid_format is True

    rng = random.Random(303)
    cfg = SyntheticPlateConfig()
    for code in ("US", "CA", "JP", "KR"):
        profile = get_country_profile(code)
        plate = profile.sample_plate(rng)
        assert profile.is_valid(plate) is True
        grid, png_bytes, bbox = generate_synthetic_sample(plate, cfg, rng)
        assert len(grid) == 32
        assert len(png_bytes) > 64
        assert bbox.width > 0
