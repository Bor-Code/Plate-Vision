import random

from plate_vision.data.profiles import get_country_profile
from plate_vision.data.synthetic import SyntheticPlateConfig, generate_synthetic_sample
from plate_vision.pipeline.postprocess import MultiCountryPostProcessor


def test_global_16_country_profiles_resolution_and_synthesis() -> None:
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
        "BR",
        "MX",
        "AE",
        "SA",
    }.issubset(set(processor.country_codes))

    br_res = processor.resolve("ABC-1-D-23")
    assert br_res.country_code == "BR"
    assert br_res.corrected_text == "ABC1D23"
    assert br_res.is_valid_format is True

    latam_processor = MultiCountryPostProcessor(country_codes=("BR", "MX", "AE", "SA"))
    br_corrected = latam_processor.resolve("A8C-1-D-23")
    assert br_corrected.country_code == "BR"
    assert br_corrected.corrected_text == "ABC1D23"
    assert br_corrected.is_valid_format is True
    assert br_corrected.was_corrected is True

    mx_res = processor.resolve("ABC-123-D")
    assert mx_res.country_code == "MX"
    assert mx_res.corrected_text == "ABC123D"
    assert mx_res.is_valid_format is True

    ae_res = processor.resolve("A-12345")
    assert ae_res.country_code == "AE"
    assert ae_res.corrected_text == "A12345"
    assert ae_res.is_valid_format is True

    sa_res = processor.resolve("1234-AEU")
    assert sa_res.country_code == "SA"
    assert sa_res.corrected_text == "1234AEU"
    assert sa_res.is_valid_format is True

    rng = random.Random(404)
    cfg = SyntheticPlateConfig()
    for code in ("BR", "MX", "AE", "SA"):
        profile = get_country_profile(code)
        plate = profile.sample_plate(rng)
        assert profile.is_valid(plate) is True
        grid, png_bytes, bbox = generate_synthetic_sample(plate, cfg, rng)
        assert len(grid) == 32
        assert len(png_bytes) > 64
        assert bbox.width > 0
