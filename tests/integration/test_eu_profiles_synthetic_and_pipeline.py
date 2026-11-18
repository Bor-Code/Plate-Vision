import random

from plate_vision.data.profiles import get_country_profile
from plate_vision.data.synthetic import SyntheticPlateConfig, generate_synthetic_sample
from plate_vision.pipeline.postprocess import MultiCountryPostProcessor


def test_eu_profiles_synthetic_generation_and_multi_country_postprocess() -> None:
    processor = MultiCountryPostProcessor()
    assert processor.country_codes == (
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
    )

    es_res = processor.resolve("9234 8CD")
    assert es_res.country_code == "ES"
    assert es_res.corrected_text == "9234BCD"
    assert es_res.is_valid_format is True
    assert es_res.was_corrected is True

    nl_res = processor.resolve("45-KBB-7")
    assert nl_res.country_code == "NL"
    assert nl_res.corrected_text == "45KBB7"
    assert nl_res.is_valid_format is True

    fr_res = processor.resolve("AB-123-CD")
    assert fr_res.country_code == "FR"
    assert fr_res.corrected_text == "AB123CD"
    assert fr_res.is_valid_format is True

    rng = random.Random(77)
    cfg = SyntheticPlateConfig()
    for code in ("FR", "IT", "ES", "NL"):
        profile = get_country_profile(code)
        plate_text = profile.sample_plate(rng)
        assert profile.is_valid(plate_text) is True
        grid, png_bytes, bbox = generate_synthetic_sample(plate_text, cfg, rng)
        assert len(grid) == 32
        assert len(png_bytes) > 64
        assert bbox.width > 0
