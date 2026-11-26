import random

from plate_vision.data.profiles import (
    STANDARD_LATIN_CHARSET,
    AustriaCountryProfile,
    BelgiumCountryProfile,
    PortugalCountryProfile,
    SwitzerlandCountryProfile,
    get_country_profile,
)


def test_switzerland_and_austria_profiles_validation_correction_and_sampling() -> None:
    ch = SwitzerlandCountryProfile()
    assert get_country_profile("CH").country_code == "CH"
    assert ch.charset == STANDARD_LATIN_CHARSET
    assert ch.is_valid("ZH-123456") is True
    assert ch.is_valid("ZH-012345") is False
    assert ch.correct_confusions("2H I234SO") == "ZH123450"
    assert ch.correct_confusions("SHORT") == "SHORT"

    at = AustriaCountryProfile()
    assert get_country_profile("AT").country_code == "AT"
    assert at.charset == STANDARD_LATIN_CHARSET
    assert at.is_valid("WZ-123-K") is True
    assert at.is_valid("W-1234-K") is False
    assert at.correct_confusions("W2 I2O 8") == "WZ120B"
    assert at.correct_confusions("SHORT") == "SHORT"

    rng = random.Random(808)
    for _ in range(25):
        assert ch.is_valid(ch.sample_plate(rng)) is True
        assert at.is_valid(at.sample_plate(rng)) is True


def test_belgium_and_portugal_profiles_validation_correction_and_sampling() -> None:
    be = BelgiumCountryProfile()
    assert get_country_profile("BE").country_code == "BE"
    assert be.is_valid("123-ABC") is True
    assert be.is_valid("1234-AB") is False
    assert be.correct_confusions("I2O A8C") == "120ABC"
    assert be.correct_confusions("SHORT") == "SHORT"

    pt = PortugalCountryProfile()
    assert get_country_profile("PT").country_code == "PT"
    assert pt.is_valid("12-AB-34") is True
    assert pt.is_valid("123-AB-4") is False
    assert pt.correct_confusions("I2 A8 S4") == "12AB54"
    assert pt.correct_confusions("SHORT") == "SHORT"

    rng = random.Random(909)
    for _ in range(25):
        assert be.is_valid(be.sample_plate(rng)) is True
        assert pt.is_valid(pt.sample_plate(rng)) is True
