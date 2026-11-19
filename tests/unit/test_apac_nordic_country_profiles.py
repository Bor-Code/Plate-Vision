import random

from plate_vision.data.profiles import (
    STANDARD_LATIN_CHARSET,
    AustraliaCountryProfile,
    IndiaCountryProfile,
    PolandCountryProfile,
    SwedenCountryProfile,
    get_country_profile,
)


def test_india_and_australia_profiles_validation_correction_and_sampling() -> None:
    ind = IndiaCountryProfile()
    assert get_country_profile("IN").country_code == "IN"
    assert ind.charset == STANDARD_LATIN_CHARSET
    assert ind.is_valid("MH-12-AB-1234") is True
    assert ind.is_valid("MH-123-AB-12") is False
    assert ind.correct_confusions("MH I2 8B I23O") == "MH12BB1230"
    assert ind.correct_confusions("SHORT") == "SHORT"

    au = AustraliaCountryProfile()
    assert get_country_profile("AU").country_code == "AU"
    assert au.charset == STANDARD_LATIN_CHARSET
    assert au.is_valid("AB-12-CD") is True
    assert au.is_valid("ABC-123") is False
    assert au.correct_confusions("A8 I2 CK") == "AB12CK"
    assert au.correct_confusions("SHORT") == "SHORT"

    rng = random.Random(505)
    for _ in range(25):
        assert ind.is_valid(ind.sample_plate(rng)) is True
        assert au.is_valid(au.sample_plate(rng)) is True


def test_poland_and_sweden_profiles_validation_correction_and_sampling() -> None:
    pl = PolandCountryProfile()
    assert get_country_profile("PL").country_code == "PL"
    assert pl.is_valid("WA-1234-K") is True
    assert pl.is_valid("WA-123-KR") is False
    assert pl.correct_confusions("W8 I2S4 8") == "WB1254B"
    assert pl.correct_confusions("SHORT") == "SHORT"

    se = SwedenCountryProfile()
    assert get_country_profile("SE").country_code == "SE"
    assert se.is_valid("ABC-12-X") is True
    assert se.is_valid("AB-123-X") is False
    assert se.correct_confusions("A8C I2 5") == "ABC12S"
    assert se.correct_confusions("SHORT") == "SHORT"

    rng = random.Random(606)
    for _ in range(25):
        assert pl.is_valid(pl.sample_plate(rng)) is True
        assert se.is_valid(se.sample_plate(rng)) is True
