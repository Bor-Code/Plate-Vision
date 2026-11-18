import random

from plate_vision.data.profiles import (
    CanadaCountryProfile,
    JapanCountryProfile,
    SouthKoreaCountryProfile,
    UnitedStatesCountryProfile,
    get_country_profile,
)


def test_us_and_canada_profiles_validation_correction_and_sampling() -> None:
    us = UnitedStatesCountryProfile()
    assert get_country_profile("US").country_code == "US"
    assert us.is_valid("ABC-1234") is True
    assert us.is_valid("7XYZ456") is True
    assert us.is_valid("AB-123") is False
    assert us.correct_confusions("A8C I2O4") == "ABC1204"
    assert us.correct_confusions("7XYZ 4S6") == "7XYZ456"

    ca = CanadaCountryProfile()
    assert get_country_profile("CA").country_code == "CA"
    assert ca.is_valid("ABCD 123") is True
    assert ca.is_valid("ABC-123") is True
    assert ca.is_valid("AB1234") is False
    assert ca.correct_confusions("A8CD I2S") == "ABCD125"
    assert ca.correct_confusions("A8C I2O") == "ABC120"

    rng = random.Random(101)
    for _ in range(25):
        assert us.is_valid(us.sample_plate(rng)) is True
        assert ca.is_valid(ca.sample_plate(rng)) is True


def test_japan_and_south_korea_profiles_validation_correction_and_sampling() -> None:
    jp = JapanCountryProfile()
    assert get_country_profile("JP").country_code == "JP"
    assert jp.is_valid("300-A-1234") is True
    assert jp.is_valid("30-A-1234") is False
    assert jp.correct_confusions("3OO-8-I2S4") == "300B1254"

    kr = SouthKoreaCountryProfile()
    assert get_country_profile("KR").country_code == "KR"
    assert kr.is_valid("12A 3456") is True
    assert kr.is_valid("123B 4567") is True
    assert kr.is_valid("1A 3456") is False
    assert kr.correct_confusions("I2-8-34S6") == "12B3456"
    assert kr.correct_confusions("I23-8-4S67") == "123B4567"

    rng = random.Random(202)
    for _ in range(25):
        assert jp.is_valid(jp.sample_plate(rng)) is True
        assert kr.is_valid(kr.sample_plate(rng)) is True
