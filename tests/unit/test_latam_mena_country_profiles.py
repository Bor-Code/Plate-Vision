import random

from plate_vision.data.profiles import (
    BrazilCountryProfile,
    MexicoCountryProfile,
    SaudiArabiaCountryProfile,
    UnitedArabEmiratesCountryProfile,
    get_country_profile,
)


def test_brazil_and_mexico_profiles_validation_correction_and_sampling() -> None:
    br = BrazilCountryProfile()
    assert get_country_profile("BR").country_code == "BR"
    assert br.is_valid("ABC1D23") is True
    assert br.is_valid("ABC1234") is False
    assert br.correct_confusions("A8C I D 2S") == "ABC1D25"

    mx = MexicoCountryProfile()
    assert get_country_profile("MX").country_code == "MX"
    assert mx.is_valid("ABC-123-D") is True
    assert mx.is_valid("AB-123-CD") is False
    assert mx.correct_confusions("A8C-I2O-5") == "ABC120S"

    rng = random.Random(101)
    for _ in range(25):
        assert br.is_valid(br.sample_plate(rng)) is True
        assert mx.is_valid(mx.sample_plate(rng)) is True


def test_uae_and_saudi_profiles_validation_correction_and_sampling() -> None:
    ae = UnitedArabEmiratesCountryProfile()
    assert get_country_profile("AE").country_code == "AE"
    assert ae.is_valid("A 12345") is True
    assert ae.is_valid("AB 9876") is True
    assert ae.is_valid("A 0123") is False
    assert ae.correct_confusions("8 I2S45") == "B12545"

    sa = SaudiArabiaCountryProfile()
    assert get_country_profile("SA").country_code == "SA"
    assert sa.is_valid("1234 ABC") is True
    assert sa.is_valid("123 ABC") is False
    assert sa.correct_confusions("I23O A8C") == "1230ABC"

    rng = random.Random(202)
    for _ in range(25):
        assert ae.is_valid(ae.sample_plate(rng)) is True
        assert sa.is_valid(sa.sample_plate(rng)) is True
