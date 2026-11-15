import random

from plate_vision.data.profiles import (
    FranceCountryProfile,
    ItalyCountryProfile,
    NetherlandsCountryProfile,
    SpainCountryProfile,
    get_country_profile,
)


def test_france_and_italy_profiles_validation_correction_and_sampling() -> None:
    fr = FranceCountryProfile()
    assert get_country_profile("FR").country_code == "FR"
    assert fr.is_valid("AB-123-CD") is True
    assert fr.is_valid("AI-123-CD") is False
    assert fr.correct_confusions("A8-I2O-C5") == "AB120CS"

    it = ItalyCountryProfile()
    assert get_country_profile("IT").country_code == "IT"
    assert it.is_valid("AB 456 CD") is True
    assert it.is_valid("AQ 456 CD") is False
    assert it.correct_confusions("A8 4S6 C2") == "AB456CZ"

    rng = random.Random(101)
    for _ in range(25):
        assert fr.is_valid(fr.sample_plate(rng)) is True
        assert it.is_valid(it.sample_plate(rng)) is True


def test_spain_and_netherlands_profiles_validation_correction_and_sampling() -> None:
    es = SpainCountryProfile()
    assert get_country_profile("ES").country_code == "ES"
    assert es.is_valid("1234 BBB") is True
    assert es.is_valid("1234 ABE") is False
    assert es.correct_confusions("I23O 852") == "1230BSZ"

    nl = NetherlandsCountryProfile()
    assert get_country_profile("NL").country_code == "NL"
    assert nl.is_valid("12-KBB-3") is True
    assert nl.is_valid("9-KBB-45") is True
    assert nl.is_valid("12-AAA-3") is False
    assert nl.correct_confusions("I2-K85-O") == "12KBS0"
    assert nl.correct_confusions("I-K85-4O") == "1KBS40"

    rng = random.Random(202)
    for _ in range(25):
        assert es.is_valid(es.sample_plate(rng)) is True
        assert nl.is_valid(nl.sample_plate(rng)) is True
