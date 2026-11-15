import random

import pytest

from plate_vision.data.profiles import (
    AzerbaijanCountryProfile,
    GermanyCountryProfile,
    UnitedKingdomCountryProfile,
)
from plate_vision.pipeline.postprocess import MultiCountryPostProcessor
from plate_vision.utils.exceptions import ConfigurationError


def test_germany_country_profile_validation_correction_and_sampling() -> None:
    profile = GermanyCountryProfile()
    assert profile.country_code == "DE"
    assert profile.is_valid("M-AB 1234") is True
    assert profile.is_valid("B-A 99") is True
    assert profile.is_valid("AB") is False
    assert profile.is_valid("M-AB 0123") is False

    assert profile.correct_confusions("M A8 12S4") == "MAB1254"
    assert profile.correct_confusions("8 A I9") == "BAI9"
    assert profile.correct_confusions("X") == "X"

    rng = random.Random(42)
    for _ in range(5):
        sampled = profile.sample_plate(rng)
        assert profile.is_valid(sampled) is True


def test_uk_country_profile_validation_correction_and_sampling() -> None:
    profile = UnitedKingdomCountryProfile()
    assert profile.country_code == "GB"
    assert profile.is_valid("BD51 SMR") is True
    assert profile.is_valid("BD51SM") is False

    assert profile.correct_confusions("8D5I 5MR") == "BD51SMR"
    assert profile.correct_confusions("SHORT") == "SHORT"

    rng = random.Random(42)
    for _ in range(5):
        sampled = profile.sample_plate(rng)
        assert profile.is_valid(sampled) is True


def test_azerbaijan_country_profile_validation_correction_and_sampling() -> None:
    profile = AzerbaijanCountryProfile()
    assert profile.country_code == "AZ"
    assert profile.is_valid("10-AA-123") is True
    assert profile.is_valid("00-AA-123") is False

    assert profile.correct_confusions("I0-A8-I2S") == "10AB125"
    assert profile.correct_confusions("SHORT") == "SHORT"

    rng = random.Random(42)
    for _ in range(5):
        sampled = profile.sample_plate(rng)
        assert profile.is_valid(sampled) is True


def test_multi_country_post_processor_rejects_empty_list() -> None:
    with pytest.raises(ConfigurationError):
        MultiCountryPostProcessor(country_codes=())
