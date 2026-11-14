from __future__ import annotations

import random
import re
from re import Pattern
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

from plate_vision.data import (
    AugmentationConfig,
    AugmentationPipeline,
    CountryProfileRegistry,
    SyntheticPlateConfig,
    TurkeyCountryProfile,
    encode_grayscale_png,
    generate_synthetic_dataset,
    generate_synthetic_sample,
    get_country_profile,
    register_country_profile,
)
from plate_vision.utils.exceptions import ConfigurationError


class CustomMockCountryProfile:
    @property
    def country_code(self) -> str:
        return "DE"

    @property
    def charset(self) -> str:
        return "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    @property
    def regex_patterns(self) -> tuple[Pattern[str], ...]:
        return (re.compile(r"^[A-Z]{1,3}[A-Z]{1,2}[0-9]{1,4}$"),)

    @property
    def digit_to_alpha_map(self) -> Mapping[str, str]:
        return {"0": "O"}

    @property
    def alpha_to_digit_map(self) -> Mapping[str, str]:
        return {"O": "0"}

    def normalize(self, raw_text: str) -> str:
        return raw_text.strip().upper()

    def is_valid(self, plate_text: str) -> bool:
        return self.regex_patterns[0].fullmatch(self.normalize(plate_text)) is not None

    def correct_confusions(self, raw_text: str) -> str:
        return self.normalize(raw_text)

    def sample_plate(self, rng: random.Random) -> str:
        return f"BMA{rng.randint(100, 999)}"


def test_turkey_country_profile_validation_and_confusion_correction() -> None:
    profile = TurkeyCountryProfile()
    assert profile.country_code == "TR"
    assert "Q" not in profile.charset
    assert "W" not in profile.charset
    assert "X" not in profile.charset

    assert profile.is_valid("34 ABC 123")
    assert profile.is_valid("06-A-1234")
    assert profile.is_valid("81 ZV 9876")
    assert not profile.is_valid("00 ABC 123")
    assert not profile.is_valid("82 ABC 123")
    assert not profile.is_valid("34 QWE 123")

    assert profile.correct_confusions("34 A8C 123") == "34ABC123"
    assert profile.correct_confusions("O6 A8 12S4") == "06AB1254"
    assert profile.correct_confusions("I6 0Z 1I8") == "16OZ118"
    assert profile.correct_confusions("34ABC123") == "34ABC123"
    assert profile.correct_confusions("TOO_LONG_INVALID_99999") == "TOOLONGINVALID99999"
    assert profile.correct_confusions("99QQQ99") == "99QQQ99"


def test_country_profile_registry_extensibility() -> None:
    registry = CountryProfileRegistry()
    tr = TurkeyCountryProfile()
    de = CustomMockCountryProfile()
    registry.register(tr)
    registry.register(de)
    assert registry.available_codes() == ("DE", "TR")
    assert registry.get("de").country_code == "DE"

    with pytest.raises(ConfigurationError, match="Unregistered country profile"):
        registry.get("UNKNOWN")

    class EmptyCodeProfile(CustomMockCountryProfile):
        @property
        def country_code(self) -> str:
            return "   "

    with pytest.raises(ConfigurationError, match="Country profile code must not be empty"):
        registry.register(EmptyCodeProfile())

    register_country_profile(de)
    assert get_country_profile("DE").country_code == "DE"
    assert get_country_profile("TR").country_code == "TR"


def test_synthetic_plate_rendering_and_png_encoding(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive width and height"):
        encode_grayscale_png([])

    config = SyntheticPlateConfig(seed=7)
    rng = random.Random(7)
    pixels, png_bytes, bbox = generate_synthetic_sample("34ABC123", config, rng)
    assert len(pixels) == config.image_height
    assert len(pixels[0]) == config.image_width
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert bbox.width == float(config.image_width)

    with pytest.raises(ValueError, match="count must be a positive integer"):
        generate_synthetic_dataset(0, tmp_path, config, TurkeyCountryProfile())


def test_augmentation_pipeline_spec_and_pixel_transforms() -> None:
    disabled_pipeline = AugmentationPipeline(AugmentationConfig(enabled=False))
    assert disabled_pipeline.to_albumentations_spec() == []
    sample = [[100, 120], [140, 160]]
    assert disabled_pipeline.apply(sample, random.Random(1)) == sample

    active_pipeline = AugmentationPipeline(
        AugmentationConfig(
            enabled=True,
            probability=1.0,
            brightness_limit=0.1,
            contrast_limit=0.1,
            motion_blur_kernel=3,
            noise_std_limit=3.0,
            max_rotation_degrees=4.0,
            occlusion_max_fraction=0.2,
        )
    )
    spec = active_pipeline.to_albumentations_spec()
    assert len(spec) == 5
    grid = [[128 + ((x + y) % 40) for x in range(32)] for y in range(16)]
    transformed = active_pipeline.apply(grid, random.Random(42))
    assert len(transformed) == 16
    assert len(transformed[0]) == 32
    assert transformed != grid
