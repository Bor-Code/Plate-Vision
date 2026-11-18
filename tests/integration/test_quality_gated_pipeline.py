import random

from plate_vision.data.profiles import TurkeyCountryProfile
from plate_vision.data.synthetic import SyntheticPlateConfig, generate_synthetic_sample
from plate_vision.pipeline.quality import QualityIssue, QualityThresholds, assess_image_quality
from plate_vision.pipeline.rectification import decode_image_to_grayscale


def test_quality_assessment_on_synthetic_and_glare_crops() -> None:
    rng = random.Random(42)
    profile = TurkeyCountryProfile()
    plate_text = profile.sample_plate(rng)
    _, png_bytes, _ = generate_synthetic_sample(
        plate_text,
        SyntheticPlateConfig(noise_std=2.0),
        rng,
    )
    grid = decode_image_to_grayscale(png_bytes)
    report = assess_image_quality(
        grid,
        thresholds=QualityThresholds(
            min_mean_brightness=20.0,
            max_mean_brightness=235.0,
            min_contrast_std=8.0,
            min_sharpness_gradient=2.0,
        ),
    )
    assert report.is_acceptable is True
    assert report.quality_score == 1.0

    overexposed_grid = [[250 if c < 10 else 252 for c in range(32)] for _ in range(16)]
    bad_report = assess_image_quality(overexposed_grid)
    assert bad_report.is_acceptable is False
    assert QualityIssue.OVEREXPOSED_GLARE in bad_report.issues
    assert bad_report.quality_score < 1.0
