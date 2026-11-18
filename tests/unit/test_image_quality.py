import pytest

from plate_vision.pipeline.quality import (
    QualityIssue,
    QualityThresholds,
    assess_image_quality,
)
from plate_vision.utils.exceptions import InvalidInputError


def test_assess_image_quality_validation() -> None:
    with pytest.raises(InvalidInputError):
        assess_image_quality([])
    with pytest.raises(InvalidInputError):
        assess_image_quality([[10, 20], [30]])


def test_assess_image_quality_clean_and_degraded_cases() -> None:
    clean = [[60 if (r + c) % 2 == 0 else 190 for c in range(20)] for r in range(20)]
    clean_rep = assess_image_quality(clean)
    assert clean_rep.is_acceptable is True
    assert clean_rep.quality_score == 1.0
    assert clean_rep.issues == ()

    dark = [[10 for _ in range(16)] for _ in range(16)]
    dark_rep = assess_image_quality(dark)
    assert dark_rep.is_acceptable is False
    assert QualityIssue.LOW_LIGHT in dark_rep.issues
    assert QualityIssue.LOW_CONTRAST in dark_rep.issues
    assert QualityIssue.MOTION_BLUR in dark_rep.issues

    glare = [[252 for _ in range(16)] for _ in range(16)]
    glare_rep = assess_image_quality(glare, QualityThresholds(max_glare_pixel_ratio=0.10))
    assert glare_rep.is_acceptable is False
    assert QualityIssue.OVEREXPOSED_GLARE in glare_rep.issues
