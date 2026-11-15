from plate_vision.data.active_learning import (
    ActiveLearningConfig,
    ActiveLearningSampler,
    SelectionReason,
)
from plate_vision.data.models import BoundingBox


def test_active_learning_sampler_skips_confident_valid_predictions() -> None:
    sampler = ActiveLearningSampler()
    bbox = BoundingBox(x_min=10.0, y_min=10.0, x_max=100.0, y_max=40.0)
    candidate = sampler.evaluate_prediction(
        sample_id="s-clean",
        bbox=bbox,
        raw_text="34ABC123",
        plate_text="34ABC123",
        country_code="TR",
        detection_confidence=0.95,
        recognition_confidence=0.92,
        is_valid_format=True,
        was_corrected=False,
    )
    assert candidate is None


def test_active_learning_sampler_scores_and_deduplicates() -> None:
    cfg = ActiveLearningConfig(
        recognition_confidence_threshold=0.80,
        detection_confidence_threshold=0.75,
        max_candidates=2,
        deduplicate_by_plate_text=True,
    )
    sampler = ActiveLearningSampler(cfg)
    bbox = BoundingBox(x_min=10.0, y_min=10.0, x_max=100.0, y_max=40.0)

    c1 = sampler.evaluate_prediction(
        sample_id="s-1",
        bbox=bbox,
        raw_text="34A8C123",
        plate_text="34ABC123",
        country_code="TR",
        detection_confidence=0.90,
        recognition_confidence=0.88,
        is_valid_format=True,
        was_corrected=True,
    )
    assert c1 is not None
    assert SelectionReason.CONFUSION_CORRECTED in c1.reasons

    c2 = sampler.evaluate_prediction(
        sample_id="s-2",
        bbox=bbox,
        raw_text="99ZZ999",
        plate_text="99ZZ999",
        country_code="TR",
        detection_confidence=0.60,
        recognition_confidence=0.55,
        is_valid_format=False,
        was_corrected=False,
    )
    assert c2 is not None
    assert SelectionReason.LOW_RECOGNITION_CONFIDENCE in c2.reasons
    assert SelectionReason.LOW_DETECTION_CONFIDENCE in c2.reasons
    assert SelectionReason.INVALID_PLATE_FORMAT in c2.reasons

    c3_dup = sampler.evaluate_prediction(
        sample_id="s-3",
        bbox=bbox,
        raw_text="34ABC123",
        plate_text="34ABC123",
        country_code="TR",
        detection_confidence=0.70,
        recognition_confidence=0.70,
        is_valid_format=True,
        was_corrected=False,
    )
    assert c3_dup is not None

    selected = sampler.select_top_candidates([c1, c2, c3_dup])
    assert len(selected) == 2
    assert selected[0].sample_id == "s-2"
    assert selected[1].plate_text == "34ABC123"
