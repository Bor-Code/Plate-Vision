import pytest

from plate_vision.data.models import BoundingBox
from plate_vision.pipeline.runner import PlatePredictionResult
from plate_vision.video.tracker import PlateTracker
from plate_vision.video.voting import TemporalVoter


def _make_pred(
    x_min: float,
    text: str,
    conf: float = 0.85,
    is_valid: bool = True,
) -> PlatePredictionResult:
    return PlatePredictionResult(
        bbox=BoundingBox(x_min=x_min, y_min=10.0, x_max=x_min + 80.0, y_max=40.0),
        detection_confidence=conf,
        recognition_confidence=conf,
        combined_confidence=conf,
        raw_text=text,
        plate_text=text,
        country_code="TR",
        is_valid_format=is_valid,
        was_corrected=False,
    )


def test_plate_tracker_validates_init_and_tracks_across_missed_frames() -> None:
    with pytest.raises(ValueError, match="iou_threshold"):
        PlateTracker(iou_threshold=0.0)
    with pytest.raises(ValueError, match="max_missed_frames"):
        PlateTracker(max_missed_frames=-1)
    with pytest.raises(ValueError, match="smoothing_alpha"):
        PlateTracker(smoothing_alpha=0.0)

    tracker = PlateTracker(iou_threshold=0.30, max_missed_frames=1, smoothing_alpha=0.50)
    f0 = tracker.update(0, [_make_pred(10.0, "34ABC123")])
    assert len(f0) == 1
    assert f0[0].track_id == 1

    f1 = tracker.update(1, [_make_pred(14.0, "34ABC123")])
    assert len(f1) == 1
    assert f1[0].track_id == 1
    assert f1[0].track_hits == 2

    f2_missed = tracker.update(2, [])
    assert len(f2_missed) == 0
    assert len(tracker.active_tracks()) == 1

    f3_expired = tracker.update(3, [])
    assert len(f3_expired) == 0
    assert len(tracker.active_tracks()) == 0


def test_temporal_voter_consensus_weights_valid_formats_and_window() -> None:
    with pytest.raises(ValueError, match="window_size"):
        TemporalVoter(window_size=0)
    with pytest.raises(ValueError, match="valid_format_bonus"):
        TemporalVoter(valid_format_bonus=0.5)

    voter = TemporalVoter(window_size=5, valid_format_bonus=1.5)
    with pytest.raises(KeyError):
        voter.get_consensus(99)

    voter.observe(1, _make_pred(10.0, "34AB", conf=0.70, is_valid=False))
    voter.observe(1, _make_pred(12.0, "34ABC123", conf=0.80, is_valid=True))
    res = voter.observe(1, _make_pred(14.0, "34ABC123", conf=0.90, is_valid=True))

    assert res.consensus_text == "34ABC123"
    assert res.is_valid_format is True
    assert res.vote_count == 3
    assert res.agreement_ratio == pytest.approx(2.0 / 3.0)
