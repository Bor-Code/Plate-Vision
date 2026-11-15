import pytest

from plate_vision.data.models import BoundingBox
from plate_vision.pipeline.runner import PlatePredictionResult
from plate_vision.video.analytics import CalibrationZone, VideoTrafficAnalyzer
from plate_vision.video.processor import VideoFrameResult, VideoStreamSummary
from plate_vision.video.tracker import TrackedPlateDetection
from plate_vision.video.voting import StabilizedTrackPrediction


def _make_prediction(
    bbox: BoundingBox, text: str, country_code: str = "TR"
) -> PlatePredictionResult:
    return PlatePredictionResult(
        bbox=bbox,
        detection_confidence=0.95,
        recognition_confidence=0.92,
        combined_confidence=0.874,
        raw_text=text,
        plate_text=text,
        country_code=country_code,
        is_valid_format=True,
        was_corrected=False,
    )


def test_video_traffic_analyzer_empty_and_speed_calculation() -> None:
    analyzer = VideoTrafficAnalyzer(CalibrationZone(pixels_per_meter=10.0, speed_limit_kmh=50.0))
    empty_summary = VideoStreamSummary(
        frames_processed=0,
        unique_tracks=0,
        final_tracks=(),
        frame_results=(),
    )
    empty_report = analyzer.analyze(empty_summary)
    assert empty_report.total_vehicles == 0
    assert empty_report.mean_dwell_time_seconds == 0.0
    assert empty_report.mean_speed_kmh is None
    assert empty_report.speeding_count == 0

    b1 = BoundingBox(x_min=0.0, y_min=0.0, x_max=40.0, y_max=20.0)
    b2 = BoundingBox(x_min=200.0, y_min=0.0, x_max=240.0, y_max=20.0)
    b_slow = BoundingBox(x_min=10.0, y_min=40.0, x_max=50.0, y_max=60.0)

    p1_t0 = _make_prediction(b1, "34ABC123", "TR")
    p1_t1 = _make_prediction(b2, "34ABC123", "TR")
    p2_t0 = _make_prediction(b_slow, "BAB1234", "DE")

    summary = VideoStreamSummary(
        frames_processed=2,
        unique_tracks=2,
        final_tracks=(
            StabilizedTrackPrediction(
                track_id=1,
                consensus_text="34ABC123",
                consensus_confidence=0.9,
                is_valid_format=True,
                vote_count=2,
                agreement_ratio=1.0,
            ),
            StabilizedTrackPrediction(
                track_id=2,
                consensus_text="BAB1234",
                consensus_confidence=0.88,
                is_valid_format=True,
                vote_count=1,
                agreement_ratio=1.0,
            ),
        ),
        frame_results=(
            VideoFrameResult(
                frame_index=0,
                timestamp_ms=0.0,
                tracked_detections=(
                    TrackedPlateDetection(
                        track_id=1,
                        frame_index=0,
                        prediction=p1_t0,
                        track_hits=1,
                    ),
                    TrackedPlateDetection(
                        track_id=2,
                        frame_index=0,
                        prediction=p2_t0,
                        track_hits=1,
                    ),
                ),
                stabilized_predictions=(),
                frame_latency_ms=1.0,
            ),
            VideoFrameResult(
                frame_index=1,
                timestamp_ms=1000.0,
                tracked_detections=(
                    TrackedPlateDetection(
                        track_id=1,
                        frame_index=1,
                        prediction=p1_t1,
                        track_hits=2,
                    ),
                ),
                stabilized_predictions=(),
                frame_latency_ms=1.0,
            ),
        ),
    )

    report = analyzer.analyze(summary)
    assert report.total_vehicles == 2
    assert report.speeding_count == 1
    assert report.country_distribution == {"DE": 1, "TR": 1}
    assert report.mean_dwell_time_seconds == pytest.approx(0.5)

    t1 = report.track_summaries[0]
    assert t1.dwell_time_seconds == pytest.approx(1.0)
    assert t1.displacement_pixels == pytest.approx(200.0)
    assert t1.estimated_speed_kmh == pytest.approx(72.0)
    assert t1.is_speeding is True

    t2 = report.track_summaries[1]
    assert t2.dwell_time_seconds == pytest.approx(0.0)
    assert t2.estimated_speed_kmh is None
    assert t2.is_speeding is False
