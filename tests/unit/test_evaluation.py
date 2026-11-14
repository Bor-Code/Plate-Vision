from pathlib import Path

import pytest

from plate_vision.data.models import BoundingBox
from plate_vision.detection.models import DetectionPrediction
from plate_vision.evaluation.analyzer import (
    build_evaluation_report,
    compute_end_to_end_metrics,
    compute_percentile,
    extract_confusion_pairs,
    extract_worst_samples,
    save_evaluation_artifacts,
)


def test_compute_percentile_handles_empty_single_and_interpolated() -> None:
    assert compute_percentile([], 50.0) == 0.0
    assert compute_percentile([12.5], 95.0) == 12.5
    p50 = compute_percentile([10.0, 20.0, 30.0, 40.0], 50.0)
    assert p50 == pytest.approx(25.0)
    p95 = compute_percentile([10.0, 20.0, 30.0, 40.0], 95.0)
    assert p95 == pytest.approx(38.5)


def test_compute_percentile_rejects_out_of_bounds() -> None:
    with pytest.raises(ValueError, match="percentile"):
        compute_percentile([1.0, 2.0], -1.0)
    with pytest.raises(ValueError, match="percentile"):
        compute_percentile([1.0, 2.0], 101.0)


def test_extract_confusion_pairs_ranks_most_frequent_substitutions() -> None:
    preds = ["34AB0123", "06XY099", "35AB012"]
    targets = ["34ABO123", "06XYO99", "35AB812"]
    pairs = extract_confusion_pairs(preds, targets, top_k=5)
    assert len(pairs) == 2
    assert pairs[0].target_char == "O"
    assert pairs[0].predicted_char == "0"
    assert pairs[0].count == 2
    assert pairs[1].target_char == "8"
    assert pairs[1].predicted_char == "0"
    assert pairs[1].count == 1


def test_extract_confusion_pairs_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        extract_confusion_pairs(["34ABC12"], ["34ABC12"], top_k=0)


def test_extract_worst_samples_orders_by_cer_and_edit_distance() -> None:
    sample_ids = ["s1", "s2", "s3"]
    preds = ["34ABC123", "34XYZ999", "34ABD123"]
    targets = ["34ABC123", "34ABC123", "34ABC123"]
    confidences = [0.98, 0.45, 0.75]

    worst = extract_worst_samples(sample_ids, preds, targets, confidences, top_k=2)
    assert len(worst) == 2
    assert worst[0].sample_id == "s2"
    assert worst[0].edit_distance == 6
    assert worst[1].sample_id == "s3"
    assert worst[1].edit_distance == 1


def test_extract_worst_samples_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        extract_worst_samples(["s1"], ["34ABC12"], ["34ABC12"], top_k=0)


def test_compute_end_to_end_metrics_and_empty_handling() -> None:
    empty_metrics = compute_end_to_end_metrics([], [], [])
    assert empty_metrics.plate_level_accuracy == 0.0
    assert empty_metrics.throughput_fps == 0.0

    e2e = compute_end_to_end_metrics(
        end_to_end_predictions=["34ABC123", "06DEF45"],
        end_to_end_targets=["34ABC123", "06DEF99"],
        latencies_ms=[20.0, 30.0],
    )
    assert e2e.plate_level_accuracy == pytest.approx(0.5)
    assert e2e.latency_p50_ms == pytest.approx(25.0)
    assert e2e.throughput_fps == pytest.approx(40.0)


def test_build_and_save_evaluation_report(tmp_path: Path) -> None:
    box = BoundingBox(x_min=10.0, y_min=10.0, x_max=100.0, y_max=40.0)
    pred = DetectionPrediction(bbox=box, confidence=0.95, class_id=0)
    report = build_evaluation_report(
        detector_predictions={"img1": [pred]},
        detector_ground_truths={"img1": [box]},
        sample_ids=["img1"],
        recognizer_predictions=["34AB0123"],
        recognizer_targets=["34ABO123"],
        end_to_end_predictions=["34AB0123"],
        end_to_end_targets=["34ABO123"],
        latencies_ms=[15.0],
    )
    assert report.detector.map50 == pytest.approx(1.0)
    assert len(report.confusion_pairs) == 1
    assert len(report.worst_samples) == 1

    artifacts = save_evaluation_artifacts(report, tmp_path / "eval_out")
    assert artifacts["report"].is_file()
    assert artifacts["confusion_pairs"].is_file()
    assert artifacts["worst_samples"].is_file()
