from pathlib import Path

from plate_vision.data.models import BoundingBox
from plate_vision.detection.models import DetectionPrediction
from plate_vision.evaluation.analyzer import build_evaluation_report, save_evaluation_artifacts
from plate_vision.utils.tracking import MlflowExperimentTracker


def test_evaluation_pipeline_logs_artifacts_to_mlflow_tracker(tmp_path: Path) -> None:
    box1 = BoundingBox(x_min=5.0, y_min=5.0, x_max=80.0, y_max=30.0)
    box2 = BoundingBox(x_min=12.0, y_min=15.0, x_max=95.0, y_max=45.0)
    det_preds = {
        "img_1": [DetectionPrediction(bbox=box1, confidence=0.92, class_id=0)],
        "img_2": [DetectionPrediction(bbox=box2, confidence=0.88, class_id=0)],
    }
    det_gts = {
        "img_1": [box1],
        "img_2": [box2],
    }

    report = build_evaluation_report(
        detector_predictions=det_preds,
        detector_ground_truths=det_gts,
        sample_ids=["img_1", "img_2"],
        recognizer_predictions=["34ABC123", "06AB812"],
        recognizer_targets=["34ABC123", "06ABB12"],
        end_to_end_predictions=["34ABC123", "06AB812"],
        end_to_end_targets=["34ABC123", "06ABB12"],
        latencies_ms=[12.0, 18.0],
        confidences=[0.95, 0.62],
    )

    artifact_dir = tmp_path / "eval_artifacts"
    saved = save_evaluation_artifacts(report, artifact_dir)

    tracker = MlflowExperimentTracker(
        tracking_dir=tmp_path / "mlruns",
        experiment_name="plate-vision-eval",
    )
    run_id = tracker.start_run("eval-suite")
    tracker.log_metrics(
        {
            "det_map50": report.detector.map50,
            "rec_exact_match": report.recognizer.exact_match_accuracy,
            "e2e_accuracy": report.end_to_end.plate_level_accuracy,
            "latency_p50_ms": report.end_to_end.latency_p50_ms,
            "latency_p95_ms": report.end_to_end.latency_p95_ms,
            "throughput_fps": report.end_to_end.throughput_fps,
        }
    )
    for path in saved.values():
        tracker.log_artifact(path)
    tracker.end_run()

    run_dir = tmp_path / "mlruns" / "plate-vision-eval" / run_id
    assert (run_dir / "artifacts" / "evaluation_report.json").is_file()
    assert (run_dir / "artifacts" / "confusion_pairs.json").is_file()
    assert (run_dir / "artifacts" / "worst_samples.json").is_file()
    assert (run_dir / "metrics" / "e2e_accuracy").is_file()
