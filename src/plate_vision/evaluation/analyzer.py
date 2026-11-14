import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from plate_vision.data.models import BoundingBox
from plate_vision.detection.metrics import evaluate_detector_predictions
from plate_vision.detection.models import DetectionPrediction
from plate_vision.evaluation.models import (
    ConfusionPair,
    EndToEndMetrics,
    EvaluationReport,
    WorstSampleEntry,
)
from plate_vision.recognition.trainer import (
    compute_levenshtein_distance,
    compute_recognizer_metrics,
)
from plate_vision.utils.exceptions import DataIOError


def compute_percentile(values: Sequence[float], percentile: float) -> float:
    if not 0.0 <= percentile <= 100.0:
        msg = "percentile must be between 0.0 and 100.0"
        raise ValueError(msg)
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower_idx = math.floor(rank)
    upper_idx = math.ceil(rank)
    if lower_idx == upper_idx:
        return ordered[lower_idx]
    weight = rank - lower_idx
    return ordered[lower_idx] * (1.0 - weight) + ordered[upper_idx] * weight


def extract_confusion_pairs(
    predictions: Sequence[str],
    targets: Sequence[str],
    *,
    top_k: int = 20,
) -> tuple[ConfusionPair, ...]:
    if top_k < 1:
        msg = "top_k must be >= 1"
        raise ValueError(msg)
    pair_counts: Counter[tuple[str, str]] = Counter()
    for pred, target in zip(predictions, targets, strict=True):
        norm_p = pred.strip().upper()
        norm_t = target.strip().upper()
        for ch_p, ch_t in zip(norm_p, norm_t, strict=False):
            if ch_p != ch_t:
                pair_counts[(ch_t, ch_p)] += 1

    ranked = sorted(
        pair_counts.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    )[:top_k]
    return tuple(
        ConfusionPair(target_char=t_ch, predicted_char=p_ch, count=count)
        for (t_ch, p_ch), count in ranked
    )


def extract_worst_samples(
    sample_ids: Sequence[str],
    predictions: Sequence[str],
    targets: Sequence[str],
    confidences: Sequence[float] | None = None,
    *,
    top_k: int = 20,
) -> tuple[WorstSampleEntry, ...]:
    if top_k < 1:
        msg = "top_k must be >= 1"
        raise ValueError(msg)
    if confidences is None:
        resolved_confidences: Sequence[float] = [1.0] * len(sample_ids)
    else:
        resolved_confidences = confidences

    entries: list[WorstSampleEntry] = []
    for sample_id, pred, target, conf in zip(
        sample_ids, predictions, targets, resolved_confidences, strict=True
    ):
        norm_p = pred.strip().upper()
        norm_t = target.strip().upper()
        edits = compute_levenshtein_distance(norm_p, norm_t)
        cer = edits / float(max(1, len(norm_t)))
        entries.append(
            WorstSampleEntry(
                sample_id=sample_id,
                target_text=norm_t,
                predicted_text=norm_p,
                edit_distance=edits,
                character_error_rate=round(cer, 6),
                confidence=round(min(1.0, max(0.0, conf)), 6),
            )
        )

    entries.sort(
        key=lambda item: (
            -item.character_error_rate,
            -item.edit_distance,
            item.confidence,
            item.sample_id,
        )
    )
    return tuple(entries[:top_k])


def compute_end_to_end_metrics(
    end_to_end_predictions: Sequence[str],
    end_to_end_targets: Sequence[str],
    latencies_ms: Sequence[float],
) -> EndToEndMetrics:
    if not end_to_end_targets:
        return EndToEndMetrics(
            plate_level_accuracy=0.0,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            throughput_fps=0.0,
        )
    matches = sum(
        1
        for pred, target in zip(end_to_end_predictions, end_to_end_targets, strict=True)
        if pred.strip().upper() == target.strip().upper()
    )
    accuracy = matches / float(len(end_to_end_targets))
    p50 = compute_percentile(latencies_ms, 50.0)
    p95 = compute_percentile(latencies_ms, 95.0)
    total_ms = sum(max(0.0, val) for val in latencies_ms)
    throughput = (len(latencies_ms) * 1000.0 / total_ms) if total_ms > 0.0 else 0.0

    return EndToEndMetrics(
        plate_level_accuracy=round(accuracy, 6),
        latency_p50_ms=round(p50, 4),
        latency_p95_ms=round(p95, 4),
        throughput_fps=round(throughput, 4),
    )


def build_evaluation_report(
    *,
    detector_predictions: Mapping[str, Sequence[DetectionPrediction]],
    detector_ground_truths: Mapping[str, Sequence[BoundingBox]],
    sample_ids: Sequence[str],
    recognizer_predictions: Sequence[str],
    recognizer_targets: Sequence[str],
    end_to_end_predictions: Sequence[str],
    end_to_end_targets: Sequence[str],
    latencies_ms: Sequence[float],
    confidences: Sequence[float] | None = None,
    top_k_errors: int = 20,
) -> EvaluationReport:
    detector_metrics = evaluate_detector_predictions(
        predictions_by_image=detector_predictions,
        ground_truths_by_image=detector_ground_truths,
    )
    recognizer_metrics = compute_recognizer_metrics(
        predictions=recognizer_predictions,
        targets=recognizer_targets,
    )
    e2e_metrics = compute_end_to_end_metrics(
        end_to_end_predictions=end_to_end_predictions,
        end_to_end_targets=end_to_end_targets,
        latencies_ms=latencies_ms,
    )
    confusions = extract_confusion_pairs(
        predictions=recognizer_predictions,
        targets=recognizer_targets,
        top_k=top_k_errors,
    )
    worst = extract_worst_samples(
        sample_ids=sample_ids,
        predictions=recognizer_predictions,
        targets=recognizer_targets,
        confidences=confidences,
        top_k=top_k_errors,
    )
    return EvaluationReport(
        detector=detector_metrics,
        recognizer=recognizer_metrics,
        end_to_end=e2e_metrics,
        confusion_pairs=confusions,
        worst_samples=worst,
    )


def save_evaluation_artifacts(
    report: EvaluationReport,
    output_dir: Path,
) -> dict[str, Path]:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "evaluation_report.json"
        confusions_path = output_dir / "confusion_pairs.json"
        worst_path = output_dir / "worst_samples.json"

        report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        confusions_payload = [item.model_dump() for item in report.confusion_pairs]
        confusions_path.write_text(
            json.dumps(confusions_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        worst_payload = [item.model_dump() for item in report.worst_samples]
        worst_path.write_text(
            json.dumps(worst_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return {
            "report": report_path,
            "confusion_pairs": confusions_path,
            "worst_samples": worst_path,
        }
    except OSError as exc:
        raise DataIOError(
            f"Failed to write evaluation artifacts to {output_dir}",
            details={"output_dir": str(output_dir), "reason": str(exc)},
        ) from exc


__all__ = [
    "build_evaluation_report",
    "compute_end_to_end_metrics",
    "compute_percentile",
    "extract_confusion_pairs",
    "extract_worst_samples",
    "save_evaluation_artifacts",
]
