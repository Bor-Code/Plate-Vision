from collections.abc import Mapping, Sequence

from plate_vision.data.models import BoundingBox
from plate_vision.detection.models import DetectionPrediction, DetectorMetrics


def compute_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    inter_x1 = max(box_a.x_min, box_b.x_min)
    inter_y1 = max(box_a.y_min, box_b.y_min)
    inter_x2 = min(box_a.x_max, box_b.x_max)
    inter_y2 = min(box_a.y_max, box_b.y_max)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    union_area = (box_a.width * box_a.height) + (box_b.width * box_b.height) - inter_area
    if union_area <= 0.0:
        return 0.0
    return min(1.0, max(0.0, inter_area / union_area))


def apply_nms(
    predictions: Sequence[DetectionPrediction],
    *,
    confidence_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> list[DetectionPrediction]:
    filtered = [pred for pred in predictions if pred.confidence >= confidence_threshold]
    ordered = sorted(filtered, key=lambda item: item.confidence, reverse=True)
    kept: list[DetectionPrediction] = []

    for candidate in ordered:
        if len(kept) >= max_detections:
            break
        should_suppress = any(
            existing.class_id == candidate.class_id
            and compute_iou(existing.bbox, candidate.bbox) > iou_threshold
            for existing in kept
        )
        if not should_suppress:
            kept.append(candidate)

    return kept


def compute_ap_at_iou(
    predictions_by_image: Mapping[str, Sequence[DetectionPrediction]],
    ground_truths_by_image: Mapping[str, Sequence[BoundingBox]],
    iou_threshold: float,
) -> tuple[float, float, float]:
    total_gt = sum(len(boxes) for boxes in ground_truths_by_image.values())
    scored_predictions: list[tuple[float, str, BoundingBox]] = []
    for image_id, preds in predictions_by_image.items():
        for pred in preds:
            scored_predictions.append((pred.confidence, image_id, pred.bbox))

    scored_predictions.sort(key=lambda item: item[0], reverse=True)
    if total_gt == 0:
        if not scored_predictions:
            return (1.0, 1.0, 1.0)
        return (0.0, 0.0, 0.0)

    if not scored_predictions:
        return (0.0, 0.0, 0.0)

    matched_gt: dict[str, set[int]] = {img_id: set() for img_id in ground_truths_by_image}
    tp_cumsum: list[int] = []
    fp_cumsum: list[int] = []
    tp_running = 0
    fp_running = 0

    for _, image_id, pred_box in scored_predictions:
        gt_boxes = ground_truths_by_image.get(image_id, ())
        best_iou = 0.0
        best_idx = -1
        used = matched_gt.setdefault(image_id, set())
        for idx, gt_box in enumerate(gt_boxes):
            if idx in used:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_iou >= iou_threshold and best_idx >= 0:
            used.add(best_idx)
            tp_running += 1
        else:
            fp_running += 1

        tp_cumsum.append(tp_running)
        fp_cumsum.append(fp_running)

    precisions = [tp / float(tp + fp) for tp, fp in zip(tp_cumsum, fp_cumsum, strict=True)]
    recalls = [tp / float(total_gt) for tp in tp_cumsum]

    interpolated_ap = 0.0
    for step in range(101):
        recall_level = step / 100.0
        eligible = [p for p, r in zip(precisions, recalls, strict=True) if r >= recall_level]
        interpolated_ap += max(eligible) if eligible else 0.0
    ap = min(1.0, max(0.0, interpolated_ap / 101.0))

    final_precision = precisions[-1]
    final_recall = recalls[-1]
    return (final_precision, final_recall, ap)


def evaluate_detector_predictions(
    predictions_by_image: Mapping[str, Sequence[DetectionPrediction]],
    ground_truths_by_image: Mapping[str, Sequence[BoundingBox]],
) -> DetectorMetrics:
    precision_50, recall_50, map50 = compute_ap_at_iou(
        predictions_by_image=predictions_by_image,
        ground_truths_by_image=ground_truths_by_image,
        iou_threshold=0.50,
    )
    iou_thresholds = [0.50 + 0.05 * idx for idx in range(10)]
    aps = [
        compute_ap_at_iou(
            predictions_by_image=predictions_by_image,
            ground_truths_by_image=ground_truths_by_image,
            iou_threshold=thresh,
        )[2]
        for thresh in iou_thresholds
    ]
    map50_95 = sum(aps) / len(aps)
    return DetectorMetrics(
        precision=round(precision_50, 6),
        recall=round(recall_50, 6),
        map50=round(map50, 6),
        map50_95=round(map50_95, 6),
    )


__all__ = [
    "apply_nms",
    "compute_ap_at_iou",
    "compute_iou",
    "evaluate_detector_predictions",
]
