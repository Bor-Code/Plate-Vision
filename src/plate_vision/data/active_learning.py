import json
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from plate_vision.data.models import BoundingBox
from plate_vision.utils.exceptions import DataIOError


class SelectionReason(StrEnum):
    LOW_RECOGNITION_CONFIDENCE = "low_recognition_confidence"
    LOW_DETECTION_CONFIDENCE = "low_detection_confidence"
    INVALID_PLATE_FORMAT = "invalid_plate_format"
    CONFUSION_CORRECTED = "confusion_corrected"


class ActiveLearningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    recognition_confidence_threshold: float = Field(default=0.80, gt=0.0, lt=1.0)
    detection_confidence_threshold: float = Field(default=0.75, gt=0.0, lt=1.0)
    invalid_format_penalty: float = Field(default=0.35, ge=0.0, le=1.0)
    confusion_correction_penalty: float = Field(default=0.15, ge=0.0, le=1.0)
    max_candidates: int = Field(default=100, ge=1)
    deduplicate_by_plate_text: bool = Field(default=True)


class HardExampleCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str = Field(min_length=1)
    bbox: BoundingBox
    raw_text: str
    plate_text: str
    country_code: str = Field(min_length=2)
    detection_confidence: float = Field(ge=0.0, le=1.0)
    recognition_confidence: float = Field(ge=0.0, le=1.0)
    is_valid_format: bool
    was_corrected: bool
    priority_score: float = Field(ge=0.0, le=2.0)
    reasons: tuple[SelectionReason, ...] = Field(min_length=1)


class ActiveLearningSampler:
    def __init__(self, config: ActiveLearningConfig | None = None) -> None:
        self.config = config if config is not None else ActiveLearningConfig()

    def evaluate_prediction(
        self,
        *,
        sample_id: str,
        bbox: BoundingBox,
        raw_text: str,
        plate_text: str,
        country_code: str,
        detection_confidence: float,
        recognition_confidence: float,
        is_valid_format: bool,
        was_corrected: bool,
    ) -> HardExampleCandidate | None:
        reasons: list[SelectionReason] = []
        uncertainty = 1.0 - min(1.0, max(0.0, recognition_confidence * detection_confidence))
        score = uncertainty

        if recognition_confidence < self.config.recognition_confidence_threshold:
            reasons.append(SelectionReason.LOW_RECOGNITION_CONFIDENCE)
        if detection_confidence < self.config.detection_confidence_threshold:
            reasons.append(SelectionReason.LOW_DETECTION_CONFIDENCE)
        if not is_valid_format:
            reasons.append(SelectionReason.INVALID_PLATE_FORMAT)
            score += self.config.invalid_format_penalty
        if was_corrected:
            reasons.append(SelectionReason.CONFUSION_CORRECTED)
            score += self.config.confusion_correction_penalty

        if not reasons:
            return None

        return HardExampleCandidate(
            sample_id=sample_id,
            bbox=bbox,
            raw_text=raw_text,
            plate_text=plate_text,
            country_code=country_code,
            detection_confidence=round(detection_confidence, 6),
            recognition_confidence=round(recognition_confidence, 6),
            is_valid_format=is_valid_format,
            was_corrected=was_corrected,
            priority_score=round(min(2.0, max(0.0, score)), 6),
            reasons=tuple(reasons),
        )

    def select_top_candidates(
        self,
        candidates: Sequence[HardExampleCandidate],
    ) -> tuple[HardExampleCandidate, ...]:
        ordered = sorted(
            candidates,
            key=lambda item: (-item.priority_score, item.sample_id),
        )
        selected: list[HardExampleCandidate] = []
        seen_texts: set[str] = set()

        for candidate in ordered:
            norm_key = candidate.plate_text.strip().upper()
            if self.config.deduplicate_by_plate_text and norm_key and norm_key in seen_texts:
                continue
            if norm_key:
                seen_texts.add(norm_key)
            selected.append(candidate)
            if len(selected) >= self.config.max_candidates:
                break

        return tuple(selected)

    @staticmethod
    def export_manifest(
        candidates: Sequence[HardExampleCandidate],
        output_path: Path,
    ) -> Path:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                json.dumps(item.model_dump(mode="json"), sort_keys=True) for item in candidates
            ]
            content = "\n".join(lines) + ("\n" if lines else "")
            output_path.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise DataIOError(
                f"Failed to write active learning manifest to {output_path}",
                details={"path": str(output_path), "reason": str(exc)},
            ) from exc
        return output_path


__all__ = [
    "ActiveLearningConfig",
    "ActiveLearningSampler",
    "HardExampleCandidate",
    "SelectionReason",
]
