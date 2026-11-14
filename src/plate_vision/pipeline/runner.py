import time
from collections.abc import Callable, Mapping

from pydantic import BaseModel, ConfigDict, Field

from plate_vision.data.models import BoundingBox
from plate_vision.data.profiles import CountryProfileProtocol, get_country_profile
from plate_vision.detection.metrics import apply_nms
from plate_vision.detection.models import PlateDetectorProtocol
from plate_vision.pipeline.postprocess import postprocess_plate_text
from plate_vision.pipeline.rectification import (
    Point2D,
    decode_image_to_grayscale,
    extract_plate_crop,
    normalize_plate_crop,
    rectify_perspective_crop,
)
from plate_vision.recognition.model import PlateRecognizerProtocol
from plate_vision.utils.exceptions import EmptyDetectionError, InferenceTimeoutError


class PlatePredictionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    bbox: BoundingBox
    detection_confidence: float = Field(ge=0.0, le=1.0)
    recognition_confidence: float = Field(ge=0.0, le=1.0)
    combined_confidence: float = Field(ge=0.0, le=1.0)
    raw_text: str
    plate_text: str
    country_code: str = Field(min_length=2)
    is_valid_format: bool
    was_corrected: bool


class PipelineInferenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plates: tuple[PlatePredictionResult, ...]
    image_width: int = Field(ge=1)
    image_height: int = Field(ge=1)
    latency_ms: float = Field(ge=0.0)


class PlateVisionPipeline:
    def __init__(
        self,
        detector: PlateDetectorProtocol,
        recognizer: PlateRecognizerProtocol,
        *,
        country_code: str = "TR",
        profile: CountryProfileProtocol | None = None,
        max_payload_bytes: int = 10_485_760,
        timeout_seconds: float = 5.0,
        min_plate_width: int = 12,
        min_plate_height: int = 6,
        target_crop_width: int = 128,
        target_crop_height: int = 32,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 10,
        require_detection: bool = False,
        use_beam_search: bool = True,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.detector = detector
        self.recognizer = recognizer
        self.profile = profile if profile is not None else get_country_profile(country_code)
        self.country_code = self.profile.country_code
        self.max_payload_bytes = max_payload_bytes
        self.timeout_seconds = timeout_seconds
        self.min_plate_width = min_plate_width
        self.min_plate_height = min_plate_height
        self.target_crop_width = target_crop_width
        self.target_crop_height = target_crop_height
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.require_detection = require_detection
        self.use_beam_search = use_beam_search
        self.clock = clock

    def _check_timeout(self, start_time: float, stage: str) -> None:
        elapsed = self.clock() - start_time
        if elapsed > self.timeout_seconds:
            msg = (
                f"Inference timed out during {stage} ({elapsed:.3f}s > {self.timeout_seconds:.3f}s)"
            )
            raise InferenceTimeoutError(
                msg,
                details={
                    "stage": stage,
                    "elapsed_seconds": round(elapsed, 4),
                    "timeout_seconds": self.timeout_seconds,
                },
            )

    def predict_bytes(
        self,
        image_bytes: bytes,
        *,
        corners_override: Mapping[int, tuple[Point2D, Point2D, Point2D, Point2D]] | None = None,
    ) -> PipelineInferenceResult:
        start_time = self.clock()
        image_pixels = decode_image_to_grayscale(
            image_bytes,
            max_payload_bytes=self.max_payload_bytes,
        )
        image_height = len(image_pixels)
        image_width = len(image_pixels[0]) if image_height > 0 else 0
        self._check_timeout(start_time, "decode")

        raw_detections = self.detector.detect(image_bytes, image_width, image_height)
        self._check_timeout(start_time, "detection")

        nms_detections = apply_nms(
            raw_detections,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            max_detections=self.max_detections,
        )

        valid_detections = [
            det
            for det in nms_detections
            if det.bbox.width >= self.min_plate_width and det.bbox.height >= self.min_plate_height
        ]

        if not valid_detections:
            if self.require_detection:
                raise EmptyDetectionError(
                    "No license plates detected in image",
                    details={"image_width": image_width, "image_height": image_height},
                )
            elapsed_ms = max(0.0, (self.clock() - start_time) * 1000.0)
            return PipelineInferenceResult(
                plates=(),
                image_width=image_width,
                image_height=image_height,
                latency_ms=round(elapsed_ms, 4),
            )

        results: list[PlatePredictionResult] = []
        for idx, det in enumerate(valid_detections):
            self._check_timeout(start_time, f"crop_{idx}")
            if corners_override is not None and idx in corners_override:
                crop = rectify_perspective_crop(
                    image_pixels,
                    corners_override[idx],
                    output_width=self.target_crop_width,
                    output_height=self.target_crop_height,
                )
            else:
                crop = extract_plate_crop(
                    image_pixels,
                    det.bbox,
                    min_width=self.min_plate_width,
                    min_height=self.min_plate_height,
                )
            normalized_crop = normalize_plate_crop(
                crop,
                target_width=self.target_crop_width,
                target_height=self.target_crop_height,
            )
            decoded = self.recognizer.recognize(
                normalized_crop,
                use_beam_search=self.use_beam_search,
            )
            self._check_timeout(start_time, f"recognition_{idx}")

            post = postprocess_plate_text(
                decoded.text,
                profile=self.profile,
            )
            combined_conf = min(1.0, max(0.0, det.confidence * decoded.confidence))
            results.append(
                PlatePredictionResult(
                    bbox=det.bbox,
                    detection_confidence=round(det.confidence, 6),
                    recognition_confidence=round(decoded.confidence, 6),
                    combined_confidence=round(combined_conf, 6),
                    raw_text=post.raw_text,
                    plate_text=post.corrected_text,
                    country_code=post.country_code,
                    is_valid_format=post.is_valid_format,
                    was_corrected=post.was_corrected,
                )
            )

        elapsed_ms = max(0.0, (self.clock() - start_time) * 1000.0)
        return PipelineInferenceResult(
            plates=tuple(results),
            image_width=image_width,
            image_height=image_height,
            latency_ms=round(elapsed_ms, 4),
        )


__all__ = [
    "PipelineInferenceResult",
    "PlatePredictionResult",
    "PlateVisionPipeline",
]
