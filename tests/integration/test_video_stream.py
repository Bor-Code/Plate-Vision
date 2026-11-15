from collections.abc import Sequence

from plate_vision.data.models import BoundingBox
from plate_vision.data.synthetic import encode_grayscale_png
from plate_vision.detection.models import DetectionPrediction
from plate_vision.pipeline.runner import PlateVisionPipeline
from plate_vision.recognition.decoding import DecodedPlate
from plate_vision.video.processor import VideoFrameInput, VideoStreamProcessor


class _StreamDetector:
    def __init__(self, boxes_per_frame: list[list[BoundingBox]]) -> None:
        self._boxes = boxes_per_frame
        self._idx = 0

    def detect(
        self,
        image_bytes: bytes,
        image_width: int,
        image_height: int,
    ) -> list[DetectionPrediction]:
        _ = (image_bytes, image_width, image_height)
        boxes = self._boxes[self._idx % len(self._boxes)]
        self._idx += 1
        return [DetectionPrediction(bbox=b, confidence=0.95, class_id=0) for b in boxes]


class _StreamRecognizer:
    def __init__(self, texts: list[tuple[str, float]]) -> None:
        self._texts = texts
        self._idx = 0

    def recognize(
        self,
        crop_pixels: Sequence[Sequence[int]],
        *,
        use_beam_search: bool = False,
    ) -> DecodedPlate:
        _ = (crop_pixels, use_beam_search)
        text, conf = self._texts[self._idx % len(self._texts)]
        self._idx += 1
        return DecodedPlate(text=text, confidence=conf, token_indices=(1, 2))


def test_video_stream_processor_tracks_multiple_vehicles_and_votes() -> None:
    car1_f0 = BoundingBox(x_min=10.0, y_min=10.0, x_max=90.0, y_max=40.0)
    car2_f0 = BoundingBox(x_min=100.0, y_min=45.0, x_max=180.0, y_max=75.0)

    car1_f1 = BoundingBox(x_min=12.0, y_min=10.0, x_max=92.0, y_max=40.0)
    car2_f1 = BoundingBox(x_min=102.0, y_min=45.0, x_max=182.0, y_max=75.0)

    car1_f2 = BoundingBox(x_min=14.0, y_min=10.0, x_max=94.0, y_max=40.0)
    car2_f2 = BoundingBox(x_min=104.0, y_min=45.0, x_max=184.0, y_max=75.0)

    detector = _StreamDetector(
        [
            [car1_f0, car2_f0],
            [car1_f1, car2_f1],
            [car1_f2, car2_f2],
        ]
    )
    recognizer = _StreamRecognizer(
        [
            ("34ABC12", 0.55),
            ("06DEF99", 0.90),
            ("34ABC123", 0.92),
            ("06DEF99", 0.88),
            ("34ABC123", 0.95),
            ("06DEF98", 0.50),
        ]
    )
    pipeline = PlateVisionPipeline(detector=detector, recognizer=recognizer)
    processor = VideoStreamProcessor(pipeline=pipeline)

    frame_bytes = encode_grayscale_png([[125 for _ in range(200)] for _ in range(90)])
    frames = [
        VideoFrameInput(frame_index=0, timestamp_ms=0.0, image_bytes=frame_bytes),
        VideoFrameInput(frame_index=1, timestamp_ms=33.3, image_bytes=frame_bytes),
        VideoFrameInput(frame_index=2, timestamp_ms=66.7, image_bytes=frame_bytes),
    ]
    summary = processor.process_stream(frames)

    assert summary.frames_processed == 3
    assert summary.unique_tracks == 2
    assert len(summary.final_tracks) == 2
    assert summary.final_tracks[0].consensus_text == "34ABC123"
    assert summary.final_tracks[1].consensus_text == "06DEF99"
