from collections.abc import Sequence

from plate_vision.data.models import BoundingBox
from plate_vision.data.synthetic import encode_grayscale_png
from plate_vision.detection.models import DetectionPrediction
from plate_vision.pipeline.runner import PlateVisionPipeline
from plate_vision.recognition.decoding import DecodedPlate
from plate_vision.video.analytics import CalibrationZone, VideoTrafficAnalyzer
from plate_vision.video.processor import VideoFrameInput, VideoStreamProcessor


class _MovingPlateDetector:
    def __init__(self) -> None:
        self._frame = 0

    def detect(
        self,
        image_bytes: bytes,
        image_width: int,
        image_height: int,
    ) -> list[DetectionPrediction]:
        _ = (image_bytes, image_width, image_height)
        offset = float(self._frame * 10)
        self._frame += 1
        return [
            DetectionPrediction(
                bbox=BoundingBox(
                    x_min=10.0 + offset,
                    y_min=8.0,
                    x_max=80.0 + offset,
                    y_max=34.0,
                ),
                confidence=0.95,
                class_id=0,
            )
        ]


class _ConstantPlateRecognizer:
    def recognize(
        self,
        crop_pixels: Sequence[Sequence[int]],
        *,
        use_beam_search: bool = False,
    ) -> DecodedPlate:
        _ = (crop_pixels, use_beam_search)
        return DecodedPlate(text="34ABC123", confidence=0.94, token_indices=(1,))


def test_video_stream_to_traffic_analytics_end_to_end() -> None:
    pipeline = PlateVisionPipeline(
        detector=_MovingPlateDetector(),
        recognizer=_ConstantPlateRecognizer(),
    )
    processor = VideoStreamProcessor(pipeline=pipeline)
    png_bytes = encode_grayscale_png([[140 for _ in range(160)] for _ in range(48)])

    frames = [
        VideoFrameInput(frame_index=idx, timestamp_ms=float(idx * 500), image_bytes=png_bytes)
        for idx in range(3)
    ]
    stream_summary = processor.process_stream(frames)
    assert stream_summary.frames_processed == 3
    assert stream_summary.unique_tracks == 1

    analyzer = VideoTrafficAnalyzer(CalibrationZone(pixels_per_meter=20.0, speed_limit_kmh=50.0))
    report = analyzer.analyze(stream_summary)
    assert report.total_vehicles == 1
    assert report.country_distribution == {"TR": 1}
    track = report.track_summaries[0]
    assert track.observations == 3
    assert track.dwell_time_seconds == 1.0
    assert track.displacement_pixels == 20.0
    assert track.estimated_speed_kmh == 3.6
    assert track.is_speeding is False
