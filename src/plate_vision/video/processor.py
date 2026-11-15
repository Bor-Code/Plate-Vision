from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from plate_vision.pipeline.runner import PlateVisionPipeline
from plate_vision.video.tracker import (
    MultiObjectTrackerProtocol,
    PlateTracker,
    TrackedPlateDetection,
)
from plate_vision.video.voting import StabilizedTrackPrediction, TemporalVoter


class VideoFrameInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    frame_index: int = Field(ge=0)
    timestamp_ms: float = Field(ge=0.0)
    image_bytes: bytes


class VideoFrameResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    frame_index: int = Field(ge=0)
    timestamp_ms: float = Field(ge=0.0)
    tracked_detections: tuple[TrackedPlateDetection, ...]
    stabilized_predictions: tuple[StabilizedTrackPrediction, ...]
    frame_latency_ms: float = Field(ge=0.0)


class VideoStreamSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    frames_processed: int = Field(ge=0)
    unique_tracks: int = Field(ge=0)
    final_tracks: tuple[StabilizedTrackPrediction, ...]
    frame_results: tuple[VideoFrameResult, ...]


class VideoStreamProcessor:
    def __init__(
        self,
        pipeline: PlateVisionPipeline,
        tracker: MultiObjectTrackerProtocol | None = None,
        voter: TemporalVoter | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.tracker = tracker if tracker is not None else PlateTracker()
        self.voter = voter if voter is not None else TemporalVoter()
        self._seen_track_ids: set[int] = set()

    def process_frame(self, frame: VideoFrameInput) -> VideoFrameResult:
        infer_res = self.pipeline.predict_bytes(frame.image_bytes)
        tracked = self.tracker.update(frame.frame_index, infer_res.plates)
        stabilized: list[StabilizedTrackPrediction] = []
        for item in tracked:
            self._seen_track_ids.add(item.track_id)
            consensus = self.voter.observe(item.track_id, item.prediction)
            stabilized.append(consensus)

        return VideoFrameResult(
            frame_index=frame.frame_index,
            timestamp_ms=frame.timestamp_ms,
            tracked_detections=tracked,
            stabilized_predictions=tuple(stabilized),
            frame_latency_ms=infer_res.latency_ms,
        )

    def process_stream(self, frames: Iterable[VideoFrameInput]) -> VideoStreamSummary:
        frame_results: list[VideoFrameResult] = []
        for frame in frames:
            frame_results.append(self.process_frame(frame))

        final_tracks = tuple(self.voter.get_consensus(tid) for tid in sorted(self._seen_track_ids))
        return VideoStreamSummary(
            frames_processed=len(frame_results),
            unique_tracks=len(self._seen_track_ids),
            final_tracks=final_tracks,
            frame_results=tuple(frame_results),
        )


__all__ = [
    "VideoFrameInput",
    "VideoFrameResult",
    "VideoStreamProcessor",
    "VideoStreamSummary",
]
