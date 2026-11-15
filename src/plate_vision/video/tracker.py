from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from plate_vision.data.models import BoundingBox
from plate_vision.detection.metrics import compute_iou
from plate_vision.pipeline.runner import PlatePredictionResult


class TrackState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    track_id: int = Field(ge=1)
    bbox: BoundingBox
    hits: int = Field(ge=1)
    age: int = Field(ge=1)
    missed_frames: int = Field(ge=0)
    last_frame_index: int = Field(ge=0)


class TrackedPlateDetection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    track_id: int = Field(ge=1)
    frame_index: int = Field(ge=0)
    prediction: PlatePredictionResult
    track_hits: int = Field(ge=1)


class MultiObjectTrackerProtocol(Protocol):
    def update(
        self,
        frame_index: int,
        predictions: Sequence[PlatePredictionResult],
    ) -> tuple[TrackedPlateDetection, ...]: ...

    def active_tracks(self) -> tuple[TrackState, ...]: ...


class PlateTracker:
    def __init__(
        self,
        *,
        iou_threshold: float = 0.30,
        max_missed_frames: int = 5,
        smoothing_alpha: float = 0.70,
    ) -> None:
        if not 0.0 < iou_threshold <= 1.0:
            msg = "iou_threshold must be in (0.0, 1.0]"
            raise ValueError(msg)
        if max_missed_frames < 0:
            msg = "max_missed_frames must be >= 0"
            raise ValueError(msg)
        if not 0.0 < smoothing_alpha <= 1.0:
            msg = "smoothing_alpha must be in (0.0, 1.0]"
            raise ValueError(msg)

        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.smoothing_alpha = smoothing_alpha
        self._next_track_id = 1
        self._tracks: dict[int, TrackState] = {}

    def _smooth_bbox(self, prev: BoundingBox, current: BoundingBox) -> BoundingBox:
        a = self.smoothing_alpha
        b = 1.0 - a
        return BoundingBox(
            x_min=round(a * current.x_min + b * prev.x_min, 4),
            y_min=round(a * current.y_min + b * prev.y_min, 4),
            x_max=round(a * current.x_max + b * prev.x_max, 4),
            y_max=round(a * current.y_max + b * prev.y_max, 4),
        )

    def update(
        self,
        frame_index: int,
        predictions: Sequence[PlatePredictionResult],
    ) -> tuple[TrackedPlateDetection, ...]:
        candidate_pairs: list[tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            for det_idx, pred in enumerate(predictions):
                iou = compute_iou(track.bbox, pred.bbox)
                if iou >= self.iou_threshold:
                    candidate_pairs.append((iou, track_id, det_idx))

        candidate_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
        matched_tracks: set[int] = set()
        matched_dets: dict[int, int] = {}

        for _, track_id, det_idx in candidate_pairs:
            if track_id in matched_tracks or det_idx in matched_dets:
                continue
            matched_tracks.add(track_id)
            matched_dets[det_idx] = track_id

        updated_tracks: dict[int, TrackState] = {}
        for track_id, prev_track in self._tracks.items():
            if track_id not in matched_tracks:
                missed = prev_track.missed_frames + 1
                if missed <= self.max_missed_frames:
                    updated_tracks[track_id] = TrackState(
                        track_id=track_id,
                        bbox=prev_track.bbox,
                        hits=prev_track.hits,
                        age=prev_track.age + 1,
                        missed_frames=missed,
                        last_frame_index=prev_track.last_frame_index,
                    )

        outputs: list[TrackedPlateDetection] = []
        for det_idx, pred in enumerate(predictions):
            if det_idx in matched_dets:
                tid = matched_dets[det_idx]
                prev = self._tracks[tid]
                smoothed = self._smooth_bbox(prev.bbox, pred.bbox)
                new_state = TrackState(
                    track_id=tid,
                    bbox=smoothed,
                    hits=prev.hits + 1,
                    age=prev.age + 1,
                    missed_frames=0,
                    last_frame_index=frame_index,
                )
            else:
                tid = self._next_track_id
                self._next_track_id += 1
                new_state = TrackState(
                    track_id=tid,
                    bbox=pred.bbox,
                    hits=1,
                    age=1,
                    missed_frames=0,
                    last_frame_index=frame_index,
                )
            updated_tracks[tid] = new_state
            outputs.append(
                TrackedPlateDetection(
                    track_id=tid,
                    frame_index=frame_index,
                    prediction=pred,
                    track_hits=new_state.hits,
                )
            )

        self._tracks = updated_tracks
        return tuple(outputs)

    def active_tracks(self) -> tuple[TrackState, ...]:
        return tuple(self._tracks[k] for k in sorted(self._tracks.keys()))


__all__ = [
    "MultiObjectTrackerProtocol",
    "PlateTracker",
    "TrackState",
    "TrackedPlateDetection",
]
