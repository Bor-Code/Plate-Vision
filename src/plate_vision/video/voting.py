from collections import defaultdict, deque

from pydantic import BaseModel, ConfigDict, Field

from plate_vision.pipeline.runner import PlatePredictionResult


class VoteObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plate_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    is_valid_format: bool


class StabilizedTrackPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    track_id: int = Field(ge=1)
    consensus_text: str
    consensus_confidence: float = Field(ge=0.0, le=1.0)
    is_valid_format: bool
    vote_count: int = Field(ge=1)
    agreement_ratio: float = Field(ge=0.0, le=1.0)


class TemporalVoter:
    def __init__(
        self,
        *,
        window_size: int = 15,
        valid_format_bonus: float = 1.25,
    ) -> None:
        if window_size < 1:
            msg = "window_size must be >= 1"
            raise ValueError(msg)
        if valid_format_bonus < 1.0:
            msg = "valid_format_bonus must be >= 1.0"
            raise ValueError(msg)

        self.window_size = window_size
        self.valid_format_bonus = valid_format_bonus
        self._history: dict[int, deque[VoteObservation]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

    def observe(
        self,
        track_id: int,
        prediction: PlatePredictionResult,
    ) -> StabilizedTrackPrediction:
        buf = self._history[track_id]
        buf.append(
            VoteObservation(
                plate_text=prediction.plate_text,
                confidence=prediction.combined_confidence,
                is_valid_format=prediction.is_valid_format,
            )
        )
        return self.get_consensus(track_id)

    def get_consensus(self, track_id: int) -> StabilizedTrackPrediction:
        buf = self._history.get(track_id)
        if not buf:
            msg = f"No observations recorded for track_id={track_id}"
            raise KeyError(msg)

        weighted_scores: dict[str, float] = defaultdict(float)
        raw_conf_sums: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        validity_map: dict[str, bool] = {}

        for obs in buf:
            text = obs.plate_text
            multiplier = self.valid_format_bonus if obs.is_valid_format else 1.0
            weighted_scores[text] += max(1e-4, obs.confidence) * multiplier
            raw_conf_sums[text] += obs.confidence
            counts[text] += 1
            validity_map[text] = obs.is_valid_format

        best_text = max(
            weighted_scores.keys(),
            key=lambda candidate: (
                weighted_scores[candidate],
                counts[candidate],
                validity_map[candidate],
                candidate,
            ),
        )
        winner_count = counts[best_text]
        mean_conf = raw_conf_sums[best_text] / float(winner_count)
        agreement = winner_count / float(len(buf))

        return StabilizedTrackPrediction(
            track_id=track_id,
            consensus_text=best_text,
            consensus_confidence=round(min(1.0, max(0.0, mean_conf)), 6),
            is_valid_format=validity_map[best_text],
            vote_count=len(buf),
            agreement_ratio=round(agreement, 6),
        )


__all__ = [
    "StabilizedTrackPrediction",
    "TemporalVoter",
    "VoteObservation",
]
