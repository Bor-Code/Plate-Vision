import math
from collections import Counter, defaultdict

from pydantic import BaseModel, ConfigDict, Field

from plate_vision.video.processor import VideoStreamSummary


class CalibrationZone(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pixels_per_meter: float = Field(default=25.0, gt=0.0)
    speed_limit_kmh: float = Field(default=90.0, gt=0.0)


class TrackAnalyticsSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    track_id: int = Field(ge=1)
    plate_text: str
    country_code: str
    observations: int = Field(ge=1)
    dwell_time_seconds: float = Field(ge=0.0)
    displacement_pixels: float = Field(ge=0.0)
    estimated_speed_kmh: float | None = None
    is_speeding: bool = False


class TrafficAnalyticsReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    total_vehicles: int = Field(ge=0)
    mean_dwell_time_seconds: float = Field(ge=0.0)
    mean_speed_kmh: float | None = None
    speeding_count: int = Field(ge=0)
    country_distribution: dict[str, int]
    track_summaries: tuple[TrackAnalyticsSummary, ...]


class VideoTrafficAnalyzer:
    def __init__(self, zone: CalibrationZone | None = None) -> None:
        self.zone = zone if zone is not None else CalibrationZone()

    def analyze(self, stream_summary: VideoStreamSummary) -> TrafficAnalyticsReport:
        observations_by_track: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
        country_votes: dict[int, Counter[str]] = defaultdict(Counter)

        for frame_res in stream_summary.frame_results:
            for det in frame_res.tracked_detections:
                cx = (det.prediction.bbox.x_min + det.prediction.bbox.x_max) / 2.0
                cy = (det.prediction.bbox.y_min + det.prediction.bbox.y_max) / 2.0
                observations_by_track[det.track_id].append((frame_res.timestamp_ms, cx, cy))
                country_votes[det.track_id][det.prediction.country_code] += 1

        track_summaries: list[TrackAnalyticsSummary] = []
        country_distribution: Counter[str] = Counter()

        for track in stream_summary.final_tracks:
            pts = observations_by_track.get(track.track_id, [])
            if pts:
                pts_sorted = sorted(pts, key=lambda item: item[0])
                first_ts, first_cx, first_cy = pts_sorted[0]
                last_ts, last_cx, last_cy = pts_sorted[-1]
                raw_dwell = max(0.0, (last_ts - first_ts) / 1000.0)
                raw_disp = math.hypot(last_cx - first_cx, last_cy - first_cy)
                obs_count = len(pts_sorted)
            else:
                raw_dwell = 0.0
                raw_disp = 0.0
                obs_count = track.vote_count

            votes = country_votes.get(track.track_id)
            country_code = votes.most_common(1)[0][0] if votes else "TR"
            country_distribution[country_code] += 1

            estimated_speed: float | None = None
            is_speeding = False
            if raw_dwell > 0.0:
                distance_m = raw_disp / self.zone.pixels_per_meter
                estimated_speed = round((distance_m / raw_dwell) * 3.6, 4)
                is_speeding = estimated_speed > self.zone.speed_limit_kmh

            track_summaries.append(
                TrackAnalyticsSummary(
                    track_id=track.track_id,
                    plate_text=track.consensus_text,
                    country_code=country_code,
                    observations=obs_count,
                    dwell_time_seconds=round(raw_dwell, 4),
                    displacement_pixels=round(raw_disp, 4),
                    estimated_speed_kmh=estimated_speed,
                    is_speeding=is_speeding,
                )
            )

        total_vehicles = len(track_summaries)
        mean_dwell = (
            round(
                sum(item.dwell_time_seconds for item in track_summaries) / total_vehicles,
                4,
            )
            if total_vehicles > 0
            else 0.0
        )
        speeds = [
            item.estimated_speed_kmh
            for item in track_summaries
            if item.estimated_speed_kmh is not None
        ]
        mean_speed = round(sum(speeds) / len(speeds), 4) if speeds else None
        speeding_count = sum(1 for item in track_summaries if item.is_speeding)

        return TrafficAnalyticsReport(
            total_vehicles=total_vehicles,
            mean_dwell_time_seconds=mean_dwell,
            mean_speed_kmh=mean_speed,
            speeding_count=speeding_count,
            country_distribution=dict(sorted(country_distribution.items())),
            track_summaries=tuple(track_summaries),
        )


__all__ = [
    "CalibrationZone",
    "TrackAnalyticsSummary",
    "TrafficAnalyticsReport",
    "VideoTrafficAnalyzer",
]
