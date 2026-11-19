import itertools
import math
from collections import defaultdict
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from plate_vision.data.profiles import get_country_profile
from plate_vision.utils.exceptions import ConfigurationError, InvalidInputError


def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    earth_radius_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (math.sin(d_phi / 2.0) ** 2) + (
        math.cos(phi1) * math.cos(phi2) * (math.sin(d_lambda / 2.0) ** 2)
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return round(earth_radius_km * c, 6)


class CameraNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    camera_id: str = Field(min_length=1)
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    zone_name: str = Field(default="DEFAULT", min_length=1)


class CameraSighting(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    camera_id: str = Field(min_length=1)
    plate_text: str = Field(min_length=2)
    country_code: str = Field(default="TR", min_length=2)
    timestamp_ms: int = Field(ge=0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class JourneyLeg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    from_camera_id: str = Field(min_length=1)
    to_camera_id: str = Field(min_length=1)
    distance_km: float = Field(ge=0.0)
    duration_seconds: float = Field(ge=0.0)
    estimated_speed_kmh: float | None = None
    is_cloned_plate_anomaly: bool = False


class VehicleJourneyReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plate_text: str = Field(min_length=2)
    country_code: str = Field(min_length=2)
    sightings_count: int = Field(ge=1)
    visited_camera_ids: tuple[str, ...]
    visited_zones: tuple[str, ...]
    total_distance_km: float = Field(ge=0.0)
    total_duration_seconds: float = Field(ge=0.0)
    legs: tuple[JourneyLeg, ...]
    has_cloned_plate_anomaly: bool


class MultiCameraJourneyTracker:
    def __init__(
        self,
        cameras: Sequence[CameraNode] = (),
        *,
        cloned_plate_speed_threshold_kmh: float = 260.0,
    ) -> None:
        if cloned_plate_speed_threshold_kmh <= 0.0:
            raise ConfigurationError(
                "cloned_plate_speed_threshold_kmh must be > 0.0",
                details={"cloned_plate_speed_threshold_kmh": cloned_plate_speed_threshold_kmh},
            )
        self.cloned_plate_speed_threshold_kmh = cloned_plate_speed_threshold_kmh
        self._cameras: dict[str, CameraNode] = {}
        self._sightings: dict[tuple[str, str], list[CameraSighting]] = defaultdict(list)
        for node in cameras:
            self.register_camera(node)

    @staticmethod
    def _normalize_plate(plate_text: str, country_code: str) -> tuple[str, str]:
        code = country_code.strip().upper()
        profile = get_country_profile(code)
        norm = profile.normalize(plate_text)
        if len(norm) < 2:
            raise InvalidInputError(
                "Normalized plate_text must contain at least 2 characters",
                details={"plate_text": plate_text, "country_code": code},
            )
        return code, norm

    def register_camera(self, node: CameraNode) -> None:
        clean_id = node.camera_id.strip()
        if not clean_id:
            raise InvalidInputError("camera_id must not be empty")
        self._cameras[clean_id] = CameraNode(
            camera_id=clean_id,
            latitude=node.latitude,
            longitude=node.longitude,
            zone_name=node.zone_name.strip() or "DEFAULT",
        )

    def record_sighting(self, sighting: CameraSighting) -> CameraSighting:
        cam_id = sighting.camera_id.strip()
        if cam_id not in self._cameras:
            raise InvalidInputError(
                "Unregistered camera_id",
                details={"camera_id": sighting.camera_id},
            )
        code, norm = self._normalize_plate(sighting.plate_text, sighting.country_code)
        normalized_sighting = CameraSighting(
            camera_id=cam_id,
            plate_text=norm,
            country_code=code,
            timestamp_ms=sighting.timestamp_ms,
            confidence=sighting.confidence,
        )
        self._sightings[(code, norm)].append(normalized_sighting)
        return normalized_sighting

    def build_journey(
        self,
        plate_text: str,
        *,
        country_code: str = "TR",
    ) -> VehicleJourneyReport:
        code, norm = self._normalize_plate(plate_text, country_code)
        raw_sightings = self._sightings.get((code, norm))
        if not raw_sightings:
            raise InvalidInputError(
                "No sightings found for vehicle",
                details={"plate_text": norm, "country_code": code},
            )

        ordered = sorted(raw_sightings, key=lambda item: (item.timestamp_ms, item.camera_id))
        visited_cameras = tuple(item.camera_id for item in ordered)
        visited_zones_list: list[str] = []
        seen_zones: set[str] = set()
        for item in ordered:
            zone = self._cameras[item.camera_id].zone_name
            if zone not in seen_zones:
                seen_zones.add(zone)
                visited_zones_list.append(zone)

        legs: list[JourneyLeg] = []
        for prev, curr in itertools.pairwise(ordered):
            cam_from = self._cameras[prev.camera_id]
            cam_to = self._cameras[curr.camera_id]
            raw_dist = haversine_distance_km(
                cam_from.latitude,
                cam_from.longitude,
                cam_to.latitude,
                cam_to.longitude,
            )
            distance_km = round(raw_dist, 4)
            raw_duration = (curr.timestamp_ms - prev.timestamp_ms) / 1000.0
            duration_seconds = round(raw_duration, 4)
            if duration_seconds > 0.0:
                estimated_speed = round((raw_dist / raw_duration) * 3600.0, 4)
                is_anomaly = estimated_speed > self.cloned_plate_speed_threshold_kmh
            else:
                estimated_speed = None
                is_anomaly = distance_km > 0.05

            legs.append(
                JourneyLeg(
                    from_camera_id=prev.camera_id,
                    to_camera_id=curr.camera_id,
                    distance_km=distance_km,
                    duration_seconds=duration_seconds,
                    estimated_speed_kmh=estimated_speed,
                    is_cloned_plate_anomaly=is_anomaly,
                )
            )

        total_distance = round(sum(leg.distance_km for leg in legs), 4)
        total_duration = round(sum(leg.duration_seconds for leg in legs), 4)
        has_anomaly = any(leg.is_cloned_plate_anomaly for leg in legs)
        return VehicleJourneyReport(
            plate_text=norm,
            country_code=code,
            sightings_count=len(ordered),
            visited_camera_ids=visited_cameras,
            visited_zones=tuple(visited_zones_list),
            total_distance_km=total_distance,
            total_duration_seconds=total_duration,
            legs=tuple(legs),
            has_cloned_plate_anomaly=has_anomaly,
        )

    def anomalous_vehicles(self) -> tuple[tuple[str, str], ...]:
        flagged: list[tuple[str, str]] = []
        for code, norm in self._sightings:
            report = self.build_journey(norm, country_code=code)
            if report.has_cloned_plate_anomaly:
                flagged.append((code, norm))
        return tuple(sorted(flagged))


__all__ = [
    "CameraNode",
    "CameraSighting",
    "JourneyLeg",
    "MultiCameraJourneyTracker",
    "VehicleJourneyReport",
    "haversine_distance_km",
]
