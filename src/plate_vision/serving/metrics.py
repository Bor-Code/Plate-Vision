from collections import Counter

from pydantic import BaseModel, ConfigDict, Field


class MetricsSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requests_total: int = Field(ge=0)
    errors_total: int = Field(ge=0)
    plates_detected_total: int = Field(ge=0)
    latency_sum_ms: float = Field(ge=0.0)
    latency_count: int = Field(ge=0)


class PrometheusMetricsRegistry:
    def __init__(self) -> None:
        self._requests_by_route: Counter[tuple[str, int]] = Counter()
        self._errors_by_code: Counter[str] = Counter()
        self._plates_detected_total = 0
        self._latency_buckets_ms: tuple[float, ...] = (
            5.0,
            10.0,
            25.0,
            50.0,
            100.0,
            250.0,
            500.0,
            1000.0,
        )
        self._bucket_counts: dict[float, int] = dict.fromkeys(self._latency_buckets_ms, 0)
        self._latency_inf_count = 0
        self._latency_sum_ms = 0.0

    def record_request(
        self,
        route: str,
        status_code: int,
        latency_ms: float,
        plates_detected: int = 0,
    ) -> None:
        self._requests_by_route[(route, status_code)] += 1
        self._plates_detected_total += max(0, plates_detected)
        clamped_ms = max(0.0, latency_ms)
        self._latency_sum_ms += clamped_ms
        self._latency_inf_count += 1
        for bound in self._latency_buckets_ms:
            if clamped_ms <= bound:
                self._bucket_counts[bound] += 1

    def record_error(self, error_code: str) -> None:
        self._errors_by_code[error_code] += 1

    def snapshot(self) -> MetricsSnapshot:
        return MetricsSnapshot(
            requests_total=sum(self._requests_by_route.values()),
            errors_total=sum(self._errors_by_code.values()),
            plates_detected_total=self._plates_detected_total,
            latency_sum_ms=round(self._latency_sum_ms, 4),
            latency_count=self._latency_inf_count,
        )

    def render_prometheus_text(self) -> str:
        lines: list[str] = [
            "plate_vision_requests_total",
        ]
        lines.clear()
        for (route, status), count in sorted(self._requests_by_route.items()):
            lines.append(
                f'plate_vision_http_requests_total{{route="{route}",status="{status}"}} {count}'
            )
        for error_code, count in sorted(self._errors_by_code.items()):
            lines.append(f'plate_vision_errors_total{{error_code="{error_code}"}} {count}')
        lines.append(f"plate_vision_plates_detected_total {self._plates_detected_total}")
        for bound in self._latency_buckets_ms:
            lines.append(
                f'plate_vision_inference_latency_ms_bucket{{le="{bound}"}} '
                f"{self._bucket_counts[bound]}"
            )
        lines.append(
            f'plate_vision_inference_latency_ms_bucket{{le="+Inf"}} {self._latency_inf_count}'
        )
        lines.append(f"plate_vision_inference_latency_ms_sum {self._latency_sum_ms:.4f}")
        lines.append(f"plate_vision_inference_latency_ms_count {self._latency_inf_count}")
        return "\n".join(lines) + "\n"


__all__ = [
    "MetricsSnapshot",
    "PrometheusMetricsRegistry",
]
