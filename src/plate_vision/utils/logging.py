from __future__ import annotations

import logging
import platform
import sys
from typing import IO, TYPE_CHECKING, Any

import structlog

from plate_vision import __version__

if TYPE_CHECKING:
    from structlog.types import Processor


def configure_logging(
    *,
    level: str = "INFO",
    json_format: bool = True,
    service_name: str = "plate-vision",
    include_timestamp: bool = True,
    stream: IO[str] | None = None,
) -> None:
    target_stream = stream if stream is not None else sys.stdout
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        msg = f"Unsupported log level: {level}"
        raise ValueError(msg)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    handler = logging.StreamHandler(target_stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
    ]
    if include_timestamp:
        shared_processors.append(structlog.processors.TimeStamper(fmt="iso", utc=True))

    shared_processors.extend(
        [
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]
    )

    renderer: Processor = (
        structlog.processors.JSONRenderer(sort_keys=True)
        if json_format
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(service=service_name, version=__version__)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger


def bind_context(**kwargs: Any) -> None:
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()


def collect_environment_info(seed: int) -> dict[str, str | int]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "package_version": __version__,
        "seed": seed,
    }


def log_environment_info(
    logger: structlog.stdlib.BoundLogger,
    seed: int,
) -> dict[str, str | int]:
    env_info = collect_environment_info(seed=seed)
    logger.info("environment_initialized", **env_info)
    return env_info


__all__ = [
    "bind_context",
    "clear_context",
    "collect_environment_info",
    "configure_logging",
    "get_logger",
    "log_environment_info",
]
