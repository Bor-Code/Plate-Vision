FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /build

COPY --from=ghcr.io/astral-sh/uv:0.5.14 /uv /uvx /bin/
COPY pyproject.toml uv.lock .python-version LICENSE ./
COPY src/ ./src/

RUN uv sync --frozen --no-dev --no-editable


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    PLATE_VISION_ENVIRONMENT="production" \
    PLATE_VISION_SERVING__HOST="0.0.0.0" \
    PLATE_VISION_SERVING__PORT="8000"

RUN groupadd --gid 10001 appgroup && \
    useradd --uid 10001 --gid appgroup --shell /usr/sbin/nologin --create-home appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appgroup /build/.venv /app/.venv
COPY --chown=appuser:appgroup configs/ /app/configs/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"]

CMD ["uvicorn", "plate_vision.serving.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
