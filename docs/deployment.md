# Container Deployment and Smoke Verification

Plate-Vision provides a production-hardened multi-stage `docker/Dockerfile` and `docker/docker-compose.yml`.

## Container Architecture

- **Builder Stage (`python:3.11-slim`)**: Installs dependencies reproducibly from `uv.lock` (`uv sync --frozen --no-dev --no-editable`) into `/build/.venv`.
- **Runtime Stage (`python:3.11-slim`)**: Copies only `/app/.venv` and `/app/configs`, runs as unprivileged user `appuser` (`UID 10001`, `GID 10001`), exposes port `8000`, and configures a native container `HEALTHCHECK` against `http://127.0.0.1:8000/healthz`.

## Building and Running

```bash
docker build -f docker/Dockerfile -t plate-vision:latest .
docker compose -f docker/docker-compose.yml up -d
```

## CI Smoke Verification

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs an automated container and service smoke check verifying:
1. Multi-stage `Dockerfile` and `docker-compose.yml` structure and non-root execution directives.
2. Live ONNX export and end-to-end FastAPI smoke requests (`/healthz`, `/readyz`, `/v1/predict`, `/metrics`).
