from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
DOCKERFILE_PATH = ROOT_DIR / "docker" / "Dockerfile"
COMPOSE_PATH = ROOT_DIR / "docker" / "docker-compose.yml"


def test_dockerfile_is_multistage_non_root_and_comment_free() -> None:
    assert DOCKERFILE_PATH.is_file()
    raw = DOCKERFILE_PATH.read_text(encoding="utf-8")
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    comment_lines = [line for line in lines if line.startswith("#")]
    assert comment_lines == []

    from_lines = [line for line in lines if line.startswith("FROM ")]
    assert len(from_lines) == 2
    assert "AS builder" in from_lines[0]
    assert "AS runtime" in from_lines[1]
    assert "USER appuser" in lines
    assert any(line.startswith("HEALTHCHECK ") for line in lines)


def test_docker_compose_is_valid_yaml_with_healthcheck_and_no_comments() -> None:
    assert COMPOSE_PATH.is_file()
    raw = COMPOSE_PATH.read_text(encoding="utf-8")
    comment_lines = [line for line in raw.splitlines() if line.lstrip().startswith("#")]
    assert comment_lines == []

    parsed = yaml.safe_load(raw)
    assert isinstance(parsed, dict)
    services = parsed.get("services", {})
    assert "plate-vision-api" in services
    service = services["plate-vision-api"]
    assert service["build"]["dockerfile"] == "docker/Dockerfile"
    assert "healthcheck" in service
    assert service["healthcheck"]["retries"] == 3
