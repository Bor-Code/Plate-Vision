from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
CI_WORKFLOW_PATH = ROOT_DIR / ".github" / "workflows" / "ci.yml"


def read_workflow_lines() -> list[str]:
    content = CI_WORKFLOW_PATH.read_text(encoding="utf-8")
    return [line.strip() for line in content.splitlines() if line.strip()]


def test_ci_workflow_file_exists() -> None:
    assert CI_WORKFLOW_PATH.is_file()


def test_ci_workflow_triggers_and_caching() -> None:
    lines = read_workflow_lines()
    assert "name: CI" in lines
    assert "push:" in lines
    assert "pull_request:" in lines
    assert "- main" in lines
    assert "uses: astral-sh/setup-uv@v4" in lines
    assert "enable-cache: true" in lines
    assert 'cache-dependency-glob: "uv.lock"' in lines
    assert 'python-version: "3.11"' in lines


def test_ci_workflow_quality_gate_commands() -> None:
    lines = read_workflow_lines()
    expected_commands = [
        "run: uv sync --locked --dev",
        "run: uv run ruff check .",
        "run: uv run ruff format --check .",
        "run: uv run mypy src tests",
        "run: uv run pytest --cov=plate_vision --cov-report=term-missing --cov-report=xml",
    ]
    for command in expected_commands:
        assert command in lines


def test_ci_workflow_has_no_comments() -> None:
    raw_lines = CI_WORKFLOW_PATH.read_text(encoding="utf-8").splitlines()
    comment_lines = [line for line in raw_lines if line.lstrip().startswith("#")]
    assert comment_lines == []
