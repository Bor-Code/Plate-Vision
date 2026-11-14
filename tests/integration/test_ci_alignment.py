import tomllib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def test_ci_python_version_matches_project_files() -> None:
    python_version = (ROOT_DIR / ".python-version").read_text(encoding="utf-8").strip()
    ci_content = (ROOT_DIR / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    pyproject_data = tomllib.loads((ROOT_DIR / "pyproject.toml").read_text(encoding="utf-8"))

    assert f'python-version: "{python_version}"' in ci_content
    assert pyproject_data["tool"]["mypy"]["python_version"] == python_version


def test_ci_commands_align_with_makefile_targets() -> None:
    ci_content = (ROOT_DIR / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    makefile_content = (ROOT_DIR / "Makefile").read_text(encoding="utf-8")

    shared_commands = [
        "uv run ruff check .",
        "uv run ruff format --check .",
        "uv run mypy src tests",
    ]
    for command in shared_commands:
        assert command in ci_content
        assert command in makefile_content
