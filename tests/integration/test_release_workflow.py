from pathlib import Path

import yaml

from plate_vision import __version__
from plate_vision.utils.release import generate_changelog_markdown, validate_release_tag

ROOT_DIR = Path(__file__).resolve().parents[2]
RELEASE_WORKFLOW_PATH = ROOT_DIR / ".github" / "workflows" / "release.yml"


def test_release_workflow_yaml_and_ghcr_publishing_steps() -> None:
    assert RELEASE_WORKFLOW_PATH.is_file()
    raw = RELEASE_WORKFLOW_PATH.read_text(encoding="utf-8")
    comment_lines = [line for line in raw.splitlines() if line.lstrip().startswith("#")]
    assert comment_lines == []

    parsed = yaml.safe_load(raw)
    assert isinstance(parsed, dict)
    assert parsed["name"] == "Release"
    assert "v[0-9]+.[0-9]+.[0-9]+" in parsed["on"]["push"]["tags"]
    assert parsed["permissions"]["packages"] == "write"
    assert parsed["permissions"]["contents"] == "write"

    steps = parsed["jobs"]["release"]["steps"]
    uses_list = [step.get("uses", "") for step in steps if "uses" in step]
    assert "docker/login-action@v3" in uses_list
    assert "docker/metadata-action@v5" in uses_list
    assert "docker/build-push-action@v6" in uses_list

    tag = f"v{__version__}"
    ver = validate_release_tag(tag, __version__)
    notes = generate_changelog_markdown(
        ver.to_git_tag(),
        ["feat(release): publish container image to ghcr.io"],
    )
    assert f"Release {tag}" in notes
