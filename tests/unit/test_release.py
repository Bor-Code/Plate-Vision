import pytest

from plate_vision.utils.exceptions import ConfigurationError
from plate_vision.utils.release import (
    bump_semantic_version,
    generate_changelog_markdown,
    parse_semantic_version,
    validate_release_tag,
)


def test_parse_and_bump_semantic_version() -> None:
    ver = parse_semantic_version("v0.1.0")
    assert ver.major == 0
    assert ver.minor == 1
    assert ver.patch == 0
    assert ver.to_version_string() == "0.1.0"
    assert ver.to_git_tag() == "v0.1.0"

    assert bump_semantic_version(ver, "patch").to_version_string() == "0.1.1"
    assert bump_semantic_version(ver, "minor").to_version_string() == "0.2.0"
    assert bump_semantic_version(ver, "major").to_version_string() == "1.0.0"

    with pytest.raises(ConfigurationError, match="Invalid semantic version"):
        parse_semantic_version("0.1")
    with pytest.raises(ConfigurationError, match="Unsupported semantic version bump"):
        bump_semantic_version(ver, "prerelease")


def test_validate_release_tag_checks_prefix_and_parity() -> None:
    validated = validate_release_tag("v0.1.0", "0.1.0")
    assert validated.to_git_tag() == "v0.1.0"

    with pytest.raises(ConfigurationError, match="start with 'v'"):
        validate_release_tag("0.1.0", "0.1.0")

    with pytest.raises(ConfigurationError, match="does not match"):
        validate_release_tag("v0.2.0", "0.1.0")


def test_generate_changelog_markdown_groups_conventional_commits() -> None:
    commits = [
        "feat(serving): add FastAPI inference service",
        "fix(pipeline)!: handle truncated WebP headers",
        "perf(export): add dynamic int8 quantization",
        "docs: add README and architecture overview",
        "ci: add GHCR release workflow",
        "test(video): add temporal voter unit tests",
        "Merge pull request #16 from Bor-Code/docs/comprehensive-documentation",
    ]
    md = generate_changelog_markdown("v0.1.0", commits)
    assert "Release v0.1.0" in md
    assert "Features:" in md
    assert "- **serving**: add FastAPI inference service" in md
    assert "- [BREAKING] **pipeline**: handle truncated WebP headers" in md
    assert "Build and CI:" in md

    empty_md = generate_changelog_markdown("v0.1.0", ["Merge branch 'main'"])
    assert "- Maintenance release." in empty_md
