import re
from collections.abc import Sequence
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from plate_vision.utils.exceptions import ConfigurationError

SEMVER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^v?(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"
)
CONVENTIONAL_COMMIT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(feat|fix|perf|refactor|docs|build|ci|test|chore)(?:\(([^)]+)\))?(!)?:\s+(.+)$"
)


class SemanticVersion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    major: int = Field(ge=0)
    minor: int = Field(ge=0)
    patch: int = Field(ge=0)

    def to_version_string(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def to_git_tag(self) -> str:
        return f"v{self.to_version_string()}"


class ChangelogSection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=1)
    entries: tuple[str, ...]


def parse_semantic_version(version_str: str) -> SemanticVersion:
    match = SEMVER_PATTERN.fullmatch(version_str.strip())
    if match is None:
        raise ConfigurationError(
            f"Invalid semantic version string: {version_str}",
            details={"version": version_str},
        )
    return SemanticVersion(
        major=int(match.group(1)),
        minor=int(match.group(2)),
        patch=int(match.group(3)),
    )


def bump_semantic_version(version: SemanticVersion, part: str) -> SemanticVersion:
    normalized = part.strip().lower()
    if normalized == "major":
        return SemanticVersion(major=version.major + 1, minor=0, patch=0)
    if normalized == "minor":
        return SemanticVersion(major=version.major, minor=version.minor + 1, patch=0)
    if normalized == "patch":
        return SemanticVersion(major=version.major, minor=version.minor, patch=version.patch + 1)
    raise ConfigurationError(
        f"Unsupported semantic version bump part: {part}",
        details={"part": part, "allowed": ["major", "minor", "patch"]},
    )


def validate_release_tag(git_tag: str, package_version: str) -> SemanticVersion:
    cleaned_tag = git_tag.strip()
    if not cleaned_tag.startswith("v"):
        raise ConfigurationError(
            f"Release tag must start with 'v': {git_tag}",
            details={"git_tag": git_tag},
        )
    tag_ver = parse_semantic_version(cleaned_tag)
    pkg_ver = parse_semantic_version(package_version)
    if tag_ver != pkg_ver:
        raise ConfigurationError(
            f"Release tag {cleaned_tag} does not match package version {package_version}",
            details={
                "git_tag": cleaned_tag,
                "package_version": package_version,
            },
        )
    return tag_ver


def generate_changelog_markdown(
    version_tag: str,
    commit_subjects: Sequence[str],
) -> str:
    parsed_ver = parse_semantic_version(version_tag)
    tag_label = parsed_ver.to_git_tag()

    buckets: dict[str, list[str]] = {
        "Features": [],
        "Bug Fixes": [],
        "Performance": [],
        "Refactoring": [],
        "Documentation": [],
        "Build and CI": [],
        "Tests and Chores": [],
    }
    type_to_section: dict[str, str] = {
        "feat": "Features",
        "fix": "Bug Fixes",
        "perf": "Performance",
        "refactor": "Refactoring",
        "docs": "Documentation",
        "build": "Build and CI",
        "ci": "Build and CI",
        "test": "Tests and Chores",
        "chore": "Tests and Chores",
    }

    for raw_subject in commit_subjects:
        subject = raw_subject.strip()
        if not subject:
            continue
        match = CONVENTIONAL_COMMIT_PATTERN.match(subject)
        if match is not None:
            ctype, scope, breaking, description = match.groups()
            section_name = type_to_section[ctype]
            prefix = f"**{scope}**: " if scope else ""
            breaking_tag = "[BREAKING] " if breaking else ""
            buckets[section_name].append(f"- {breaking_tag}{prefix}{description}")

    lines: list[str] = [f"Release {tag_label}", ""]
    for section_title, items in buckets.items():
        if items:
            lines.append(f"{section_title}:")
            lines.extend(items)
            lines.append("")

    if len(lines) == 2:
        lines.append("- Maintenance release.")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "ChangelogSection",
    "SemanticVersion",
    "bump_semantic_version",
    "generate_changelog_markdown",
    "parse_semantic_version",
    "validate_release_tag",
]
