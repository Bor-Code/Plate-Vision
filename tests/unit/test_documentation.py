from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

REQUIRED_DOCS = (
    ROOT_DIR / "README.md",
    ROOT_DIR / "CONTRIBUTING.md",
    ROOT_DIR / "docs" / "architecture.md",
    ROOT_DIR / "docs" / "model_card.md",
    ROOT_DIR / "docs" / "datasets.md",
    ROOT_DIR / "docs" / "benchmarks.md",
    ROOT_DIR / "docs" / "configuration.md",
    ROOT_DIR / "docs" / "detection.md",
    ROOT_DIR / "docs" / "recognition.md",
    ROOT_DIR / "docs" / "experiment_tracking.md",
    ROOT_DIR / "docs" / "evaluation.md",
    ROOT_DIR / "docs" / "pipeline.md",
    ROOT_DIR / "docs" / "export.md",
    ROOT_DIR / "docs" / "serving.md",
    ROOT_DIR / "docs" / "deployment.md",
    ROOT_DIR / "docs" / "video.md",
)


def test_all_required_documentation_files_exist_and_have_no_emojis() -> None:
    for doc_path in REQUIRED_DOCS:
        assert doc_path.is_file(), f"Missing documentation file: {doc_path}"
        content = doc_path.read_text(encoding="utf-8")
        assert len(content.strip()) > 100
        for ch in content:
            assert ord(ch) < 0x1F000, f"Disallowed emoji character in {doc_path}"


def test_architecture_doc_contains_mermaid_diagrams() -> None:
    arch_text = (ROOT_DIR / "docs" / "architecture.md").read_text(encoding="utf-8")
    assert "```mermaid" in arch_text
    assert "PlateVisionPipeline" in arch_text
