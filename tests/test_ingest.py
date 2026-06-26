"""Tests for the 3-tier section/parent fallback in app.ingest. Pure logic, no API calls."""
from app import config
from app.ingest import _split_into_sections


def test_section_headers_used_when_present():
    text = (
        "## §1 Purpose\nThis section explains purpose.\n\n"
        "## §2 Scope\nThis section explains scope."
    )
    sections = _split_into_sections(text, "Some Policy")
    assert [s.metadata["section"] for s in sections] == ["§1 Purpose", "§2 Scope"]
    assert all(s.metadata["source_doc"] == "Some Policy" for s in sections)


def test_markdown_headers_used_when_no_section_markers():
    text = "# Doc Title\n\n## Background\nSome background text.\n\n## Requirements\nSome requirements text."
    sections = _split_into_sections(text, "Uploaded Doc")
    titles = [s.metadata["section"] for s in sections]
    assert "Background" in titles
    assert "Requirements" in titles


def test_unstructured_long_text_falls_back_to_capped_pseudo_sections():
    long_text = "word " * (config.PARENT_MAX_CHARS // 3)  # well over PARENT_MAX_CHARS, no headers
    assert len(long_text) > config.PARENT_MAX_CHARS

    sections = _split_into_sections(long_text, "Plain Upload")

    assert len(sections) > 1
    for s in sections:
        assert len(s.page_content) <= config.PARENT_MAX_CHARS
    assert sections[0].metadata["section"].startswith("Part")


def test_oversized_real_section_is_capped_into_parts():
    body = "x" * (config.PARENT_MAX_CHARS * 2)
    text = f"## §1 Huge Section\n{body}"
    sections = _split_into_sections(text, "Some Policy")

    assert len(sections) > 1
    for s in sections:
        assert len(s.page_content) <= config.PARENT_MAX_CHARS
    assert sections[0].metadata["section"] == "§1 Huge Section"
    assert sections[1].metadata["section"] == "§1 Huge Section (part 2)"
