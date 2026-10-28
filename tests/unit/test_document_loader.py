import logging
import socket
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent.rag.contracts import KnowledgeDocument
from agent.rag.document_loader import DocumentLoader


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Document loading must not use a network service")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.fixture
def knowledge(tmp_path: Path) -> Path:
    directory = tmp_path / "knowledge"
    directory.mkdir()
    return directory


def test_markdown_preserves_source_identity_title_and_heading_context(knowledge: Path) -> None:
    source = knowledge / "agents.md"
    content = "# Agents\n\n## Concepts\nEvidence.\n\n### Tools\nDetails."
    source.write_text(content, encoding="utf-8")
    loader = DocumentLoader(knowledge)

    document = loader.load_document("agents.md")

    assert isinstance(document, KnowledgeDocument)
    assert document.title == "Agents"
    assert document.content == content
    assert document.source_path == source.resolve().as_posix()
    assert document.metadata == {
        "file_name": "agents.md", "relative_path": "agents.md", "file_type": ".md",
        "headings": [
            {"level": 1, "text": "Agents", "line": 1},
            {"level": 2, "text": "Concepts", "line": 3},
            {"level": 3, "text": "Tools", "line": 6},
        ],
    }
    assert loader.load_document(source) == document


def test_commonmark_headings_exclude_fenced_code_and_support_setext(knowledge: Path) -> None:
    (knowledge / "headings.md").write_text(
        "Study Notes\n===========\n\n```markdown\n# Not a heading\n```\n\nRules\n-----\n\n## Examples ##\n", encoding="utf-8",
    )

    document = DocumentLoader(knowledge).load_document("headings.md")

    assert document is not None
    assert document.title == "Study Notes"
    assert document.metadata["headings"] == [
        {"level": 1, "text": "Study Notes", "line": 1},
        {"level": 2, "text": "Rules", "line": 8},
        {"level": 2, "text": "Examples", "line": 11},
    ]


@pytest.mark.parametrize("name", ["notes.txt", "NOTES.TXT", "notes.md", "NOTES.MD", "notes.markdown", "NOTES.MARKDOWN"])
def test_supported_extensions_and_filename_title_fallback(knowledge: Path, name: str) -> None:
    # Arrange
    (knowledge / name).write_text("Plain study content.", encoding="utf-8")

    # Act
    document = DocumentLoader(knowledge).load_document(name)

    # Assert
    assert document is not None
    assert document.title == Path(name).stem
    assert document.metadata["headings"] == []


def test_txt_is_not_parsed_as_markdown(knowledge: Path) -> None:
    (knowledge / "notes.txt").write_text("# Literal text\n## More text", encoding="utf-8")

    document = DocumentLoader(knowledge).load_document("notes.txt")

    assert document is not None
    assert document.title == "notes"
    assert document.metadata["headings"] == []
    assert document.content.startswith("# Literal text")


def test_recursive_loading_is_sorted_and_ids_are_stable_across_root_locations(knowledge: Path, tmp_path: Path) -> None:
    (knowledge / "nested").mkdir()
    (knowledge / "z.txt").write_text("Z", encoding="utf-8")
    (knowledge / "nested" / "a.md").write_text("# Nested", encoding="utf-8")
    (knowledge / "a.md").write_text("# Root", encoding="utf-8")
    loader = DocumentLoader(knowledge)

    first = loader.load_documents()
    (knowledge / "a.md").write_text("# Changed content", encoding="utf-8")
    second = loader.load_documents()
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    (relocated / "a.md").write_text("# Changed content", encoding="utf-8")
    relocated_document = DocumentLoader(relocated).load_document("a.md")

    assert [document.metadata["relative_path"] for document in first] == ["a.md", "nested/a.md", "z.txt"]
    assert len({document.document_id for document in first}) == 3
    assert [document.document_id for document in first] == [document.document_id for document in second]
    assert relocated_document is not None
    assert relocated_document.document_id == first[0].document_id


@pytest.mark.parametrize("name", ["ignored.json", "ignored.pdf", "ignored.png", "ignored.docx"])
def test_unsupported_files_are_not_read_and_have_a_logged_reason(
    knowledge: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    # Arrange
    (knowledge / name).write_bytes(b"\xff")

    def forbidden_read(*args: object, **kwargs: object) -> str:
        raise AssertionError("Unsupported files must not be read")

    monkeypatch.setattr(Path, "read_text", forbidden_read)

    # Act
    with caplog.at_level(logging.INFO):
        documents = DocumentLoader(knowledge).load_documents()

    # Assert
    assert documents == []
    assert name in caplog.text
    assert "unsupported file type" in caplog.text


@pytest.mark.parametrize("absolute", [False, True])
def test_outside_paths_are_rejected_before_reading(
    knowledge: Path, tmp_path: Path, absolute: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    outside = tmp_path / "outside.md"
    outside.write_text("# Outside", encoding="utf-8")
    path = outside if absolute else Path("../outside.md")
    reads: list[Path] = []

    def forbidden_read(source: Path, *args: object, **kwargs: object) -> str:
        reads.append(source)
        raise AssertionError("Outside files must be rejected before reading")

    monkeypatch.setattr(Path, "read_text", forbidden_read)

    # Act
    with pytest.raises(PermissionError, match="outside the configured knowledge directory"):
        DocumentLoader(knowledge).load_document(path)

    # Assert
    assert reads == []


def test_resolved_link_escape_is_rejected(knowledge: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loader = DocumentLoader(knowledge)
    outside = tmp_path / "outside.md"
    outside.write_text("# Outside", encoding="utf-8")
    link = knowledge / "link.md"
    original_resolve = Path.resolve

    def resolve(path: Path, strict: bool = False) -> Path:
        return outside if path == link else original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", resolve)

    with pytest.raises(PermissionError, match="outside the configured knowledge directory"):
        loader.load_document("link.md")


def test_invalid_utf8_is_logged_and_propagated(knowledge: Path, caplog: pytest.LogCaptureFixture) -> None:
    (knowledge / "invalid.txt").write_bytes(b"\xff\xfe")

    with pytest.raises(UnicodeDecodeError):
        DocumentLoader(knowledge).load_documents()

    assert "Cannot read knowledge document" in caplog.text
    assert "invalid.txt" in caplog.text


def test_reading_failure_is_logged_and_propagated(knowledge: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    (knowledge / "protected.md").write_text("# Protected", encoding="utf-8")

    def denied(*args: object, **kwargs: object) -> str:
        raise PermissionError("Reading denied")

    monkeypatch.setattr(Path, "read_text", denied)

    with pytest.raises(PermissionError, match="Reading denied"):
        DocumentLoader(knowledge).load_documents()

    assert "protected.md" in caplog.text


def test_empty_documents_fail_contract_validation(knowledge: Path) -> None:
    (knowledge / "empty.md").write_text(" \n\t", encoding="utf-8")

    with pytest.raises(ValidationError):
        DocumentLoader(knowledge).load_documents()


def test_missing_sources_and_invalid_roots_are_not_silently_ignored(knowledge: Path) -> None:
    with pytest.raises(FileNotFoundError):
        DocumentLoader(knowledge / "missing")
    source = knowledge / "file.txt"
    source.write_text("Notes", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        DocumentLoader(source)
    with pytest.raises(FileNotFoundError):
        DocumentLoader(knowledge).load_document("missing.md")
    with pytest.raises(IsADirectoryError):
        DocumentLoader(knowledge).load_document(".")


def test_empty_knowledge_directory_returns_no_documents(knowledge: Path) -> None:
    assert DocumentLoader(knowledge).load_documents() == []


def test_utf8_text_and_heading_metadata_preserve_unicode(knowledge: Path) -> None:
    content = "# Caf\u00e9\n\n## R\u00e9sum\u00e9\nStudy notes."
    (knowledge / "unicode.md").write_text(content, encoding="utf-8")

    document = DocumentLoader(knowledge).load_document("unicode.md")

    assert document is not None
    assert document.title == "Caf\u00e9"
    assert document.content == content
    assert document.metadata["headings"] == [
        {"level": 1, "text": "Caf\u00e9", "line": 1},
        {"level": 2, "text": "R\u00e9sum\u00e9", "line": 3},
    ]


def test_enumeration_errors_are_logged_and_propagated(knowledge: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def denied(
        directory: Path, *, followlinks: bool, onerror: Callable[[OSError], None],
    ) -> Iterator[tuple[str, list[str], list[str]]]:
        onerror(PermissionError("Cannot enumerate protected directory"))
        return iter(())

    monkeypatch.setattr("agent.rag.document_loader.os.walk", denied)

    with pytest.raises(PermissionError, match="Cannot enumerate protected directory"):
        DocumentLoader(knowledge).load_documents()

    assert "Cannot enumerate knowledge directory" in caplog.text