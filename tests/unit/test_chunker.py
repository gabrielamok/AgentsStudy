import socket

import pytest

from agent.rag.chunker import MarkdownChunker
from agent.rag.contracts import DocumentChunk, KnowledgeDocument


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Chunking must not use a network service")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def document(content: str, **changes: object) -> KnowledgeDocument:
    return KnowledgeDocument.model_validate({
        "document_id": "study", "title": "Study", "source_path": "knowledge/study.md",
        "content": content, "metadata": {"tags": ["trusted"]}, **changes,
    })


def test_heading_hierarchy_and_source_identity_are_preserved() -> None:
    source = document("# Study\n\nIntroduction.\n\n## Retrieval\n\nRelevant evidence.\n\n### Sources\n\nCite the evidence.")

    chunks = MarkdownChunker().chunk_document(source)

    assert [chunk.section for chunk in chunks] == ["Study", "Retrieval", "Sources"]
    assert chunks[-1].content == "# Study\n## Retrieval\n### Sources\n\nCite the evidence."
    for index, chunk in enumerate(chunks):
        assert isinstance(chunk, DocumentChunk)
        assert chunk.document_id == source.document_id
        assert chunk.source_title == source.title
        assert chunk.metadata["source_path"] == source.source_path
        assert chunk.metadata["chunk_index"] == index
        assert DocumentChunk.model_validate_json(chunk.model_dump_json()) == chunk
    assert chunks[-1].metadata["heading_path"] == ["Study", "Retrieval", "Sources"]


def test_paragraph_boundaries_are_preferred_over_arbitrary_character_cuts() -> None:
    paragraphs = ["First paragraph has context.", "Second paragraph has evidence.", "Third paragraph has attribution."]

    chunks = MarkdownChunker(max_chunk_size=75, overlap=0).chunk_document(document("\n\n".join(paragraphs)))

    assert len(chunks) == 2
    assert chunks[0].content == "# Study\n\n" + "\n\n".join(paragraphs[:2])
    assert chunks[1].content == "# Study\n\n" + paragraphs[2]


@pytest.mark.parametrize("overlap", [0, 15, 50])
def test_long_paragraphs_preserve_words_and_respect_character_limit(overlap: int) -> None:
    # Arrange
    words = [f"word{index:03d}" for index in range(40)]

    # Act
    chunks = MarkdownChunker(max_chunk_size=65, overlap=overlap).chunk_document(document(" ".join(words)))

    # Assert
    bodies = [chunk.content.split("\n\n", 1)[1].split() for chunk in chunks]
    assert len(chunks) > 1
    assert all(bodies)
    assert all(0 < len(chunk.content) <= 65 for chunk in chunks)
    assert {word for body in bodies for word in body} == set(words)
    assert len({chunk.content for chunk in chunks}) == len(chunks)
    if overlap:
        for first, second in zip(bodies, bodies[1:]):
            shared = [word for word in first if word in second]
            assert shared
            assert shared == first[-len(shared):] == second[:len(shared)]
            assert len(" ".join(shared)) <= overlap
            assert second[-1] not in first
    else:
        assert all(not (set(first) & set(second)) for first, second in zip(bodies, bodies[1:]))


def test_paragraph_overlap_is_whole_word_context_and_never_crosses_sections() -> None:
    source = document("# Study\n\n## Rules\n\nFirst useful context.\n\nSecond useful paragraph.\n\n## Examples\n\nSeparate evidence.")

    chunks = MarkdownChunker(max_chunk_size=60, overlap=15).chunk_document(source)

    assert chunks[1].section == "Rules"
    assert chunks[1].content.endswith("useful context.\n\nSecond useful paragraph.")
    assert chunks[-1].section == "Examples"
    assert "useful context" not in chunks[-1].content
    assert all(len(chunk.content) <= 60 for chunk in chunks)


def test_setext_headings_are_sections_but_fenced_examples_are_not() -> None:
    source = document("Study\n=====\n\nRules\n-----\n\n```markdown\n# Not a section\n\nExample text.\n```\n\n## Sources\n\nReal evidence.")

    chunks = MarkdownChunker().chunk_document(source)

    assert [chunk.section for chunk in chunks] == ["Rules", "Sources"]
    assert "# Not a section" in chunks[0].content
    assert "```markdown" in chunks[0].content


def test_preamble_and_documents_without_headings_keep_title_context() -> None:
    source = document("Preamble context.\n\n## Rules\n\nFollow the evidence.")

    chunks = MarkdownChunker().chunk_document(source)

    assert chunks[0].section == "Study"
    assert chunks[0].content == "# Study\n\nPreamble context."
    assert chunks[1].content == "# Study\n## Rules\n\nFollow the evidence."


def test_heading_only_sections_do_not_produce_empty_chunks() -> None:
    assert MarkdownChunker().chunk_document(document("# Study\n\n## Empty\n\n### Also empty")) == []


def test_duplicate_complete_chunks_are_removed_without_losing_distinct_sections() -> None:
    # Arrange
    source = document("## Rules\n\nSame evidence.\n\n## Rules\n\nSame evidence.\n\n## Examples\n\nSame evidence.")

    # Act
    chunks = MarkdownChunker().chunk_document(source)

    # Assert
    assert [chunk.section for chunk in chunks] == ["Rules", "Examples"]
    assert len({chunk.chunk_id for chunk in chunks}) == 2
    assert len({chunk.content for chunk in chunks}) == len(chunks)


def test_repeated_leaf_headings_keep_distinct_parent_context() -> None:
    source = document("# Agents\n\n## Rules\n\nSame evidence.\n\n# Retrieval\n\n## Rules\n\nSame evidence.")

    chunks = MarkdownChunker().chunk_document(source)

    assert [chunk.section for chunk in chunks] == ["Rules", "Rules"]
    assert chunks[0].chunk_id != chunks[1].chunk_id
    assert chunks[0].metadata["heading_path"] == ["Agents", "Rules"]
    assert chunks[1].metadata["heading_path"] == ["Retrieval", "Rules"]


def test_chunk_ids_are_repeatable_and_document_specific_and_input_is_unchanged() -> None:
    source = document("## Rules\n\nTrusted evidence.")
    before = source.model_dump()
    chunker = MarkdownChunker()

    first = chunker.chunk_document(source)
    second = chunker.chunk_document(source)
    other = chunker.chunk_document(document(source.content, document_id="other"))

    assert first == second
    assert first[0].chunk_id != other[0].chunk_id
    assert source.model_dump() == before
    first[0].metadata["document_metadata"] = {"changed": True}
    assert second[0].metadata["document_metadata"] == source.metadata


@pytest.mark.parametrize(("maximum", "overlap"), [(0, 0), (-1, 0), (50, -1), (50, 50), (50, 51)])
def test_invalid_size_ranges_are_rejected(maximum: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        MarkdownChunker(max_chunk_size=maximum, overlap=overlap)


@pytest.mark.parametrize(("maximum", "overlap"), [(True, 0), (50, False), (50.0, 0), (50, "1")])
def test_configuration_does_not_coerce_noninteger_values(maximum: object, overlap: object) -> None:
    with pytest.raises(TypeError):
        MarkdownChunker(max_chunk_size=maximum, overlap=overlap)


def test_unfittable_word_is_rejected_without_cutting_it() -> None:
    with pytest.raises(ValueError, match="word cannot fit"):
        MarkdownChunker(max_chunk_size=30, overlap=0).chunk_document(document("x" * 40))


def test_unfittable_heading_context_is_rejected() -> None:
    with pytest.raises(ValueError, match="Heading context"):
        MarkdownChunker(max_chunk_size=10, overlap=0).chunk_document(document("## A very long section heading\n\nEvidence."))


def test_txt_documents_do_not_turn_literal_headings_into_sections() -> None:
    source = document("# Literal text\n\nMore text.", metadata={"file_type": ".txt"})

    chunks = MarkdownChunker().chunk_document(source)

    assert len(chunks) == 1
    assert chunks[0].section == "Study"
    assert chunks[0].content.startswith("# Study\n\n# Literal text")