import socket
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import pytest

from agent.rag.chunker import MarkdownChunker
from agent.rag.contracts import RetrievalQuery, RetrievalResult
from agent.rag.document_loader import DocumentLoader
from agent.rag.service import RAGService


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Retriever tests must not access Azure or other network services")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.fixture
def knowledge(tmp_path: Path) -> Path:
    directory = tmp_path / "knowledge"
    directory.mkdir()
    documents = {
        "rag.md": (
            "# Grounding\n\n## Evidence\n\n"
            "Retrieval-augmented generation (RAG) retrieves evidence from a knowledge base. "
            "Retrieval uses document chunks. Retrieval connects queries to knowledge."
        ),
        "index.md": (
            "# Indexing\n\n## Ranking\n\n"
            "Retrieval ranks document chunks using lexical scores."
        ),
        "sources.md": (
            "# Attribution\n\n## Citations\n\n"
            "Retrieval results retain source titles and sections for citations."
        ),
        "tuning.md": (
            "# Adaptation\n\n## Parameters\n\n"
            "Fine-tuning updates model parameters using training examples."
        ),
    }
    for name, content in documents.items():
        (directory / name).write_text(content, encoding="utf-8")
    return directory


@pytest.fixture
def service(knowledge: Path) -> Iterator[RAGService]:
    instance = RAGService(knowledge)
    yield instance
    instance.close()


@pytest.mark.parametrize(("terminology", "title", "section"), [
    ("RAG", "Grounding", "Evidence"),
    ("fine-tuning", "Adaptation", "Parameters"),
    ("lexical", "Indexing", "Ranking"),
])
def test_exact_terminology_retrieves_the_expected_source(
    service: RAGService, terminology: str, title: str, section: str,
) -> None:
    # Arrange
    query = RetrievalQuery(query=terminology, top_k=10, filters={})

    # Act
    results = service.retrieve(query)

    # Assert
    assert len(results) == 1
    assert results[0].source_title == title
    assert results[0].section == section
    assert results[0].score > 0


@pytest.mark.parametrize(("query_text", "expected_title"), [
    ("Where does RAG get its evidence?", "Grounding"),
    ("How are document chunks ordered using lexical scores?", "Indexing"),
    ("What information supports source citations?", "Attribution"),
])
def test_paraphrased_questions_with_shared_terms_find_evidence(
    service: RAGService, query_text: str, expected_title: str,
) -> None:
    # Arrange
    query = RetrievalQuery(query=query_text, top_k=10, filters={})

    # Act
    results = service.retrieve(query)

    # Assert
    assert expected_title in {result.source_title for result in results}
    assert all(result.score > 0 for result in results)


@pytest.mark.parametrize("top_k", [1, 2, 3, 10])
def test_top_k_returns_only_the_highest_ranked_available_matches(service: RAGService, top_k: int) -> None:
    # Arrange
    all_results = service.retrieve(RetrievalQuery(query="retrieval", top_k=10, filters={}))
    query = RetrievalQuery(query="retrieval", top_k=top_k, filters={})

    # Act
    results = service.retrieve(query)

    # Assert
    assert len(all_results) == 3
    assert len(results) == min(top_k, 3)
    assert results == all_results[:top_k]


def test_scores_are_descending_with_repeatable_tie_breaking(service: RAGService) -> None:
    # Arrange
    query = RetrievalQuery(query="retrieval", top_k=10, filters={})

    # Act
    results = service.retrieve(query)
    repeated = service.retrieve(query)
    service.index_documents()
    rebuilt = service.retrieve(query)

    # Assert
    assert len(results) == 3
    assert len({result.score for result in results}) > 1
    assert results == sorted(results, key=lambda result: (-result.score, result.chunk_id))
    assert results == repeated == rebuilt


def test_equal_scores_are_ordered_by_stable_chunk_id(service: RAGService, knowledge: Path) -> None:
    # Arrange
    content = (knowledge / "sources.md").read_text(encoding="utf-8")
    (knowledge / "sources-copy.md").write_text(content, encoding="utf-8")
    query = RetrievalQuery(query="citations", top_k=10, filters={})

    # Act
    results = service.retrieve(query)
    service.index_documents()
    rebuilt = service.retrieve(query)

    # Assert
    assert len(results) == 2
    assert results[0].score == results[1].score
    assert results[0].chunk_id != results[1].chunk_id
    assert [result.chunk_id for result in results] == sorted(result.chunk_id for result in results)
    assert results == rebuilt


@pytest.mark.parametrize("rank", [-0.25, 0.0, 0.25])
def test_relevance_threshold_requires_a_strictly_positive_returned_score(
    service: RAGService, monkeypatch: pytest.MonkeyPatch, rank: float,
) -> None:
    # Arrange
    query = RetrievalQuery(query="RAG", top_k=3, filters={})
    indexed = service.retrieve(query)
    connection = Mock(spec=sqlite3.Connection)
    connection.execute.return_value = [(indexed[0].chunk_id, rank)]

    # Act
    with monkeypatch.context() as patch:
        patch.setattr(service, "_index", connection)
        results = service.retrieve(query)

    # Assert
    if rank < 0:
        assert len(results) == 1
        assert results[0].chunk_id == indexed[0].chunk_id
        assert results[0].score == -rank
    else:
        assert results == []


@pytest.mark.parametrize("query_text", ["quasar astrophysics", "what is the", "!!!"])
def test_no_result_queries_do_not_fall_back_to_unrelated_sources(service: RAGService, query_text: str) -> None:
    # Arrange
    query = RetrievalQuery(query=query_text, top_k=3, filters={})

    # Act
    results = service.retrieve(query)

    # Assert
    assert results == []


def test_retrieved_source_metadata_and_content_match_the_original_chunks(
    service: RAGService, knowledge: Path,
) -> None:
    # Arrange
    chunker = MarkdownChunker()
    originals = {
        chunk.chunk_id: chunk
        for document in DocumentLoader(knowledge).load_documents()
        for chunk in chunker.chunk_document(document)
    }
    query = RetrievalQuery(query="retrieval", top_k=10, filters={})

    # Act
    results = service.retrieve(query)

    # Assert
    assert len(results) == 3
    for result in results:
        source = originals[result.chunk_id]
        assert result.source_title == source.source_title
        assert result.section == source.section
        assert result.content == source.content
        assert RetrievalResult.model_validate_json(result.model_dump_json()) == result
        assert set(result.model_dump()) == {"chunk_id", "source_title", "section", "content", "score"}


@pytest.mark.parametrize("context", ["title", "section"])
def test_body_evidence_outranks_repeated_heading_only_matches(tmp_path: Path, context: str) -> None:
    # Arrange
    heading = "# Quartz Quartz" if context == "title" else "# Reference\n\n## Quartz Quartz"
    (tmp_path / "context.md").write_text(
        heading + "\n\nThis page discusses unrelated maintenance procedures.", encoding="utf-8",
    )
    (tmp_path / "evidence.md").write_text(
        "# Timing\n\nThe quartz oscillator maintains precise timing in this device.", encoding="utf-8",
    )
    instance = RAGService(tmp_path)
    query = RetrievalQuery(query="quartz", top_k=1, filters={})
    try:
        # Act
        results = instance.retrieve(query)
        all_results = instance.retrieve(RetrievalQuery(query="quartz", top_k=10, filters={}))
        repeated = instance.retrieve(query)
        instance.index_documents()
        rebuilt = instance.retrieve(query)

        # Assert
        assert results[0].source_title == "Timing"
        assert "oscillator maintains precise timing" in results[0].content
        assert results == all_results[:1] == repeated == rebuilt
        assert len(all_results) == 2
        assert all(result.score > 0 for result in all_results)
        assert all_results == sorted(all_results, key=lambda result: (-result.score, result.chunk_id))
    finally:
        instance.close()


def test_ranking_preserves_context_only_matches_and_metadata_filters(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "context.md").write_text(
        "# Quartz\n\nGeneral maintenance procedures.", encoding="utf-8",
    )
    (tmp_path / "body.md").write_text(
        "# Timing\n\nA quartz oscillator supplies the clock signal.", encoding="utf-8",
    )
    instance = RAGService(tmp_path)
    try:
        # Act
        results = instance.retrieve(RetrievalQuery(
            query="quartz", top_k=1, filters={"file_name": "context.md"},
        ))

        # Assert
        assert len(results) == 1
        assert results[0].source_title == "Quartz"
        assert results[0].score > 0
        assert "General maintenance procedures." in results[0].content
    finally:
        instance.close()