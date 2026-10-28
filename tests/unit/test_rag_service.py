import socket
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from agent.rag.contracts import GroundedAnswer, RAGRequest, RAGResponse, RetrievalQuery, RetrievalResult, SourceCitation
from agent.rag.service import NO_CONTEXT_ANSWER, RAGService


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Deterministic RAG must not access the network")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.fixture
def knowledge(tmp_path: Path) -> Path:
    directory = tmp_path / "knowledge"
    directory.mkdir()
    (directory / "rules.md").write_text(
        "# Planner\n\n## Persistence\n\nProposals are not persisted until accepted.\n\n"
        "## Conflicts\n\nConflicting task writes are rejected.", encoding="utf-8",
    )
    (directory / "sources.md").write_text(
        "# Sources\n\n## Attribution\n\nEvery factual answer needs source attribution.", encoding="utf-8",
    )
    return directory


@pytest.fixture
def service(knowledge: Path) -> Iterator[RAGService]:
    instance = RAGService(knowledge)
    yield instance
    instance.close()


def test_query_orchestrates_loading_chunking_indexing_and_extractive_answer(service: RAGService) -> None:
    # Arrange
    request = RAGRequest(query="Are proposals persisted?", top_k=2, filters={})

    # Act
    response = service.query(request)

    # Assert
    assert response.success is True
    assert response.query == request.query
    assert response.error is None
    assert response.grounded is True
    assert response.insufficient_context is False
    assert response.retrieved_chunks
    assert "Proposals are not persisted until accepted." in response.answer
    assert response.answer == "\n\n".join(chunk.content for chunk in response.retrieved_chunks)
    assert response.citations == [SourceCitation(
        source_title=chunk.source_title, section=chunk.section, chunk_id=chunk.chunk_id,
    ) for chunk in response.retrieved_chunks]
    assert RAGResponse.model_validate_json(response.model_dump_json()) == response


@pytest.mark.parametrize("query", ["quasar astrophysics", "what is the", "!!!"])
def test_no_relevant_chunks_produces_honest_no_answer(service: RAGService, query: str) -> None:
    response = service.query(RAGRequest(query=query, top_k=3, filters={}))

    assert response.success is True
    assert response.retrieved_chunks == []
    assert response.citations == []
    assert response.insufficient_context is True
    assert response.grounded is False
    assert "does not contain enough information" in response.answer


def test_empty_index_returns_insufficient_context(tmp_path: Path) -> None:
    service = RAGService(tmp_path)
    try:
        response = service.query(RAGRequest(query="proposals", top_k=3, filters={}))
        assert response.success is True
        assert response.insufficient_context is True
        assert response.grounded is False
        assert response.citations == []
    finally:
        service.close()


def test_retrieval_is_repeatable_ranked_and_limited(service: RAGService) -> None:
    query = RetrievalQuery(query="proposals source", top_k=1, filters={})

    first = service.retrieve(query)
    repeated = service.retrieve(query)
    service.index_documents()
    rebuilt = service.retrieve(query)

    assert len(first) == 1
    assert first[0].score > 0
    assert first == repeated == rebuilt


@pytest.mark.parametrize(("filters", "expected_sections"), [
    ({"section": "Persistence"}, ["Persistence"]),
    ({"source_title": "Sources"}, ["Attribution"]),
    ({"file_name": "sources.md"}, ["Attribution"]),
    ({"section": "Missing"}, []),
    ({"unknown": None}, []),
])
def test_filters_are_applied_before_top_k(service: RAGService, filters: dict[str, object], expected_sections: list[str]) -> None:
    query = RetrievalQuery.model_validate({"query": "proposals source", "top_k": 1, "filters": filters})

    results = service.retrieve(query)

    assert [chunk.section for chunk in results] == expected_sections


def test_retrieval_treats_search_syntax_as_data(service: RAGService) -> None:
    service.retrieve(RetrievalQuery(query='" OR *; DROP TABLE chunks; --', top_k=3, filters={}))

    results = service.retrieve(RetrievalQuery(query="proposals", top_k=1, filters={}))

    assert results[0].section == "Persistence"


def test_reindex_removes_deleted_documents_and_stale_sources(service: RAGService, knowledge: Path) -> None:
    assert service.index_documents() == 3
    (knowledge / "sources.md").unlink()

    assert service.index_documents() == 2
    results = service.retrieve(RetrievalQuery(query="attribution", top_k=3, filters={}))

    assert results == []


def test_failed_reindex_does_not_destroy_last_successful_index(service: RAGService, knowledge: Path) -> None:
    query = RetrievalQuery(query="proposals", top_k=1, filters={})
    before = service.retrieve(query)
    (knowledge / "invalid.txt").write_bytes(b"\xff")

    with pytest.raises(UnicodeDecodeError):
        service.index_documents()

    assert service.retrieve(query) == before


class SelectingGenerator:
    def generate(self, query: str, retrieved_chunks: Sequence[RetrievalResult]) -> GroundedAnswer:
        chunk = retrieved_chunks[0]
        return GroundedAnswer(
            answer="Proposals are not persisted until accepted.",
            citations=[SourceCitation(source_title=chunk.source_title, section=chunk.section, chunk_id=chunk.chunk_id)],
            grounded=True, insufficient_context=False,
        )


def test_generator_can_be_injected_and_use_only_retrieved_excerpts(knowledge: Path) -> None:
    service = RAGService(knowledge, answer_generator=SelectingGenerator())
    try:
        response = service.query(RAGRequest(query="proposals", top_k=1, filters={}))
        assert response.success is True
        assert response.answer == "Proposals are not persisted until accepted."
        assert response.citations[0].chunk_id == response.retrieved_chunks[0].chunk_id
    finally:
        service.close()


@pytest.mark.parametrize("violation", ["citation_id", "citation_title", "citation_section", "unsupported_text", "no_citation", "duplicate", "flags"])
def test_generator_cannot_fabricate_sources_or_unsupported_claims(knowledge: Path, violation: str) -> None:
    class InvalidGenerator:
        def generate(self, query: str, retrieved_chunks: Sequence[RetrievalResult]) -> GroundedAnswer:
            answer = SelectingGenerator().generate(query, retrieved_chunks)
            if violation == "citation_id":
                answer.citations[0].chunk_id = "not-retrieved"
            elif violation == "citation_title":
                answer.citations[0].source_title = "Fabricated source"
            elif violation == "citation_section":
                answer.citations[0].section = "Fabricated section"
            elif violation == "unsupported_text":
                answer.answer = "Appointments are automatically published to an external calendar."
            elif violation == "no_citation":
                answer.citations = []
            elif violation == "duplicate":
                answer.citations.append(answer.citations[0])
            else:
                answer.grounded = False
            return answer

    service = RAGService(knowledge, answer_generator=InvalidGenerator())
    try:
        response = service.query(RAGRequest(query="proposals", top_k=1, filters={}))
        assert response.success is False
        assert response.error == "RAG_PROCESSING_ERROR"
        assert response.grounded is False
        assert response.citations == []
        assert "automatically published" not in response.answer
    finally:
        service.close()


def test_generator_cannot_cite_an_existing_but_unretrieved_source(
    service: RAGService, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    other_source = service.retrieve(RetrievalQuery(query="attribution", top_k=1, filters={}))[0]
    request = RAGRequest(query="proposals", top_k=1, filters={})

    class UnretrievedSourceGenerator:
        def generate(self, query: str, retrieved_chunks: Sequence[RetrievalResult]) -> GroundedAnswer:
            return GroundedAnswer(
                answer=other_source.content,
                citations=[SourceCitation(
                    source_title=other_source.source_title,
                    section=other_source.section, chunk_id=other_source.chunk_id,
                )], grounded=True, insufficient_context=False,
            )

    monkeypatch.setattr(service, "answer_generator", UnretrievedSourceGenerator())

    # Act
    response = service.query(request)

    # Assert
    assert len(response.retrieved_chunks) == 1
    assert other_source.chunk_id not in {chunk.chunk_id for chunk in response.retrieved_chunks}
    assert response.success is False
    assert response.grounded is False
    assert response.error == "RAG_PROCESSING_ERROR"
    assert response.citations == []
    assert response.answer != other_source.content
    assert RAGResponse.model_validate_json(response.model_dump_json()) == response


def test_generator_mutation_cannot_change_retrieved_evidence(knowledge: Path) -> None:
    class MutatingGenerator:
        def generate(self, query: str, retrieved_chunks: Sequence[RetrievalResult]) -> GroundedAnswer:
            retrieved_chunks[0].content = "Invented content."
            return GroundedAnswer(
                answer="Invented content.", citations=[SourceCitation(
                    source_title=retrieved_chunks[0].source_title,
                    section=retrieved_chunks[0].section, chunk_id=retrieved_chunks[0].chunk_id,
                )], grounded=True, insufficient_context=False,
            )

    service = RAGService(knowledge, answer_generator=MutatingGenerator())
    try:
        response = service.query(RAGRequest(query="proposals", top_k=1, filters={}))
        assert response.success is False
        assert "Invented" not in response.retrieved_chunks[0].content
    finally:
        service.close()


def test_no_context_never_calls_generator(knowledge: Path) -> None:
    class ForbiddenGenerator:
        def generate(self, query: str, retrieved_chunks: Sequence[RetrievalResult]) -> GroundedAnswer:
            raise AssertionError("No-evidence requests must not invoke generation")

    service = RAGService(knowledge, answer_generator=ForbiddenGenerator())
    try:
        response = service.query(RAGRequest(query="quasar", top_k=1, filters={}))
        assert response.success is True
        assert response.insufficient_context is True
        assert response.citations == []
    finally:
        service.close()


def test_generator_can_abstain_when_retrieved_background_is_insufficient(knowledge: Path) -> None:
    class AbstainingGenerator:
        def generate(self, query: str, retrieved_chunks: Sequence[RetrievalResult]) -> GroundedAnswer:
            assert retrieved_chunks
            return GroundedAnswer(
                answer=NO_CONTEXT_ANSWER, citations=[], grounded=False, insufficient_context=True,
            )

    service = RAGService(knowledge, answer_generator=AbstainingGenerator())
    try:
        response = service.query(RAGRequest(
            query="What is the actual persisted proposal identifier?", top_k=2, filters={},
        ))
        assert response.retrieved_chunks
        assert response.success is True
        assert response.error is None
        assert response.answer == NO_CONTEXT_ANSWER
        assert response.citations == []
        assert response.grounded is False
        assert response.insufficient_context is True
    finally:
        service.close()


@pytest.mark.parametrize("violation", ["unsupported_text", "citation", "grounded"])
def test_abstention_does_not_bypass_grounding_validation(knowledge: Path, violation: str) -> None:
    class InvalidAbstainingGenerator:
        def generate(self, query: str, retrieved_chunks: Sequence[RetrievalResult]) -> GroundedAnswer:
            answer = GroundedAnswer(
                answer=NO_CONTEXT_ANSWER, citations=[], grounded=False, insufficient_context=True,
            )
            if violation == "unsupported_text":
                answer.answer = "The deployed endpoint is https://invented.invalid."
            elif violation == "citation":
                chunk = retrieved_chunks[0]
                answer.citations = [SourceCitation(
                    source_title=chunk.source_title, section=chunk.section, chunk_id=chunk.chunk_id,
                )]
            else:
                answer.grounded = True
            return answer

    service = RAGService(knowledge, answer_generator=InvalidAbstainingGenerator())
    try:
        response = service.query(RAGRequest(query="proposals", top_k=1, filters={}))
        assert response.success is False
        assert response.error == "RAG_PROCESSING_ERROR"
        assert response.grounded is False
        assert response.citations == []
        assert "invented.invalid" not in response.answer
    finally:
        service.close()


@pytest.mark.parametrize("warning", [
    "## Common Mistakes\n\n- Assuming contract classes implement natural-language CRUD decisions.",
    "## Negative Example\n\nDo not claim that contract classes implement natural-language CRUD decisions.",
])
def test_extractive_grounding_preserves_quoted_warnings_and_negative_examples(
    service: RAGService, warning: str,
) -> None:
    chunk = RetrievalResult(
        chunk_id="warning", source_title="Contract guidance", section="Warnings",
        content=warning, score=1.0,
    )

    answer = service.generate_answer("Do contract classes prove a CRUD workflow exists?", [chunk])

    assert answer.answer == warning
    assert answer.grounded is True
    assert answer.insufficient_context is False
    assert answer.citations == [SourceCitation(
        chunk_id=chunk.chunk_id, source_title=chunk.source_title, section=chunk.section,
    )]


def test_document_errors_produce_validated_failures_without_raw_error_details(service: RAGService, knowledge: Path) -> None:
    (knowledge / "invalid.txt").write_bytes(b"\xff")

    response = service.query(RAGRequest(query="proposals", top_k=1, filters={}))

    assert response.success is False
    assert response.error == "RAG_PROCESSING_ERROR"
    assert response.grounded is False
    assert response.citations == []
    assert "invalid.txt" not in response.answer
    assert RAGResponse.model_validate_json(response.model_dump_json()) == response
    assert set(response.model_dump()) == {
        "success", "query", "answer", "retrieved_chunks", "citations",
        "grounded", "insufficient_context", "error",
    }