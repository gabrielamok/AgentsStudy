import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from threading import Barrier

import pytest

from agent.rag.contracts import GroundedAnswer, RAGRequest, RetrievalResult, SourceCitation
from agent.rag.service import ExtractiveAnswerGenerator, RAGService
from agent.rag.tracing import RAGTraceRecord, RAGTracer, TraceRedactor


class CollectingSink:
    def __init__(self) -> None:
        self.records: list[RAGTraceRecord] = []

    def emit(self, record: RAGTraceRecord) -> None:
        self.records.append(record)


@pytest.fixture
def knowledge(tmp_path: Path) -> Path:
    (tmp_path / "rules.md").write_text(
        "# Rules\n\n## Proposals\n\nProposals are not persisted.\n\n"
        "## Validation\n\nProposals require validation.\n", encoding="utf-8",
    )
    return tmp_path


def request(query: str = "  ARE   Proposals persisted?  ", top_k: int = 1) -> RAGRequest:
    return RAGRequest(query=query, top_k=top_k, filters={})


def test_successful_trace_captures_full_candidates_final_ranking_and_citations(knowledge: Path) -> None:
    sink = CollectingSink()
    service = RAGService(knowledge, tracer=RAGTracer(sink, correlation_id_factory=lambda: "fixed-id"))
    try:
        response = service.query(request())
        trace = sink.records[0]
        assert response.success and len(sink.records) == 1
        assert trace.correlation_id == "fixed-id"
        assert trace.normalized_query == "are proposals persisted?"
        assert trace.retrieval_strategy == "local_lexical"
        assert trace.requested_top_k == 1
        assert len(trace.candidate_chunk_ids) == 2
        assert trace.ranking_order == trace.candidate_chunk_ids
        assert trace.candidate_scores == tuple(sorted(trace.candidate_scores, reverse=True))
        assert trace.final_chunk_ids == tuple(result.chunk_id for result in response.retrieved_chunks)
        assert trace.final_scores == tuple(result.score for result in response.retrieved_chunks)
        assert trace.final_chunk_ids == trace.candidate_chunk_ids[:1]
        assert trace.citation_chunk_ids == tuple(citation.chunk_id for citation in response.citations)
        assert trace.grounded is True and trace.insufficient_context is False
        assert trace.processing_outcome == "success" and trace.failure_layer is None
        assert trace.candidate_scope == "filtered_candidates"
        assert "answer" not in asdict(trace) and "content" not in asdict(trace)
    finally:
        service.close()


def test_insufficient_context_trace_and_supplied_correlation_id(knowledge: Path) -> None:
    sink = CollectingSink()
    service = RAGService(knowledge, tracer=RAGTracer(sink, correlation_id_factory=lambda: "unused"))
    try:
        response = service.query(request("quasar"), correlation_id="caller-123")
        trace = sink.records[0]
        assert response.success and response.insufficient_context
        assert trace.correlation_id == "caller-123"
        assert trace.candidate_chunk_ids == trace.final_chunk_ids == trace.citation_chunk_ids == ()
        assert trace.grounded is False and trace.insufficient_context is True
        assert trace.processing_outcome == "insufficient_context" and trace.failure_layer is None
    finally:
        service.close()


def test_retrieval_failure_trace_never_records_exception_message(knowledge: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sink = CollectingSink()
    service = RAGService(knowledge, tracer=RAGTracer(sink, correlation_id_factory=lambda: "failed-request"))

    def fail(query: object) -> list[RetrievalResult]:
        raise RuntimeError("credential=fictional-private-value at C:\\private\\knowledge.md")

    monkeypatch.setattr(service, "retrieve", fail)
    try:
        response = service.query(request())
        trace = sink.records[0]
        assert not response.success
        assert trace.processing_outcome == "failure" and trace.failure_layer == "Retrieval"
        assert trace.final_chunk_ids == () and trace.citation_chunk_ids == ()
        assert "fictional-private-value" not in json.dumps(asdict(trace))
        assert "private" not in response.answer and "Traceback" not in response.model_dump_json()
    finally:
        service.close()


def test_invalid_citation_is_classified_without_logging_untrusted_citation(knowledge: Path) -> None:
    class InvalidGenerator:
        def generate(self, query: str, results: object) -> GroundedAnswer:
            return GroundedAnswer(
                answer="invented", citations=[SourceCitation(source_title="invented", section="invented", chunk_id="not-retrieved")],
                grounded=True, insufficient_context=False,
            )

    sink = CollectingSink()
    service = RAGService(knowledge, answer_generator=InvalidGenerator(), tracer=RAGTracer(sink))
    try:
        assert not service.query(request()).success
        trace = sink.records[0]
        assert trace.failure_layer == "Citation"
        assert trace.final_chunk_ids and trace.citation_chunk_ids == ()
        assert "not-retrieved" not in json.dumps(asdict(trace))
    finally:
        service.close()


@pytest.mark.parametrize("text", [
    "api_key=fictional-secret", "password: fictional-secret", "Bearer fictional-secret",
    "api key is fictional-secret", "CUSTOM_ENV=fictional-secret", "https://user:fictional-secret@example.invalid",
    '{"api_key": "fictional-secret"}', "{'CUSTOM_ENV': 'fictional-secret'}",
])
def test_secret_redaction(text: str) -> None:
    assert "fictional-secret" not in TraceRedactor().redact(text, query=True)


@pytest.mark.parametrize("path", [r"C:\private\knowledge\file.md", "/srv/private/knowledge/file.md", r"\\fictional-host\share\file.md", "/private.txt"])
def test_absolute_path_redaction(path: str) -> None:
    assert path not in TraceRedactor().redact(f"Explain {path}")


def test_registered_secret_is_redacted_without_reading_environment(knowledge: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "fictionalShortSecret"
    monkeypatch.setenv("FICTIONAL_TRACE_SECRET", secret)
    sink = CollectingSink()
    tracer = RAGTracer(sink, redactor=TraceRedactor([secret]), correlation_id_factory=lambda: "fixed")
    service = RAGService(knowledge, tracer=tracer)
    try:
        service.query(request(f"proposals {secret} C:\\private\\rules.md"))
        serialized = json.dumps(asdict(sink.records[0]))
        assert secret.casefold() not in serialized.casefold()
        assert "private" not in serialized
        assert "FICTIONAL_TRACE_SECRET" not in serialized
    finally:
        service.close()


def test_sink_failure_does_not_change_rag_response(knowledge: Path) -> None:
    class BrokenSink:
        def emit(self, record: RAGTraceRecord) -> None:
            raise RuntimeError("fictional secret in sink failure")

    service = RAGService(knowledge, tracer=RAGTracer(BrokenSink()))
    try:
        assert service.query(request()).success
    finally:
        service.close()


def test_context_does_not_leak_between_requests(knowledge: Path) -> None:
    sink = CollectingSink()
    identifiers = iter(["first", "second"])
    service = RAGService(knowledge, tracer=RAGTracer(sink, correlation_id_factory=lambda: next(identifiers)))
    try:
        service.query(request())
        service.query(request("quasar"))
        assert [record.correlation_id for record in sink.records] == ["first", "second"]
        assert sink.records[0].candidate_chunk_ids and sink.records[1].candidate_chunk_ids == ()
    finally:
        service.close()


def test_identical_requests_produce_identical_records_with_injected_id(knowledge: Path) -> None:
    sink = CollectingSink()
    service = RAGService(knowledge, tracer=RAGTracer(sink, correlation_id_factory=lambda: "fixed"))
    try:
        service.query(request())
        service.query(request())
        assert sink.records[0] == sink.records[1]
    finally:
        service.close()


def test_default_local_logger_emits_only_redacted_structured_record(knowledge: Path, caplog: pytest.LogCaptureFixture) -> None:
    service = RAGService(knowledge, tracer=RAGTracer(correlation_id_factory=lambda: "local-log"))
    try:
        with caplog.at_level("INFO", logger="agent.rag.tracing"):
            service.query(request("proposals API_KEY=fictional-private-value C:\\private\\knowledge.md"))
        records = [record for record in caplog.records if record.name == "agent.rag.tracing"]
        assert len(records) == 1 and records[0].getMessage() == "rag.trace"
        serialized = json.dumps(records[0].__dict__["rag_trace"])
        assert "fictional-private-value" not in serialized and "private" not in serialized
        assert records[0].__dict__["rag_trace"]["correlation_id"] == "local-log"
        assert records[0].exc_info is None
    finally:
        service.close()


def test_concurrent_traces_keep_correlation_ids_and_candidates_isolated(knowledge: Path) -> None:
    barrier = Barrier(2)

    class ConcurrentGenerator(ExtractiveAnswerGenerator):
        def generate(self, query, retrieved_chunks):
            barrier.wait(timeout=5)
            return super().generate(query, retrieved_chunks)

    sink = CollectingSink()
    service = RAGService(knowledge, tracer=RAGTracer(sink), answer_generator=ConcurrentGenerator())
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(service.query, request("persisted"), correlation_id="first")
            second = executor.submit(service.query, request("validation"), correlation_id="second")
            first_response, second_response = first.result(), second.result()
        traces = {record.correlation_id: record for record in sink.records}
        assert first_response.success and second_response.success and len(traces) == 2
        assert traces["first"].normalized_query == "persisted"
        assert traces["second"].normalized_query == "validation"
        assert traces["first"].final_chunk_ids == tuple(chunk.chunk_id for chunk in first_response.retrieved_chunks)
        assert traces["second"].final_chunk_ids == tuple(chunk.chunk_id for chunk in second_response.retrieved_chunks)
        assert traces["first"].candidate_chunk_ids != traces["second"].candidate_chunk_ids
    finally:
        service.close()


def test_custom_retriever_does_not_claim_unobserved_candidate_pool(knowledge: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sink = CollectingSink()
    service = RAGService(knowledge, tracer=RAGTracer(sink), retrieval_strategy="injected_fixture")
    result = RetrievalResult(chunk_id="fixture", source_title="fixture", section="rules", content="A supported fact.", score=0.75)
    monkeypatch.setattr(service, "retrieve", lambda query: [result])
    try:
        assert service.query(request()).success
        trace = sink.records[0]
        assert trace.retrieval_strategy == "injected_fixture"
        assert trace.candidate_scope == "retrieval_output"
        assert trace.candidate_scores == trace.final_scores == (0.75,)
        assert trace.candidate_chunk_ids == trace.final_chunk_ids == ("fixture",)
    finally:
        service.close()