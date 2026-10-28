"""Offline golden evaluations; run with python -m pytest tests/evals/test_rag_evals.py -v.

Every case uses top_k=3 and no filters; expectations never reach the service.
Concept checks match complete phrases after case and whitespace normalization,
without synonyms, stemming, or dropping missing concepts. Relevance is a lexical
check, not a semantic judge: each result must share a non-stopword query term
or contain a complete expected concept. Extractive grounding is checked against
the cited evidence, independently of the returned grounded flag.
"""

import json
import re
import socket
from collections import Counter
from pathlib import Path
from typing import Annotated, Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ConfigDict, Field

from agent.rag.contracts import DocumentChunk, RAGRequest, RAGResponse, RetrievalQuery, RetrievalResult
from agent.rag.service import RAGService, STOP_WORDS


DATASET = Path(__file__).with_name("rag-evals.json")
CASES = json.loads(DATASET.read_text(encoding="utf-8"))
TOP_K = 3
Text = Annotated[str, Field(min_length=1)]
NO_ANSWER = re.compile(
    r"\b(insufficient (?:context|evidence|knowledge|information)|"
    r"not enough (?:context|evidence|knowledge|information)|"
    r"does not contain enough information|"
    r"cannot (?:answer|provide|determine)|unable to (?:answer|provide|determine)|"
    r"not (?:documented|specified|provided|available) in (?:the|these) (?:knowledge|notes|documents))\b",
    re.IGNORECASE,
)


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: Text
    category: Text
    question: Text
    expected_sources: list[Text]
    expected_concepts: list[Text]
    forbidden_concepts: list[Text]
    should_retrieve: bool
    should_answer: bool
    expected_insufficient_context: bool


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Golden RAG evaluations must not access network or Azure services")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _contains(text: str, phrase: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(_normalize(phrase)) + r"(?!\w)", _normalize(text)) is not None


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[^\W_]+", text.casefold())) - STOP_WORDS


def _evaluate(raw_case: dict[str, Any], service: RAGService | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "evaluation_id": raw_case.get("id"), "category": raw_case.get("category"),
        "question": raw_case.get("question"),
        "expected_sources": raw_case.get("expected_sources", []), "actual_sources": [],
        "expected_concepts": raw_case.get("expected_concepts", []),
        "missing_concepts": list(raw_case.get("expected_concepts", [])),
        "forbidden_concepts_found": [], "retrieved_chunk_ids": [],
        "retrieval_scores": [], "returned_citations": [],
        "grounded_status": None, "insufficient_context_status": None,
        "likely_failure_layer": None, "failures": [], "top_k": TOP_K,
        "retrieval_executed": False, "actual_should_answer": None,
    }
    stage = "Query Processing"
    owned_service = service is None
    chunks: dict[str, DocumentChunk] = {}
    source_by_chunk: dict[str, str] = {}

    def check(condition: bool, layer: str, message: str) -> None:
        if not condition:
            report["failures"].append({"layer": layer, "message": message})
            if report["likely_failure_layer"] is None:
                report["likely_failure_layer"] = layer

    try:
        case = EvaluationCase.model_validate(raw_case)
        stage = "Ingestion"
        service = service if service is not None else RAGService()
        documents = service.loader.load_documents()
        source_by_document = {
            document.document_id: str(document.metadata["relative_path"])
            for document in documents
        }
        check(
            set(case.expected_sources) <= set(source_by_document.values()),
            "Ingestion", "An expected source was not ingested",
        )

        stage = "Chunking"
        for document in documents:
            for chunk in service.chunker.chunk_document(document):
                check(chunk.chunk_id not in chunks, "Chunking", "Duplicate chunk identifier")
                check(chunk.document_id == document.document_id, "Chunking", "Chunk lost document identity")
                chunks[chunk.chunk_id] = chunk
                source_by_chunk[chunk.chunk_id] = source_by_document[document.document_id]

        stage = "Indexing"
        count = service.index_documents()
        check(count == len(chunks), "Indexing", "Index chunk count differs from ingestion/chunking output")
        health = service.health()
        check(health.success and health.indexed, "Indexing", "Index is not available after indexing")
        check(health.indexed_chunks == len(chunks), "Indexing", "Index health count differs from expected chunks")

        stage = "Query Processing"
        request = RAGRequest(query=case.question, top_k=TOP_K, filters={})
        retrieved_calls: list[list[RetrievalResult]] = []
        retrieval_errors: list[str] = []
        original_retrieve = service.retrieve

        def observe(query: RetrievalQuery) -> list[RetrievalResult]:
            report["retrieval_executed"] = True
            check(query.query == case.question, "Query Processing", "Retrieval changed the question")
            check(query.top_k == TOP_K and query.filters == {}, "Query Processing", "Retrieval changed query configuration")
            try:
                results = original_retrieve(query)
                retrieved_calls.append([result.model_copy(deep=True) for result in results])
                return results
            except Exception as error:
                retrieval_errors.append(f"{type(error).__name__}: {error}")
                raise

        stage = "Answer Generation"
        with patch.object(service, "retrieve", side_effect=observe):
            raw_response = service.query(request)
        response = RAGResponse.model_validate(raw_response.model_dump())
        results = response.retrieved_chunks
        report.update(
            actual_sources=list(dict.fromkeys(source_by_chunk.get(result.chunk_id, "UNKNOWN_SOURCE") for result in results)),
            retrieved_chunk_ids=[result.chunk_id for result in results],
            retrieval_scores=[result.score for result in results],
            returned_citations=[citation.model_dump(mode="json") for citation in response.citations],
            grounded_status=response.grounded,
            insufficient_context_status=response.insufficient_context,
            answer=response.answer, success=response.success, error=response.error,
            retrieval_errors=retrieval_errors,
            missing_concepts=[concept for concept in case.expected_concepts if not _contains(response.answer, concept)],
            forbidden_concepts_found=[concept for concept in case.forbidden_concepts if _contains(response.answer, concept)],
        )
        check(report["retrieval_executed"], "Retrieval", "Retrieval was never executed")
        check(not retrieval_errors, "Retrieval", "Retrieval raised an exception")
        check(bool(retrieved_calls), "Retrieval", "Retrieval did not return successfully")
        if retrieved_calls:
            check(all(results == observed for observed in retrieved_calls), "Retrieval", "Response evidence differs from actual retrieval")
        check(response.query == case.question, "Query Processing", "Response changed the question")
        check(response.success and response.error is None, "Answer Generation", "RAG request did not complete successfully")
        check(bool(results) == case.should_retrieve, "Retrieval", "should_retrieve behavior mismatch")
        check(len(results) <= TOP_K, "Ranking", "Retrieved more than top_k chunks")
        check(len({result.chunk_id for result in results}) == len(results), "Retrieval", "Duplicate retrieved chunk")
        check(all(result.score > 0 for result in results), "Ranking", "Nonpositive relevance score")
        check(results == sorted(results, key=lambda result: (-result.score, result.chunk_id)), "Ranking", "Results are not in descending score / stable chunk-ID order")

        stage = "Retrieval"
        candidates = service.retrieve(RetrievalQuery(query=case.question, top_k=max(1, len(chunks)), filters={}))
        check(results == candidates[:TOP_K], "Ranking", "Returned results are not the leading top_k candidates")
        report["candidate_sources"] = list(dict.fromkeys(source_by_chunk.get(result.chunk_id, "UNKNOWN_SOURCE") for result in candidates))
        for source in case.expected_sources:
            if source not in report["actual_sources"]:
                layer = "Ranking" if source in report["candidate_sources"] else "Retrieval"
                if source not in source_by_document.values():
                    layer = "Ingestion"
                elif not any(name == source for name in source_by_chunk.values()):
                    layer = "Chunking"
                check(False, layer, f"Expected source not retrieved: {source}")

        query_terms = _terms(case.question)
        report["irrelevant_chunk_ids"] = []
        for result in results:
            original = chunks.get(result.chunk_id)
            check(original is not None, "Indexing", f"Unknown retrieved chunk: {result.chunk_id}")
            if original is not None:
                check(
                    (result.source_title, result.section, result.content)
                    == (original.source_title, original.section, original.content),
                    "Retrieval", f"Retrieved content or source metadata changed: {result.chunk_id}",
                )
            relevant = bool(query_terms & _terms(result.content)) or any(
                _contains(result.content, concept) for concept in case.expected_concepts
            )
            if not relevant:
                report["irrelevant_chunk_ids"].append(result.chunk_id)
            check(relevant, "Retrieval", f"Chunk lacks lexical relevance: {result.chunk_id}")

        for concept in report["missing_concepts"]:
            if any(_contains(result.content, concept) for result in results):
                layer = "Answer Generation"
            elif any(_contains(result.content, concept) for result in candidates):
                layer = "Ranking"
            elif any(_contains(chunk.content, concept) for chunk in chunks.values()):
                layer = "Retrieval" if query_terms else "Query Processing"
            elif any(_contains(document.content, concept) for document in documents):
                layer = "Chunking"
            else:
                layer = "Ingestion"
            check(False, layer, f"Expected concept missing from answer: {concept}")
        check(not report["forbidden_concepts_found"], "Grounding", "Answer contains forbidden concepts")

        by_id = {result.chunk_id: result for result in results}
        cited: list[RetrievalResult] = []
        citation_ids = [citation.chunk_id for citation in response.citations]
        check(len(citation_ids) == len(set(citation_ids)), "Citation", "Duplicate citations")
        for citation in response.citations:
            original = chunks.get(citation.chunk_id)
            result = by_id.get(citation.chunk_id)
            check(original is not None, "Citation", f"Citation names a nonexistent chunk: {citation.chunk_id}")
            check(result is not None, "Citation", f"Citation was not retrieved: {citation.chunk_id}")
            if original is not None:
                check((citation.source_title, citation.section) == (original.source_title, original.section), "Citation", f"Citation differs from original source: {citation.chunk_id}")
            if result is not None:
                check((citation.source_title, citation.section) == (result.source_title, result.section), "Citation", f"Citation differs from retrieved chunk: {citation.chunk_id}")
                cited.append(result)

        check(response.grounded == case.should_answer, "Grounding", "Grounded status differs from expected answer behavior")
        check(response.insufficient_context == case.expected_insufficient_context, "Grounding", "expected_insufficient_context behavior mismatch")
        if response.grounded:
            check(bool(cited), "Citation", "Grounded answer has no valid retrieved citation")
            paragraphs = [paragraph for paragraph in response.answer.split("\n\n") if paragraph.strip()]
            check(
                all(any(_normalize(paragraph) in _normalize(result.content) for result in cited) for paragraph in paragraphs),
                "Grounding", "Answer contains text unsupported by its cited retrieved chunks",
            )
        if not results:
            check(not response.citations, "Citation", "No-evidence response contains citations")
        no_answer = bool(NO_ANSWER.search(_normalize(response.answer)))
        actual_should_answer = response.grounded or not (no_answer and response.insufficient_context)
        report["actual_should_answer"] = actual_should_answer
        check(actual_should_answer == case.should_answer, "Answer Generation", "should_answer behavior mismatch")
        if not case.should_answer:
            check(no_answer, "Answer Generation", "Expected an explicit insufficient-evidence acknowledgment")
    except Exception as error:
        report["evaluation_error"] = f"{type(error).__name__}: {error}"
        check(False, stage, report["evaluation_error"])
    finally:
        if owned_service and service is not None:
            try:
                service.close()
            except Exception as error:
                check(False, "Indexing", f"Index cleanup failed: {type(error).__name__}: {error}")
    return report


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_rag_evaluation(case: dict[str, Any]) -> None:
    # Arrange / Act
    report = _evaluate(case)

    # Assert
    if report["failures"]:
        pytest.fail(json.dumps(report, indent=2, ensure_ascii=True), pytrace=False)


def test_golden_dataset_structure() -> None:
    assert isinstance(CASES, list) and CASES, "Golden dataset must be a nonempty array"
    cases = [EvaluationCase.model_validate(case) for case in CASES]
    duplicates = [identifier for identifier, count in Counter(case.id for case in cases).items() if count > 1]
    assert not duplicates, f"Duplicate evaluation IDs: {duplicates}"