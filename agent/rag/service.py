import logging
import re
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from threading import RLock
from typing import Protocol

from agent.rag.chunker import MarkdownChunker
from agent.rag.contracts import (
    DocumentChunk, GroundedAnswer, RAGHealthResponse, RAGRequest, RAGResponse, RetrievalQuery,
    RetrievalResult, SourceCitation,
)
from agent.rag.document_loader import DEFAULT_KNOWLEDGE_DIRECTORY, DocumentLoader
from agent.rag.tracing import RAGTracer, capture_candidates


logger = logging.getLogger(__name__)
NO_CONTEXT_ANSWER = "The knowledge base does not contain enough information to answer this query."
STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do", "does",
    "for", "from", "how", "i", "in", "is", "it", "me", "of", "on", "or", "please",
    "that", "the", "their", "these", "this", "to", "was", "what", "when", "where",
    "which", "who", "why", "will", "with", "would", "you", "your",
})


class CitationValidationError(ValueError):
    pass


class AnswerGenerator(Protocol):
    def generate(self, query: str, retrieved_chunks: Sequence[RetrievalResult]) -> GroundedAnswer:
        ...


class ExtractiveAnswerGenerator:
    def generate(self, query: str, retrieved_chunks: Sequence[RetrievalResult]) -> GroundedAnswer:
        return GroundedAnswer(
            answer="\n\n".join(chunk.content for chunk in retrieved_chunks),
            citations=[SourceCitation(
                source_title=chunk.source_title, section=chunk.section, chunk_id=chunk.chunk_id,
            ) for chunk in retrieved_chunks],
            grounded=True, insufficient_context=False,
        )


class RAGService:
    """Local lexical RAG with atomic in-memory indexing and extractive validation.

    Filters exactly match chunk identity fields, chunk metadata, or original
    document metadata. Scores are positive BM25 ranking magnitudes, not confidence.
    Body evidence has greater ranking weight than repeated title/heading context;
    body text is indexed once, with ancestor headings in a separate context field.
    The complete original content is stored and returned unchanged.
    Injected generators must currently return source-verifiable excerpts; arbitrary
    paraphrases require a future grounding verifier, not just grounded=True.
    """

    def __init__(
        self, knowledge_directory: str | Path = DEFAULT_KNOWLEDGE_DIRECTORY, *,
        loader: DocumentLoader | None = None, chunker: MarkdownChunker | None = None,
        answer_generator: AnswerGenerator | None = None,
        tracer: RAGTracer | None = None, retrieval_strategy: str | None = None,
    ) -> None:
        self.loader = loader if loader is not None else DocumentLoader(knowledge_directory)
        self.chunker = chunker if chunker is not None else MarkdownChunker()
        self.answer_generator = answer_generator if answer_generator is not None else ExtractiveAnswerGenerator()
        self.tracer = tracer if tracer is not None else RAGTracer()
        self.retrieval_strategy = retrieval_strategy if retrieval_strategy is not None else (
            "local_lexical" if type(self).retrieve is RAGService.retrieve else type(self).__name__
        )
        self._index: sqlite3.Connection | None = None
        self._chunks: dict[str, DocumentChunk] = {}
        self._lock = RLock()

    def index_documents(self) -> int:
        """Load and chunk trusted documents, replacing the index only on success."""
        with self._lock:
            return self._index_documents()

    def _index_documents(self) -> int:
        chunks: dict[str, DocumentChunk] = {}
        for document in self.loader.load_documents():
            for chunk in self.chunker.chunk_document(document):
                if chunk.chunk_id in chunks and chunks[chunk.chunk_id] != chunk:
                    raise ValueError("Conflicting duplicate chunk identifier")
                chunks[chunk.chunk_id] = chunk.model_copy(deep=True)
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        try:
            connection.execute(
                "CREATE VIRTUAL TABLE chunks USING fts5("
                                "chunk_id UNINDEXED, source_title, section, content UNINDEXED, body, "
                                "heading_context, tokenize='unicode61')"
            )
            connection.executemany(
                                "INSERT INTO chunks (chunk_id, source_title, section, content, body, heading_context) "
                                "VALUES (?, ?, ?, ?, ?, ?)",
                [(chunk.chunk_id, chunk.source_title, chunk.section, chunk.content,
                                    chunk.content.partition("\n\n")[2] if chunk.content.startswith("# ") else chunk.content,
                                    chunk.content.partition("\n\n")[0] if chunk.content.startswith("# ") else "")
                 for chunk in sorted(chunks.values(), key=lambda item: item.chunk_id)],
            )
            connection.commit()
        except Exception:
            connection.close()
            raise
        previous = self._index
        self._index, self._chunks = connection, chunks
        if previous is not None:
            previous.close()
        return len(chunks)

    @staticmethod
    def _matches_filters(chunk: DocumentChunk, query: RetrievalQuery) -> bool:
        document_metadata = chunk.metadata.get("document_metadata")
        values = dict(document_metadata) if isinstance(document_metadata, dict) else {}
        values.update(chunk.metadata)
        values.update({
            "document_id": chunk.document_id, "chunk_id": chunk.chunk_id,
            "source_title": chunk.source_title, "section": chunk.section,
        })
        return all(key in values and values[key] == value for key, value in query.filters.items())

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """Retrieve positive lexical matches; no relevant terms means no results."""
        with self._lock:
            return self._retrieve(query)

    def _retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        query = RetrievalQuery.model_validate(query.model_dump())
        if self._index is None:
            self.index_documents()
        terms = sorted(set(re.findall(r"[^\W_]+", query.query.casefold())) - STOP_WORDS)
        if not terms:
            capture_candidates([])
            return []
        if self._index is None:
            raise RuntimeError("Index initialization failed")
        expression = " OR ".join(f'"{term}"' for term in terms)
        rows = self._index.execute(
            "SELECT chunk_id, bm25(chunks, 0.0, 0.25, 0.25, 0.25, 2.0, 0.25) AS rank "
            "FROM chunks WHERE chunks MATCH ? ORDER BY rank, chunk_id", (expression,),
        )
        results: list[RetrievalResult] = []
        for chunk_id, rank in rows:
            chunk = self._chunks[chunk_id]
            if rank < 0 and self._matches_filters(chunk, query):
                results.append(RetrievalResult(
                    chunk_id=chunk.chunk_id, source_title=chunk.source_title,
                    section=chunk.section, content=chunk.content, score=-float(rank),
                ))
        capture_candidates(results)
        return results[:query.top_k]

    def generate_answer(self, query: str, retrieved_chunks: Sequence[RetrievalResult]) -> GroundedAnswer:
        """Generate only from supplied evidence and reject unsupported citations/text."""
        evidence = [RetrievalResult.model_validate(chunk.model_dump()) for chunk in retrieved_chunks]
        if not evidence:
            return GroundedAnswer(
                answer=NO_CONTEXT_ANSWER, citations=[], grounded=False, insufficient_context=True,
            )
        candidate = self.answer_generator.generate(query, tuple(chunk.model_copy(deep=True) for chunk in evidence))
        answer = GroundedAnswer.model_validate(candidate.model_dump())
        if answer.insufficient_context:
            if answer.grounded or answer.citations or answer.answer != NO_CONTEXT_ANSWER:
                raise ValueError("An abstention requires the canonical no-answer text, no citations and grounded=False")
            return answer
        by_id = {chunk.chunk_id: chunk for chunk in evidence}
        cited: list[RetrievalResult] = []
        for citation in answer.citations:
            chunk = by_id.get(citation.chunk_id)
            if chunk is None or (citation.source_title, citation.section) != (chunk.source_title, chunk.section):
                raise CitationValidationError("Citation does not identify a retrieved source")
            if any(previous.chunk_id == chunk.chunk_id for previous in cited):
                raise CitationValidationError("Duplicate citation")
            cited.append(chunk)
        if not cited or not answer.grounded or answer.insufficient_context:
            raise ValueError("An extractive answer requires citations and grounded evidence")
        paragraphs = [paragraph for paragraph in answer.answer.split("\n\n") if paragraph.strip()]
        if not all(any(paragraph in chunk.content for chunk in cited) for paragraph in paragraphs):
            raise ValueError("Answer contains text outside the cited retrieved sources")
        return answer

    def query(self, request: RAGRequest, *, correlation_id: str | None = None) -> RAGResponse:
        """Orchestrate retrieval and generation, returning validated structured output."""
        with self.tracer.workflow(request, retrieval_strategy=self.retrieval_strategy, correlation_id=correlation_id) as trace:
            request = RAGRequest.model_validate(request.model_dump())
            retrieved: list[RetrievalResult] = []
            try:
                trace.failure_layer = "Retrieval"
                retrieved = self.retrieve(RetrievalQuery.model_validate(request.model_dump()))
                if trace.candidate_scope == "unavailable":
                    capture_candidates(retrieved, scope="retrieval_output")
                trace.final = [(result.chunk_id, result.score) for result in retrieved]
                trace.failure_layer = "Grounding"
                answer = self.generate_answer(request.query, retrieved)
                response = RAGResponse(
                    success=True, query=request.query, answer=answer.answer,
                    retrieved_chunks=retrieved, citations=answer.citations,
                    grounded=answer.grounded, insufficient_context=answer.insufficient_context,
                    error=None,
                )
            except Exception as error:
                if isinstance(error, CitationValidationError):
                    trace.failure_layer = "Citation"
                logger.error("RAG request failed (%s)", type(error).__name__)
                response = RAGResponse(
                    success=False, query=request.query, answer="The RAG request could not be completed.",
                    retrieved_chunks=retrieved, citations=[], grounded=False,
                    insufficient_context=not retrieved, error="RAG_PROCESSING_ERROR",
                )
            trace.finish(response)
            return response

    def health(self) -> RAGHealthResponse:
        with self._lock:
            if self._index is not None:
                self._index.execute("SELECT 1").fetchone()
            return RAGHealthResponse(
                success=True, indexed=self._index is not None,
                indexed_chunks=len(self._chunks), error=None,
            )

    def close(self) -> None:
        with self._lock:
            if self._index is not None:
                self._index.close()
                self._index = None
            self._chunks = {}