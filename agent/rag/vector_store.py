import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from threading import RLock
from typing import Protocol

from pydantic import JsonValue

from agent.rag.contracts import DocumentChunk, RetrievalResult
from agent.rag.embeddings import normalize_embedding, validate_dimensions


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk: DocumentChunk
    embedding: Sequence[float]


def validate_min_relevance(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Minimum relevance must be a finite cosine score in [-1, 1]")
    try:
        score = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError("Minimum relevance must be a finite cosine score in [-1, 1]") from error
    if not math.isfinite(score) or not -1.0 <= score <= 1.0:
        raise ValueError("Minimum relevance must be a finite cosine score in [-1, 1]")
    return score


class VectorStore(Protocol):
    """Cosine store interface; implementations must preserve inclusive cutoffs and ties."""

    @property
    def dimensions(self) -> int:
        ...

    def replace(self, entries: Sequence[EmbeddedChunk]) -> int:
        """Atomically replace all chunks, rejecting malformed or duplicate entries."""
        ...

    def search(
        self, embedding: Sequence[float], *, top_k: int, min_relevance: float,
        filters: Mapping[str, JsonValue] | None = None,
    ) -> list[RetrievalResult]:
        """Filter before limiting; order by descending cosine score, then chunk ID."""
        ...


class InMemoryVectorStore:
    """Exact local cosine search. Scores are similarities, not confidence values."""

    def __init__(self, dimensions: int) -> None:
        self._dimensions = validate_dimensions(dimensions)
        self._entries: tuple[EmbeddedChunk, ...] = ()
        self._lock = RLock()

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def replace(self, entries: Sequence[EmbeddedChunk]) -> int:
        replacement: list[EmbeddedChunk] = []
        identifiers: set[str] = set()
        for entry in entries:
            chunk = DocumentChunk.model_validate(entry.chunk.model_dump()).model_copy(deep=True)
            if chunk.chunk_id in identifiers:
                raise ValueError("Duplicate vector chunk identifier")
            identifiers.add(chunk.chunk_id)
            replacement.append(EmbeddedChunk(chunk, normalize_embedding(entry.embedding, self.dimensions)))
        with self._lock:
            self._entries = tuple(replacement)
        return len(replacement)

    def search(
        self, embedding: Sequence[float], *, top_k: int, min_relevance: float,
        filters: Mapping[str, JsonValue] | None = None,
    ) -> list[RetrievalResult]:
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        cutoff = validate_min_relevance(min_relevance)
        query = normalize_embedding(embedding, self.dimensions)
        with self._lock:
            entries = self._entries
        results: list[RetrievalResult] = []
        for entry in entries:
            chunk = entry.chunk
            document_metadata = chunk.metadata.get("document_metadata")
            values = dict(document_metadata) if isinstance(document_metadata, dict) else {}
            values.update(chunk.metadata)
            values.update({
                "document_id": chunk.document_id, "chunk_id": chunk.chunk_id,
                "source_title": chunk.source_title, "section": chunk.section,
            })
            if filters and not all(key in values and values[key] == value for key, value in filters.items()):
                continue
            if query == entry.embedding:
                score = 1.0
            elif all(
                query_value == -stored_value
                for query_value, stored_value in zip(query, entry.embedding, strict=True)
            ):
                score = -1.0
            else:
                score = max(-1.0, min(1.0, math.fsum(
                    query_value * stored_value
                    for query_value, stored_value in zip(query, entry.embedding, strict=True)
                )))
            if score >= cutoff:
                results.append(RetrievalResult(
                    chunk_id=chunk.chunk_id, source_title=chunk.source_title,
                    section=chunk.section, content=chunk.content, score=score,
                ))
        return sorted(results, key=lambda result: (-result.score, result.chunk_id))[:top_k]