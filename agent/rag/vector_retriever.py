from collections.abc import Sequence
from typing import Protocol

from agent.rag.contracts import DocumentChunk, RetrievalQuery, RetrievalResult
from agent.rag.embeddings import EmbeddingProvider, normalize_embedding, validate_dimensions
from agent.rag.vector_store import EmbeddedChunk, VectorStore, validate_min_relevance


class Retriever(Protocol):
    """Vector retrieval interface over already-ingested, metadata-bearing chunks."""

    def index_chunks(self, chunks: Sequence[DocumentChunk]) -> int:
        ...

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        ...


class VectorRetriever:
    """Opt-in vector retrieval; never selects a provider or falls back to lexical.

    The caller must supply a provider and a store in the same embedding space.
    Minimum relevance is an inclusive cosine cutoff in [-1, 1], not confidence;
    the default 0 includes orthogonal vectors. Use a positive cutoff to exclude them.
    Indexing replaces the complete chunk set atomically. Provider and store errors
    propagate to the caller; this class does not alter the application's default.
    """

    def __init__(
        self, embedding_provider: EmbeddingProvider, vector_store: VectorStore, *,
        min_relevance: float = 0.0,
    ) -> None:
        self._dimensions = validate_dimensions(embedding_provider.dimensions)
        if validate_dimensions(vector_store.dimensions) != self._dimensions:
            raise ValueError("Embedding provider and vector store dimension mismatch")
        self._min_relevance = validate_min_relevance(min_relevance)
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    @property
    def min_relevance(self) -> float:
        return self._min_relevance

    def index_chunks(self, chunks: Sequence[DocumentChunk]) -> int:
        documents = [DocumentChunk.model_validate(chunk.model_dump()).model_copy(deep=True) for chunk in chunks]
        if not documents:
            return self.vector_store.replace([])
        embeddings = self.embedding_provider.embed_documents([chunk.content for chunk in documents])
        if (not isinstance(embeddings, Sequence) or isinstance(embeddings, (str, bytes, bytearray))
                or len(embeddings) != len(documents)):
            raise ValueError("Embedding provider must return one vector per document chunk")
        entries = [
            EmbeddedChunk(chunk, normalize_embedding(embedding, self._dimensions))
            for chunk, embedding in zip(documents, embeddings, strict=True)
        ]
        return self.vector_store.replace(entries)

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        validated = RetrievalQuery.model_validate(query.model_dump()).model_copy(deep=True)
        embedding = normalize_embedding(
            self.embedding_provider.embed_query(validated.query), self._dimensions,
        )
        return self.vector_store.search(
            embedding, top_k=validated.top_k, min_relevance=self.min_relevance,
            filters=validated.filters,
        )