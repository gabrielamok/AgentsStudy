from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, JsonValue


NonEmptyText = Annotated[str, Field(min_length=1)]


class _RAGContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, str_strip_whitespace=True, allow_inf_nan=False,
    )


class KnowledgeDocument(_RAGContract):
    document_id: NonEmptyText
    title: NonEmptyText
    source_path: NonEmptyText
    content: NonEmptyText
    metadata: dict[str, JsonValue]


class DocumentChunk(_RAGContract):
    chunk_id: NonEmptyText
    document_id: NonEmptyText
    source_title: NonEmptyText
    section: NonEmptyText
    content: NonEmptyText
    metadata: dict[str, JsonValue]


class RetrievalQuery(_RAGContract):
    query: NonEmptyText
    top_k: int = Field(gt=0)
    filters: dict[str, JsonValue]


class RetrievalResult(_RAGContract):
    chunk_id: NonEmptyText
    source_title: NonEmptyText
    section: NonEmptyText
    content: NonEmptyText
    score: float


class SourceCitation(_RAGContract):
    source_title: NonEmptyText
    section: NonEmptyText
    chunk_id: NonEmptyText


class GroundedAnswer(_RAGContract):
    answer: NonEmptyText
    citations: list[SourceCitation]
    grounded: bool
    insufficient_context: bool


class RAGRequest(RetrievalQuery):
    """Public RAG request using the same query configuration as retrieval."""


class RAGResponse(_RAGContract):
    success: bool
    query: NonEmptyText
    answer: NonEmptyText
    retrieved_chunks: list[RetrievalResult]
    citations: list[SourceCitation]
    grounded: bool
    insufficient_context: bool
    error: NonEmptyText | None


class RAGHealthResponse(_RAGContract):
    success: bool
    indexed: bool
    indexed_chunks: int = Field(ge=0)
    error: NonEmptyText | None


class RAGReindexResponse(_RAGContract):
    success: bool
    indexed_chunks: int = Field(ge=0)
    error: NonEmptyText | None