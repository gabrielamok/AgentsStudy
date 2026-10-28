import socket
from collections.abc import Sequence
from unittest.mock import Mock

import pytest

from agent.rag.contracts import DocumentChunk, RetrievalQuery
from agent.rag.embeddings import EmbeddingProvider
from agent.rag.vector_retriever import Retriever, VectorRetriever
from agent.rag.vector_store import InMemoryVectorStore, VectorStore


CLOCK_CONTENT = "A quartz oscillator supplies a stable clock signal."
REGULATOR_CONTENT = "A regulator stabilizes the voltage."


class LocalTestEmbeddingProvider:
    dimensions = 3

    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        vectors = {CLOCK_CONTENT: [1.0, 0.0, 0.0], REGULATOR_CONTENT: [3.0, 4.0, 0.0]}
        return [vectors[text].copy() for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return [1.0, 0.0, 0.0] if text == "timing" else [0.0, 0.0, 1.0]


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Vector unit tests must not access Azure or other network services")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.fixture
def chunks() -> list[DocumentChunk]:
    return [DocumentChunk(
        chunk_id=identifier, document_id="reference", source_title="Device reference",
        section="Timing" if identifier != "c" else "Regulation",
        content=CLOCK_CONTENT if identifier != "c" else REGULATOR_CONTENT,
        metadata={"document_metadata": {"file_name": "device.md"}, "category": "hardware"},
    ) for identifier in ("b", "a", "c")]


@pytest.fixture
def provider() -> LocalTestEmbeddingProvider:
    return LocalTestEmbeddingProvider()


@pytest.fixture
def retriever(provider: LocalTestEmbeddingProvider, chunks: list[DocumentChunk]) -> VectorRetriever:
    instance = VectorRetriever(provider, InMemoryVectorStore(3), min_relevance=0.5)
    assert instance.index_chunks(chunks) == 3
    return instance


def test_document_and_query_embeddings_use_the_explicit_local_provider(
    retriever: VectorRetriever, provider: LocalTestEmbeddingProvider, chunks: list[DocumentChunk],
) -> None:
    interface: Retriever = retriever
    query = RetrievalQuery(query="timing", top_k=3, filters={})
    results = interface.retrieve(query)

    assert provider.document_calls == [[chunk.content for chunk in chunks]]
    assert provider.query_calls == ["timing"]
    assert [result.chunk_id for result in results] == ["a", "b", "c"]
    assert [result.score for result in results] == pytest.approx([1.0, 1.0, 0.6])
    assert interface.retrieve(query) == results
    assert retriever.index_chunks(list(reversed(chunks))) == 3
    assert interface.retrieve(query) == results


@pytest.mark.parametrize("top_k", [1, 2, 3, 10])
def test_top_k_preserves_stable_similarity_order(retriever: VectorRetriever, top_k: int) -> None:
    full = retriever.retrieve(RetrievalQuery(query="timing", top_k=10, filters={}))
    assert retriever.retrieve(RetrievalQuery(query="timing", top_k=top_k, filters={})) == full[:top_k]


@pytest.mark.parametrize(("cutoff", "expected"), [(0.5, ["a", "b", "c"]), (0.7, ["a", "b"]), (1.0, ["a", "b"])])
def test_configurable_minimum_relevance(
    provider: LocalTestEmbeddingProvider, chunks: list[DocumentChunk], cutoff: float, expected: list[str],
) -> None:
    retriever = VectorRetriever(provider, InMemoryVectorStore(3), min_relevance=cutoff)
    retriever.index_chunks(chunks)
    assert [result.chunk_id for result in retriever.retrieve(
        RetrievalQuery(query="timing", top_k=10, filters={}),
    )] == expected


def test_no_results_never_fall_back_to_lexical(
    retriever: VectorRetriever, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Vector retrieval must not fall back to the lexical baseline")

    monkeypatch.setattr("agent.rag.service.RAGService.retrieve", forbidden)
    assert retriever.retrieve(RetrievalQuery(query="oscillator", top_k=3, filters={})) == []


def test_empty_index_and_clearing_have_no_results(
    provider: LocalTestEmbeddingProvider, chunks: list[DocumentChunk],
) -> None:
    retriever = VectorRetriever(provider, InMemoryVectorStore(3), min_relevance=0.5)
    query = RetrievalQuery(query="timing", top_k=3, filters={})
    assert retriever.retrieve(query) == []
    assert retriever.index_chunks(chunks) == 3
    assert retriever.index_chunks([]) == 0
    assert retriever.retrieve(query) == []
    assert len(provider.document_calls) == 1


def test_metadata_preservation_and_filtering_before_top_k(
    retriever: VectorRetriever, chunks: list[DocumentChunk],
) -> None:
    result = retriever.retrieve(RetrievalQuery(
        query="timing", top_k=1, filters={"file_name": "device.md", "chunk_id": "c"},
    ))[0]
    original = next(chunk for chunk in chunks if chunk.chunk_id == "c")
    assert result.chunk_id == original.chunk_id
    assert result.source_title == original.source_title
    assert result.section == original.section
    assert result.content == original.content
    assert result.score == pytest.approx(0.6)
    assert set(result.model_dump()) == {"chunk_id", "source_title", "section", "content", "score"}
    assert retriever.retrieve(RetrievalQuery(query="timing", top_k=3, filters={"category": "missing"})) == []
    original.source_title = "mutated"
    original.metadata["category"] = "mutated"
    assert retriever.retrieve(RetrievalQuery(query="timing", top_k=3, filters={"category": "hardware"}))[2] == result


def test_provider_and_store_must_be_explicit() -> None:
    with pytest.raises(TypeError):
        VectorRetriever()


def test_dimension_mismatch_is_rejected_before_provider_calls(provider: LocalTestEmbeddingProvider) -> None:
    with pytest.raises(ValueError, match="dimension mismatch"):
        VectorRetriever(provider, InMemoryVectorStore(2))
    assert provider.document_calls == []
    assert provider.query_calls == []


@pytest.mark.parametrize("cutoff", [-1.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_cutoff_is_rejected(provider: LocalTestEmbeddingProvider, cutoff: float) -> None:
    with pytest.raises(ValueError):
        VectorRetriever(provider, InMemoryVectorStore(3), min_relevance=cutoff)


@pytest.mark.parametrize("batch", [[], [[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]] * 4, None, "invalid"])
def test_invalid_embedding_batch_does_not_replace_index(
    retriever: VectorRetriever, chunks: list[DocumentChunk], batch: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = RetrievalQuery(query="timing", top_k=3, filters={})
    before = retriever.retrieve(query)
    monkeypatch.setattr(retriever.embedding_provider, "embed_documents", Mock(return_value=batch))
    with pytest.raises(ValueError, match="one vector per document chunk"):
        retriever.index_chunks(chunks)
    assert retriever.retrieve(query) == before


@pytest.mark.parametrize("vector", [[], [1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [float("nan"), 0.0, 1.0], [True, 0.0, 1.0]])
def test_malformed_document_vectors_do_not_replace_index(
    retriever: VectorRetriever, chunks: list[DocumentChunk], vector: list[float], monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = RetrievalQuery(query="timing", top_k=3, filters={})
    before = retriever.retrieve(query)
    monkeypatch.setattr(retriever.embedding_provider, "embed_documents", Mock(return_value=[[1.0, 0.0, 0.0], vector, [1.0, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        retriever.index_chunks(chunks)
    assert retriever.retrieve(query) == before


@pytest.mark.parametrize("vector", [[], [1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [float("inf"), 0.0, 1.0], [False, 0.0, 1.0]])
def test_malformed_query_vectors_raise_without_fallback(
    retriever: VectorRetriever, vector: list[float], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(retriever.embedding_provider, "embed_query", Mock(return_value=vector))
    with pytest.raises(ValueError):
        retriever.retrieve(RetrievalQuery(query="timing", top_k=3, filters={}))


def test_provider_indexing_failure_preserves_existing_index(
    retriever: VectorRetriever, chunks: list[DocumentChunk], monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = RetrievalQuery(query="timing", top_k=3, filters={})
    before = retriever.retrieve(query)
    monkeypatch.setattr(retriever.embedding_provider, "embed_documents", Mock(side_effect=RuntimeError("provider unavailable")))
    with pytest.raises(RuntimeError, match="provider unavailable"):
        retriever.index_chunks(chunks)
    assert retriever.retrieve(query) == before


def test_query_provider_failure_is_not_silently_replaced(
    retriever: VectorRetriever, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(retriever.embedding_provider, "embed_query", Mock(side_effect=RuntimeError("provider unavailable")))
    with pytest.raises(RuntimeError, match="provider unavailable"):
        retriever.retrieve(RetrievalQuery(query="timing", top_k=3, filters={}))


def test_store_failure_propagates_without_fallback(provider: LocalTestEmbeddingProvider) -> None:
    store = Mock(spec=VectorStore)
    store.dimensions = 3
    store.search.side_effect = RuntimeError("store unavailable")
    retriever = VectorRetriever(provider, store)
    with pytest.raises(RuntimeError, match="store unavailable"):
        retriever.retrieve(RetrievalQuery(query="timing", top_k=3, filters={}))


def test_invalid_query_contract_is_rejected_before_embedding(
    retriever: VectorRetriever, provider: LocalTestEmbeddingProvider,
) -> None:
    with pytest.raises(ValueError):
        retriever.retrieve(RetrievalQuery.model_construct(query="timing", top_k=0, filters={}))
    assert provider.query_calls == []


def test_provider_dimension_contract_is_validated() -> None:
    provider = Mock(spec=EmbeddingProvider)
    provider.dimensions = 0
    with pytest.raises(ValueError, match="positive integer"):
        VectorRetriever(provider, InMemoryVectorStore(3))


@pytest.mark.parametrize("vector", [[1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [1.0, 3.0, 7.0], [1.0, 11.0, 101.0]])
def test_identical_provider_vectors_survive_exact_cutoff_end_to_end(
    chunks: list[DocumentChunk], vector: list[float],
) -> None:
    provider = Mock(spec=EmbeddingProvider)
    provider.dimensions = 3
    provider.embed_documents.return_value = [vector.copy() for chunk in chunks]
    provider.embed_query.return_value = vector.copy()
    retriever = VectorRetriever(provider, InMemoryVectorStore(3), min_relevance=1.0)
    retriever.index_chunks(chunks)

    results = retriever.retrieve(RetrievalQuery(query="identical vector", top_k=3, filters={}))

    assert [result.chunk_id for result in results] == ["a", "b", "c"]
    assert [result.score for result in results] == [1.0, 1.0, 1.0]