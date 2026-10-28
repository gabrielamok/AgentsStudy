import pytest

from agent.rag.contracts import DocumentChunk
from agent.rag.vector_store import EmbeddedChunk, InMemoryVectorStore, VectorStore


def make_chunk(identifier: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=identifier, document_id="document", source_title="Clock reference",
        section="Timing", content="A quartz oscillator supplies a stable clock signal.",
        metadata={"document_metadata": {"category": "study"}, "tag": "trusted"},
    )


@pytest.fixture
def store() -> InMemoryVectorStore:
    instance = InMemoryVectorStore(2)
    instance.replace([
        EmbeddedChunk(make_chunk("b"), [1.0, 0.0]),
        EmbeddedChunk(make_chunk("a"), [2.0, 0.0]),
        EmbeddedChunk(make_chunk("c"), [3.0, 4.0]),
        EmbeddedChunk(make_chunk("d"), [0.0, 1.0]),
        EmbeddedChunk(make_chunk("e"), [-1.0, 0.0]),
    ])
    return instance


def test_cosine_ordering_and_stable_chunk_id_ties(store: InMemoryVectorStore) -> None:
    interface: VectorStore = store
    results = interface.search([1.0, 0.0], top_k=10, min_relevance=-1.0)

    assert [result.chunk_id for result in results] == ["a", "b", "c", "d", "e"]
    assert [result.score for result in results] == pytest.approx([1.0, 1.0, 0.6, 0.0, -1.0])
    assert interface.search([10.0, 0.0], top_k=10, min_relevance=-1.0) == results
    assert interface.search([1.0, 0.0], top_k=10, min_relevance=-1.0) == results


@pytest.mark.parametrize("top_k", [1, 2, 3, 10])
def test_top_k_is_a_prefix_of_the_full_order(store: InMemoryVectorStore, top_k: int) -> None:
    full = store.search([1.0, 0.0], top_k=100, min_relevance=-1.0)

    assert store.search([1.0, 0.0], top_k=top_k, min_relevance=-1.0) == full[:top_k]


@pytest.mark.parametrize(("cutoff", "expected"), [
    (-1.0, ["a", "b", "c", "d", "e"]), (0.0, ["a", "b", "c", "d"]),
    (0.6, ["a", "b", "c"]), (0.61, ["a", "b"]), (1.0, ["a", "b"]),
])
def test_minimum_relevance_is_inclusive(
    store: InMemoryVectorStore, cutoff: float, expected: list[str],
) -> None:
    assert [result.chunk_id for result in store.search(
        [1.0, 0.0], top_k=10, min_relevance=cutoff,
    )] == expected


def test_empty_index_and_unrelated_vectors_return_no_results() -> None:
    store = InMemoryVectorStore(2)
    assert store.search([1.0, 0.0], top_k=3, min_relevance=0.5) == []
    store.replace([EmbeddedChunk(make_chunk("a"), [0.0, 1.0])])
    assert store.search([1.0, 0.0], top_k=3, min_relevance=0.5) == []
    assert store.replace([]) == 0
    assert store.search([1.0, 0.0], top_k=3, min_relevance=-1.0) == []


@pytest.mark.parametrize("filters", [
    {"category": "study"}, {"tag": "trusted"}, {"chunk_id": "a"},
    {"document_id": "document"}, {"source_title": "Clock reference"}, {"section": "Timing"},
])
def test_filters_match_metadata_and_authoritative_identity(
    store: InMemoryVectorStore, filters: dict[str, str],
) -> None:
    assert store.search([1.0, 0.0], top_k=1, min_relevance=0.5, filters=filters)[0].chunk_id == "a"


def test_filtering_happens_before_top_k(store: InMemoryVectorStore) -> None:
    results = store.search([1.0, 0.0], top_k=1, min_relevance=0.5, filters={"chunk_id": "c"})
    assert [result.chunk_id for result in results] == ["c"]
    assert store.search([1.0, 0.0], top_k=1, min_relevance=-1.0, filters={"unknown": "value"}) == []


def test_metadata_is_preserved_and_store_owns_independent_copies() -> None:
    store = InMemoryVectorStore(2)
    chunk = make_chunk("original")
    chunk.metadata["chunk_id"] = "spoofed"
    vector = [1.0, 0.0]
    store.replace([EmbeddedChunk(chunk, vector)])
    chunk.source_title = "mutated"
    chunk.metadata["tag"] = "mutated"
    vector[:] = [0.0, 1.0]

    result = store.search([1.0, 0.0], top_k=1, min_relevance=1.0, filters={"tag": "trusted"})[0]
    assert result.model_dump() == {
        "chunk_id": "original", "source_title": "Clock reference", "section": "Timing",
        "content": "A quartz oscillator supplies a stable clock signal.", "score": 1.0,
    }
    assert store.search([1.0, 0.0], top_k=1, min_relevance=1.0, filters={"chunk_id": "spoofed"}) == []
    result.content = "changed result"
    assert store.search([1.0, 0.0], top_k=1, min_relevance=1.0)[0].content != result.content


@pytest.mark.parametrize("vector", [[], [1.0], [1.0, 2.0, 3.0], [0.0, 0.0], [float("nan"), 1.0], [True, 1.0]])
def test_invalid_document_vectors_do_not_replace_existing_index(
    store: InMemoryVectorStore, vector: list[float],
) -> None:
    before = store.search([1.0, 0.0], top_k=10, min_relevance=-1.0)
    with pytest.raises(ValueError):
        store.replace([EmbeddedChunk(make_chunk("valid"), [1.0, 0.0]), EmbeddedChunk(make_chunk("bad"), vector)])
    assert store.search([1.0, 0.0], top_k=10, min_relevance=-1.0) == before


def test_duplicate_ids_are_rejected_atomically(store: InMemoryVectorStore) -> None:
    before = store.search([1.0, 0.0], top_k=10, min_relevance=-1.0)
    with pytest.raises(ValueError, match="Duplicate"):
        store.replace([EmbeddedChunk(make_chunk("same"), [1.0, 0.0]), EmbeddedChunk(make_chunk("same"), [0.0, 1.0])])
    assert store.search([1.0, 0.0], top_k=10, min_relevance=-1.0) == before


@pytest.mark.parametrize("vector", [[], [1.0], [1.0, 2.0, 3.0], [0.0, 0.0], [float("inf"), 1.0], [False, 1.0]])
def test_invalid_query_vectors_are_rejected(store: InMemoryVectorStore, vector: list[float]) -> None:
    with pytest.raises(ValueError):
        store.search(vector, top_k=3, min_relevance=0.5)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k_is_rejected(store: InMemoryVectorStore, top_k: int) -> None:
    with pytest.raises(ValueError):
        store.search([1.0, 0.0], top_k=top_k, min_relevance=0.5)


@pytest.mark.parametrize("cutoff", [-1.1, 1.1, float("nan"), float("inf"), True, "0.5", None])
def test_invalid_relevance_is_rejected(store: InMemoryVectorStore, cutoff: float) -> None:
    with pytest.raises(ValueError):
        store.search([1.0, 0.0], top_k=3, min_relevance=cutoff)


@pytest.mark.parametrize("dimensions", [0, -1, True, 2.5])
def test_invalid_store_dimensions_are_rejected(dimensions: int) -> None:
    with pytest.raises(ValueError):
        InMemoryVectorStore(dimensions)


def test_rebuild_order_does_not_change_ties(store: InMemoryVectorStore) -> None:
    before = store.search([1.0, 0.0], top_k=2, min_relevance=1.0)
    store.replace([EmbeddedChunk(make_chunk("a"), [1.0, 0.0]), EmbeddedChunk(make_chunk("b"), [1.0, 0.0])])
    assert store.search([1.0, 0.0], top_k=2, min_relevance=1.0) == before


@pytest.mark.parametrize("scale", [1.0, 2.0, 1e308, 1e-308])
@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_diagonal_vectors_have_exact_endpoint_similarity(scale: float, direction: float) -> None:
    store = InMemoryVectorStore(2)
    store.replace([
        EmbeddedChunk(make_chunk("b"), [direction * scale, direction * scale]),
        EmbeddedChunk(make_chunk("a"), [direction * scale, direction * scale]),
    ])

    results = store.search([1.0, 1.0], top_k=2, min_relevance=direction)

    assert [result.chunk_id for result in results] == ["a", "b"]
    assert [result.score for result in results] == [direction, direction]


def test_nearby_but_distinct_vectors_are_not_promoted_to_perfect_similarity() -> None:
    store = InMemoryVectorStore(2)
    store.replace([EmbeddedChunk(make_chunk("a"), [1.0, 1.0])])

    assert store.search([1.0, 1.001], top_k=1, min_relevance=1.0) == []
    result = store.search([1.0, 1.001], top_k=1, min_relevance=0.0)[0]
    assert 0.99 < result.score < 1.0