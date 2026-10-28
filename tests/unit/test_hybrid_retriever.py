from collections.abc import Sequence

import pytest

from agent.rag.contracts import RetrievalQuery, RetrievalResult
from agent.rag.hybrid_retriever import HybridRetriever


def make_result(identifier: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=identifier, source_title="Trusted source", section="Evidence",
        content=f"Trusted content for {identifier}.", score=score,
    )


class StubRetriever:
    def __init__(self, results: Sequence[RetrievalResult]) -> None:
        self.results = list(results)
        self.calls: list[RetrievalQuery] = []

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        self.calls.append(query.model_copy(deep=True))
        return self.results[:query.top_k]


def query(top_k: int = 10) -> RetrievalQuery:
    return RetrievalQuery(query="trusted evidence", top_k=top_k, filters={"category": "study"})


@pytest.mark.parametrize("side", ["lexical", "vector"])
def test_single_strategy_candidates_keep_metadata_and_expose_scores(side: str) -> None:
    original = make_result("only", 4.0)
    lexical = StubRetriever([original] if side == "lexical" else [])
    vector = StubRetriever([original] if side == "vector" else [])
    retriever = HybridRetriever(lexical, vector)

    observation = retriever.retrieve_with_scores(query())[0]

    assert observation.result.model_dump() == {**original.model_dump(), "score": 0.5}
    assert observation.lexical_raw_score == (4.0 if side == "lexical" else None)
    assert observation.vector_raw_score == (4.0 if side == "vector" else None)
    assert observation.lexical_normalized_score == (1.0 if side == "lexical" else 0.0)
    assert observation.vector_normalized_score == (1.0 if side == "vector" else 0.0)
    assert observation.lexical_contribution + observation.vector_contribution == observation.result.score
    assert retriever.retrieve(query()) == [observation.result]
    assert original.score == 4.0


def test_duplicate_candidates_use_max_component_score_and_are_combined_once() -> None:
    retriever = HybridRetriever(
        StubRetriever([make_result("shared", 2.0), make_result("shared", 8.0)]),
        StubRetriever([make_result("shared", 0.2), make_result("shared", 0.8)]),
    )

    observations = retriever.retrieve_with_scores(query())

    assert len(observations) == 1
    assert observations[0].result.score == 1.0
    assert observations[0].lexical_raw_score == 8.0
    assert observations[0].vector_raw_score == 0.8


def test_normalization_and_weighted_combination_are_scale_independent() -> None:
    def build(scale: float) -> HybridRetriever:
        return HybridRetriever(
            StubRetriever([make_result("a", 100.0 * scale), make_result("b", 50.0 * scale)]),
            StubRetriever([make_result("a", 0.2), make_result("b", 0.8)]),
            lexical_weight=3.0, vector_weight=1.0,
        )

    observations = build(1.0).retrieve_with_scores(query())

    assert [item.result.chunk_id for item in observations] == ["a", "b"]
    assert [item.result.score for item in observations] == [0.8125, 0.625]
    assert [item.lexical_normalized_score for item in observations] == [1.0, 0.5]
    assert [item.vector_normalized_score for item in observations] == [0.25, 1.0]
    assert build(1000.0).retrieve(query()) == [item.result for item in observations]


@pytest.mark.parametrize("top_k", [1, 2, 10])
def test_stable_ties_top_k_and_union_membership(top_k: int) -> None:
    lexical = StubRetriever([make_result("b", 10.0)])
    vector = StubRetriever([make_result("a", 0.5), make_result("c", 0.5)])
    retriever = HybridRetriever(lexical, vector)

    results = retriever.retrieve(query(top_k))

    assert [result.chunk_id for result in results] == ["a", "b", "c"][:top_k]
    assert {result.chunk_id for result in results} <= {"a", "b", "c"}
    vector.results.reverse()
    assert retriever.retrieve(query(top_k)) == results


@pytest.mark.parametrize(("cutoff", "expected"), [(0.0, ["a", "b"]), (0.5, ["a", "b"]), (0.50001, []), (1.0, [])])
def test_minimum_combined_relevance_is_inclusive(cutoff: float, expected: list[str]) -> None:
    retriever = HybridRetriever(
        StubRetriever([make_result("a", 4.0)]), StubRetriever([make_result("b", 0.4)]), min_relevance=cutoff,
    )
    assert [result.chunk_id for result in retriever.retrieve(query())] == expected


def test_cutoff_applies_before_top_k() -> None:
    retriever = HybridRetriever(
        StubRetriever([make_result("low", 1.0), make_result("high", 10.0)]),
        StubRetriever([]), min_relevance=0.4,
    )
    assert [result.chunk_id for result in retriever.retrieve(query(1))] == ["high"]


@pytest.mark.parametrize(("lexical_weight", "vector_weight", "expected"), [(1.0, 0.0, "a"), (0.0, 1.0, "b"), (1e308, 1e308, "a")])
def test_zero_and_extreme_weights(lexical_weight: float, vector_weight: float, expected: str) -> None:
    retriever = HybridRetriever(
        StubRetriever([make_result("a", 10.0)]), StubRetriever([make_result("b", 1.0)]),
        lexical_weight=lexical_weight, vector_weight=vector_weight, min_relevance=0.1,
    )
    assert retriever.retrieve(query(1))[0].chunk_id == expected


def test_nonpositive_scores_are_not_promoted_and_empty_pools_are_safe() -> None:
    retriever = HybridRetriever(
        StubRetriever([make_result("zero", 0.0)]), StubRetriever([make_result("negative", -0.5)]),
    )
    observations = retriever.retrieve_with_scores(query())
    assert [item.result.score for item in observations] == [0.0, 0.0]
    assert observations[0].vector_raw_score == -0.5
    assert HybridRetriever(StubRetriever([]), StubRetriever([])).retrieve(query()) == []
    assert HybridRetriever(StubRetriever([]), StubRetriever([make_result("negative", -0.5)]), min_relevance=0.1).retrieve(query()) == []


def test_children_receive_independent_queries_and_requested_candidate_pools() -> None:
    class MutatingRetriever(StubRetriever):
        def retrieve(self, request: RetrievalQuery) -> list[RetrievalResult]:
            self.calls.append(request.model_copy(deep=True))
            request.query = "changed"
            request.filters["category"] = "changed"
            return self.results

    original = query(8)
    lexical, vector = MutatingRetriever([]), StubRetriever([])
    HybridRetriever(lexical, vector, candidate_top_k=12).retrieve(original)
    expected = original.model_copy(update={"top_k": 12})
    assert lexical.calls == vector.calls == [expected]
    assert original == query(8)
    HybridRetriever(StubRetriever([]), vector, candidate_top_k=2).retrieve(original)
    assert vector.calls[-1].top_k == 8


@pytest.mark.parametrize("field", ["source_title", "section", "content"])
@pytest.mark.parametrize("same_strategy", [True, False])
def test_conflicting_duplicate_metadata_is_rejected(field: str, same_strategy: bool) -> None:
    original = make_result("same", 2.0)
    conflict = original.model_copy(update={field: "conflicting value"})
    lexical = StubRetriever([original, conflict] if same_strategy else [original])
    vector = StubRetriever([] if same_strategy else [conflict])
    with pytest.raises(ValueError, match="Conflicting metadata"):
        HybridRetriever(lexical, vector).retrieve(query())


def test_returned_metadata_is_independent_of_child_results_and_prior_observations() -> None:
    original = make_result("same", 5.0)
    retriever = HybridRetriever(StubRetriever([original]), StubRetriever([]))
    observation = retriever.retrieve_with_scores(query())[0]
    observation.result.content = "mutated"
    assert retriever.retrieve(query())[0].content == original.content
    assert original.content != observation.result.content


@pytest.mark.parametrize("name", ["lexical_weight", "vector_weight", "min_relevance"])
@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), True, "0.5", None])
def test_invalid_numeric_configuration_is_rejected(name: str, value: object) -> None:
    with pytest.raises(ValueError):
        HybridRetriever(StubRetriever([]), StubRetriever([]), **{name: value})


@pytest.mark.parametrize("configuration", [
    {"lexical_weight": 0.0, "vector_weight": 0.0}, {"min_relevance": 1.01},
    {"candidate_top_k": 0}, {"candidate_top_k": -1}, {"candidate_top_k": True}, {"candidate_top_k": 1.5},
])
def test_invalid_configuration_ranges_are_rejected(configuration: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        HybridRetriever(StubRetriever([]), StubRetriever([]), **configuration)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_malformed_child_scores_are_revalidated(score: float) -> None:
    corrupted = make_result("bad", 1.0).model_copy(update={"score": score})
    with pytest.raises(ValueError):
        HybridRetriever(StubRetriever([corrupted]), StubRetriever([])).retrieve(query())


@pytest.mark.parametrize("side", ["lexical", "vector"])
def test_child_failures_propagate_without_fallback(side: str) -> None:
    class FailedRetriever:
        def retrieve(self, request: RetrievalQuery) -> list[RetrievalResult]:
            raise RuntimeError("child unavailable")

    failed, empty = FailedRetriever(), StubRetriever([])
    retriever = HybridRetriever(failed if side == "lexical" else empty, failed if side == "vector" else empty)
    with pytest.raises(RuntimeError, match="child unavailable"):
        retriever.retrieve(query())


def test_invalid_query_is_rejected_before_children_are_called() -> None:
    lexical, vector = StubRetriever([]), StubRetriever([])
    with pytest.raises(ValueError):
        HybridRetriever(lexical, vector).retrieve(query().model_copy(update={"top_k": True}))
    assert lexical.calls == vector.calls == []