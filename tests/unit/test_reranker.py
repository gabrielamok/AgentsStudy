import socket
from collections.abc import Sequence

import pytest

from agent.rag.contracts import RetrievalResult
from agent.rag.reranker import Reranker, TokenJaccardScorer


def make_result(identifier: str, content: str, score: float = 4.5) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=identifier, source_title="Trusted source", section="Evidence",
        content=content, score=score,
    )


class FixedScorer:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[str] = []

    def score(self, query: str, candidate: RetrievalResult) -> float:
        self.calls.append(candidate.chunk_id)
        return self.scores[candidate.chunk_id]


@pytest.fixture(autouse=True)
def forbid_retrieval_and_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Reranking must not retrieve evidence or access the network")

    monkeypatch.setattr("agent.rag.service.RAGService.retrieve", forbidden)
    monkeypatch.setattr("agent.rag.vector_retriever.VectorRetriever.retrieve", forbidden)
    monkeypatch.setattr("agent.rag.hybrid_retriever.HybridRetriever.retrieve", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)


def test_candidate_identity_metadata_and_original_scores_are_preserved() -> None:
    candidates = [make_result("less", "water paper", 9.0), make_result("best", "clock", 0.2)]
    before = [candidate.model_dump() for candidate in candidates]

    observations = Reranker().rerank("clock", candidates, top_k=10)

    assert [item.result.chunk_id for item in observations] == ["best", "less"]
    assert [item.reranking_score for item in observations] == [1.0, 0.0]
    assert [item.retrieval_score for item in observations] == [0.2, 9.0]
    assert {item.result.chunk_id for item in observations} == {item.chunk_id for item in candidates}
    originals = {candidate.chunk_id: candidate.model_dump() for candidate in candidates}
    assert all(item.result.model_dump() == originals[item.result.chunk_id] for item in observations)
    assert [candidate.model_dump() for candidate in candidates] == before


@pytest.mark.parametrize("top_k", [1, 2, 10])
def test_injected_scores_stable_ties_and_top_k(top_k: int) -> None:
    candidates = [make_result("c", "clock"), make_result("b", "clock"), make_result("a", "paper")]
    scorer = FixedScorer({"a": 0.5, "b": 0.5, "c": -0.5})
    reranker = Reranker(scorer)

    results = reranker.rerank("clock", candidates, top_k=top_k)

    assert [item.result.chunk_id for item in results] == ["a", "b", "c"][:top_k]
    assert scorer.calls == ["c", "b", "a"]
    assert reranker.rerank("clock", list(reversed(candidates)), top_k=top_k) == results
    assert reranker.rerank("clock", candidates, top_k=top_k) == results


@pytest.mark.parametrize(("query", "content", "expected"), [
    ("CLOCK clock", "clock", 1.0), ("clock water", "clock paper", 1 / 3),
    ("clock", "water", 0.0), ("!!!", "???", 0.0),
    ("the clock", "clock", 1.0), ("clock", "clock, CLOCK!", 1.0),
])
def test_local_scoring_is_deterministic_and_uses_content_only(query: str, content: str, expected: float) -> None:
    scorer = TokenJaccardScorer(stop_words={"the"})
    candidate = make_result("chunk", content)
    assert scorer.score(query, candidate) == expected
    assert scorer.score(query, candidate) == expected
    assert scorer.score(query, candidate.model_copy(update={"source_title": "clock water paper"})) == expected


def test_empty_candidates_do_not_call_scorer() -> None:
    scorer = FixedScorer({})
    assert Reranker(scorer).rerank("clock", [], top_k=3) == []
    assert scorer.calls == []


def test_scorer_cannot_mutate_or_fabricate_returned_evidence() -> None:
    class MutatingScorer:
        def score(self, query: str, candidate: RetrievalResult) -> float:
            candidate.chunk_id = "fabricated"
            candidate.source_title = "invented"
            candidate.section = "invented"
            candidate.content = "invented"
            candidate.score = 999.0
            return 0.7

    candidate = make_result("actual", "clock", 2.0)
    observation = Reranker(MutatingScorer()).rerank("clock", [candidate], top_k=1)[0]
    assert observation.result == candidate
    assert observation.result.chunk_id == "actual"
    assert observation.retrieval_score == 2.0
    assert observation.reranking_score == 0.7
    observation.result.content = "changed observation"
    assert candidate.content == "clock"


@pytest.mark.parametrize("candidates", [
    [dict(chunk_id="a", source_title="source", section="section", content="clock", score=1.0)],
    [None], ["chunk"], "chunk", None, {"chunk": "clock"},
])
def test_only_existing_retrieval_result_candidates_are_accepted(candidates: Sequence[RetrievalResult]) -> None:
    with pytest.raises(ValueError, match="RetrievalResult"):
        Reranker().rerank("clock", candidates, top_k=1)


def test_duplicate_ids_are_rejected_before_scoring() -> None:
    scorer = FixedScorer({"same": 1.0})
    with pytest.raises(ValueError, match="Duplicate"):
        Reranker(scorer).rerank("clock", [make_result("same", "clock"), make_result("same", "paper")], top_k=1)
    assert scorer.calls == []


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf"), True, "1", None])
def test_invalid_reranking_scores_are_rejected(score: float) -> None:
    with pytest.raises(ValueError, match="finite real"):
        Reranker(FixedScorer({"chunk": score})).rerank("clock", [make_result("chunk", "clock")], top_k=1)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5, "1"])
def test_invalid_top_k_is_rejected_before_scoring(top_k: int) -> None:
    scorer = FixedScorer({})
    with pytest.raises(ValueError, match="top_k"):
        Reranker(scorer).rerank("clock", [], top_k=top_k)
    assert scorer.calls == []


@pytest.mark.parametrize("query", ["", "   ", None, 1])
def test_invalid_queries_are_rejected(query: str) -> None:
    with pytest.raises(ValueError, match="query"):
        Reranker().rerank(query, [], top_k=1)


def test_malformed_candidate_score_is_revalidated() -> None:
    candidate = make_result("bad", "clock").model_copy(update={"score": float("nan")})
    with pytest.raises(ValueError):
        Reranker().rerank("clock", [candidate], top_k=1)


def test_scorer_failure_propagates_without_fallback() -> None:
    class FailedScorer:
        def score(self, query: str, candidate: RetrievalResult) -> float:
            raise RuntimeError("scorer unavailable")

    with pytest.raises(RuntimeError, match="scorer unavailable"):
        Reranker(FailedScorer()).rerank("clock", [make_result("chunk", "clock")], top_k=1)