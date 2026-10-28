import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Protocol

from agent.rag.contracts import RetrievalQuery, RetrievalResult


class CandidateRetriever(Protocol):
    def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        ...


@dataclass(frozen=True)
class HybridRetrievalResult:
    result: RetrievalResult
    lexical_raw_score: float | None
    vector_raw_score: float | None
    lexical_normalized_score: float
    vector_normalized_score: float
    lexical_contribution: float
    vector_contribution: float


def _nonnegative_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a nonnegative finite number")
    try:
        number = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a nonnegative finite number") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a nonnegative finite number")
    return number


class HybridRetriever:
    """Opt-in weighted fusion over independently retrieved candidate pools.

    Each pool's positive scores are divided by its maximum positive score;
    missing, zero and negative scores contribute zero. This preserves a zero
    baseline rather than promoting negative cosine similarity into relevance.
    Weights are normalized to sum to one; combined scores are in [0, 1], not
    confidence estimates. The minimum relevance cutoff is inclusive.

    Both children receive the original question/filters and at least
    candidate_top_k candidates, independent of each other's output. Increasing
    the requested top_k above that pool size expands both pools. No provider,
    index, generator, fallback or application default is selected here.
    Child errors propagate; duplicate IDs with inconsistent metadata fail.
    Observations are returned per call, with no mutable last-query state.
    """

    def __init__(
        self, lexical_retriever: CandidateRetriever, vector_retriever: CandidateRetriever, *,
        lexical_weight: float = 0.5, vector_weight: float = 0.5,
        candidate_top_k: int = 50, min_relevance: float = 0.0,
    ) -> None:
        lexical = _nonnegative_finite(lexical_weight, "lexical_weight")
        vector = _nonnegative_finite(vector_weight, "vector_weight")
        scale = max(lexical, vector)
        if scale == 0:
            raise ValueError("At least one component weight must be positive")
        if type(candidate_top_k) is not int or candidate_top_k <= 0:
            raise ValueError("candidate_top_k must be a positive integer")
        cutoff = _nonnegative_finite(min_relevance, "min_relevance")
        if cutoff > 1:
            raise ValueError("min_relevance must be in [0, 1]")
        total = lexical / scale + vector / scale
        self.lexical_weight = (lexical / scale) / total
        self.vector_weight = (vector / scale) / total
        self.candidate_top_k = candidate_top_k
        self.min_relevance = cutoff
        self.lexical_retriever = lexical_retriever
        self.vector_retriever = vector_retriever

    @staticmethod
    def _metadata(result: RetrievalResult) -> tuple[str, str, str]:
        return result.source_title, result.section, result.content

    @classmethod
    def _deduplicate(cls, results: Sequence[RetrievalResult]) -> dict[str, RetrievalResult]:
        if not isinstance(results, Sequence) or isinstance(results, (str, bytes, bytearray)):
            raise ValueError("A candidate retriever must return a sequence of retrieval results")
        candidates: dict[str, RetrievalResult] = {}
        for result in results:
            validated = RetrievalResult.model_validate(result.model_dump()).model_copy(deep=True)
            previous = candidates.get(validated.chunk_id)
            if previous is not None and cls._metadata(previous) != cls._metadata(validated):
                raise ValueError(f"Conflicting metadata for chunk_id: {validated.chunk_id}")
            if previous is None or validated.score > previous.score:
                candidates[validated.chunk_id] = validated
        return candidates

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        return [observation.result for observation in self.retrieve_with_scores(query)]

    def retrieve_with_scores(self, query: RetrievalQuery) -> list[HybridRetrievalResult]:
        validated = RetrievalQuery.model_validate(query.model_dump()).model_copy(deep=True)
        candidate_query = validated.model_copy(update={
            "top_k": max(validated.top_k, self.candidate_top_k),
        })
        lexical = self._deduplicate(self.lexical_retriever.retrieve(candidate_query.model_copy(deep=True)))
        vector = self._deduplicate(self.vector_retriever.retrieve(candidate_query.model_copy(deep=True)))
        lexical_max = max((max(0.0, result.score) for result in lexical.values()), default=0.0)
        vector_max = max((max(0.0, result.score) for result in vector.values()), default=0.0)
        observations = []
        for identifier in lexical.keys() | vector.keys():
            lexical_result, vector_result = lexical.get(identifier), vector.get(identifier)
            if lexical_result is not None and vector_result is not None:
                if self._metadata(lexical_result) != self._metadata(vector_result):
                    raise ValueError(f"Conflicting metadata for chunk_id: {identifier}")
            lexical_raw = lexical_result.score if lexical_result is not None else None
            vector_raw = vector_result.score if vector_result is not None else None
            lexical_normalized = max(0.0, lexical_raw) / lexical_max if lexical_raw is not None and lexical_max else 0.0
            vector_normalized = max(0.0, vector_raw) / vector_max if vector_raw is not None and vector_max else 0.0
            lexical_contribution = self.lexical_weight * lexical_normalized
            vector_contribution = self.vector_weight * vector_normalized
            score = math.fsum((lexical_contribution, vector_contribution))
            if score < self.min_relevance:
                continue
            original = lexical_result if lexical_result is not None else vector_result
            assert original is not None
            observations.append(HybridRetrievalResult(
                result=original.model_copy(update={"score": score}, deep=True),
                lexical_raw_score=lexical_raw, vector_raw_score=vector_raw,
                lexical_normalized_score=lexical_normalized, vector_normalized_score=vector_normalized,
                lexical_contribution=lexical_contribution, vector_contribution=vector_contribution,
            ))
        return sorted(observations, key=lambda observation: (-observation.result.score, observation.result.chunk_id))[:validated.top_k]