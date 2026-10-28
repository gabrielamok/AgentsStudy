import math
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Protocol

from agent.rag.contracts import RetrievalResult


class RerankingScorer(Protocol):
    def score(self, query: str, candidate: RetrievalResult) -> float:
        ...


class TokenJaccardScorer:
    """Deterministic token-set similarity over the supplied content only.

    Case-folded Unicode words are used without stemming, synonyms, training,
    source boosts or external services. Empty token unions score zero.
    """

    def __init__(self, *, stop_words: Collection[str] = ()) -> None:
        self.stop_words = frozenset(word.casefold() for word in stop_words)

    def _terms(self, text: str) -> set[str]:
        return set(re.findall(r"[^\W_]+", text.casefold())) - self.stop_words

    def score(self, query: str, candidate: RetrievalResult) -> float:
        query_terms, content_terms = self._terms(query), self._terms(candidate.content)
        union = query_terms | content_terms
        return len(query_terms & content_terms) / len(union) if union else 0.0


@dataclass(frozen=True)
class RerankedResult:
    result: RetrievalResult
    reranking_score: float

    @property
    def retrieval_score(self) -> float:
        return self.result.score


class Reranker:
    """Order only supplied candidates, never fetching or creating new evidence.

    Original retrieval results, including scores and metadata, are independently
    copied and retained unchanged. Reranking scores are separate observations.
    Scorers receive disposable copies, so their mutations cannot alter evidence.
    Duplicate chunk IDs are rejected, and scorer errors propagate without fallback.
    Equal scores are ordered by chunk ID, regardless of initial candidate order.
    The default scorer is local token Jaccard similarity, not semantic relevance.
    """

    def __init__(self, scorer: RerankingScorer | None = None) -> None:
        self.scorer = scorer if scorer is not None else TokenJaccardScorer()

    def rerank(
        self, query: str, candidates: Sequence[RetrievalResult], *, top_k: int,
    ) -> list[RerankedResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes, bytearray)):
            raise ValueError("candidates must be a sequence of existing RetrievalResult objects")
        validated = []
        identifiers: set[str] = set()
        for candidate in candidates:
            if not isinstance(candidate, RetrievalResult):
                raise ValueError("candidates must contain only existing RetrievalResult objects")
            result = RetrievalResult.model_validate(candidate.model_dump()).model_copy(deep=True)
            if result.chunk_id in identifiers:
                raise ValueError(f"Duplicate candidate chunk_id: {result.chunk_id}")
            identifiers.add(result.chunk_id)
            validated.append(result)
        observations = []
        for candidate in validated:
            raw_score = self.scorer.score(query, candidate.model_copy(deep=True))
            if isinstance(raw_score, bool) or not isinstance(raw_score, Real):
                raise ValueError("Reranking scores must be finite real numbers")
            try:
                score = float(raw_score)
            except (ValueError, OverflowError) as error:
                raise ValueError("Reranking scores must be finite real numbers") from error
            if not math.isfinite(score):
                raise ValueError("Reranking scores must be finite real numbers")
            observations.append(RerankedResult(result=candidate, reranking_score=score))
        return sorted(observations, key=lambda observation: (-observation.reranking_score, observation.result.chunk_id))[:top_k]