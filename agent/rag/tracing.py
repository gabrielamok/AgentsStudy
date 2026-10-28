import logging
import re
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Protocol
from uuid import uuid4

from agent.rag.contracts import RAGRequest, RAGResponse, RetrievalResult


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RAGTraceRecord:
    correlation_id: str
    normalized_query: str
    retrieval_strategy: str
    requested_top_k: int | None
    candidate_chunk_ids: tuple[str, ...]
    candidate_scores: tuple[float, ...]
    ranking_order: tuple[str, ...]
    final_chunk_ids: tuple[str, ...]
    final_scores: tuple[float, ...]
    grounded: bool | None
    insufficient_context: bool | None
    citation_chunk_ids: tuple[str, ...]
    processing_outcome: str
    failure_layer: str | None
    candidate_scope: str


class TraceSink(Protocol):
    def emit(self, record: RAGTraceRecord) -> None:
        ...


class LoggingTraceSink:
    def emit(self, record: RAGTraceRecord) -> None:
        logger.info("rag.trace", extra={"rag_trace": asdict(record)})


class TraceRedactor:
    """Never inspects environment configuration; callers register known secrets.

    Assignment values, labelled credentials, bearer tokens, URL credentials,
    absolute paths and long opaque query tokens are also redacted. Register
    application secret values explicitly; arbitrary unlabelled short strings
    cannot reliably be distinguished from ordinary language.
    """

    def __init__(self, secret_values: Sequence[str] = ()) -> None:
        if any(not isinstance(value, str) or not value for value in secret_values):
            raise ValueError("Registered secrets must be nonempty strings")
        self._secrets = tuple(sorted(secret_values, key=len, reverse=True))

    def redact(self, text: str, *, query: bool = False) -> str:
        for secret in self._secrets:
            text = re.sub(re.escape(secret), "[REDACTED]", text, flags=re.IGNORECASE)
        text = re.sub(r"\b[A-Za-z][A-Za-z0-9+.-]*://[^\s/@:]+:[^\s/@]+@", "[REDACTED]@", text)
        text = re.sub(r"\bbearer\s+[^\s,;]+", "Bearer [REDACTED]", text, flags=re.IGNORECASE)
        text = re.sub(
            r"\b(?:api[ _-]?key|password|secret|token|client[ _-]?secret|connection[ _-]?string)\s+(?:is\s+)?(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)",
            "credential [REDACTED]", text, flags=re.IGNORECASE,
        )
        text = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*[\"']?\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)", "[REDACTED]", text)
        text = re.sub(
            r"[A-Za-z]:[\\/][^\s\"<>]+|\\\\[^\s\\/]+[\\/][^\s\"<>]+|(?<![\w./:])/(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+",
            "[REDACTED_PATH]", text,
        )
        if query:
            text = re.sub(r"(?<!\w)[A-Za-z0-9_+/=-]{32,}(?!\w)", "[REDACTED]", text)
        return text


@dataclass
class _TraceState:
    correlation_id: str
    normalized_query: str
    retrieval_strategy: str
    requested_top_k: int | None
    candidates: list[tuple[str, float]] = field(default_factory=list)
    final: list[tuple[str, float]] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    grounded: bool | None = None
    insufficient_context: bool | None = None
    processing_outcome: str = "failure"
    failure_layer: str | None = "Query Processing"
    candidate_scope: str = "unavailable"

    def finish(self, response: RAGResponse) -> None:
        self.grounded = response.grounded
        self.insufficient_context = response.insufficient_context
        self.citations = [citation.chunk_id for citation in response.citations]
        if response.success:
            self.processing_outcome = "insufficient_context" if response.insufficient_context else "success"
            self.failure_layer = None


_CURRENT: ContextVar[_TraceState | None] = ContextVar("rag_trace", default=None)


def capture_candidates(results: Sequence[RetrievalResult], *, scope: str = "filtered_candidates") -> None:
    state = _CURRENT.get()
    if state is not None:
        state.candidates = [(result.chunk_id, result.score) for result in results]
        state.candidate_scope = scope


class RAGTracer:
    """One redacted record per query; context-local state and no timing dependency.

    Only IDs, scores, statuses and sanitized query/strategy are emitted: no chunk
    content, answer body, filters, exception messages or environment snapshots.
    Sink errors never change the RAG response. Inject ID generation for exact tests.
    """

    def __init__(
        self, sink: TraceSink | None = None, *,
        correlation_id_factory: Callable[[], str] | None = None,
        redactor: TraceRedactor | None = None,
    ) -> None:
        self.sink = sink if sink is not None else LoggingTraceSink()
        self.correlation_id_factory = correlation_id_factory if correlation_id_factory is not None else lambda: str(uuid4())
        self.redactor = redactor if redactor is not None else TraceRedactor()

    def correlation_id(self, supplied: str | None = None) -> str:
        identifier = supplied if supplied is not None else self.correlation_id_factory()
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("Correlation IDs must be nonempty strings")
        return self.redactor.redact(" ".join(identifier.split()))

    @contextmanager
    def workflow(
        self, request: RAGRequest, *, retrieval_strategy: str, correlation_id: str | None = None,
    ) -> Iterator[_TraceState]:
        state = _TraceState(
            correlation_id=self.correlation_id(correlation_id),
            normalized_query=self.redactor.redact(" ".join(request.query.casefold().split()), query=True),
            retrieval_strategy=self.redactor.redact(retrieval_strategy),
            requested_top_k=request.top_k if type(request.top_k) is int else None,
        )
        token = _CURRENT.set(state)
        try:
            yield state
        finally:
            _CURRENT.reset(token)
            clean = self.redactor.redact
            record = RAGTraceRecord(
                correlation_id=state.correlation_id, normalized_query=state.normalized_query,
                retrieval_strategy=state.retrieval_strategy, requested_top_k=state.requested_top_k,
                candidate_chunk_ids=tuple(clean(identifier) for identifier, score in state.candidates),
                candidate_scores=tuple(score for identifier, score in state.candidates),
                ranking_order=tuple(clean(identifier) for identifier, score in state.candidates),
                final_chunk_ids=tuple(clean(identifier) for identifier, score in state.final),
                final_scores=tuple(score for identifier, score in state.final),
                grounded=state.grounded, insufficient_context=state.insufficient_context,
                citation_chunk_ids=tuple(clean(identifier) for identifier in state.citations),
                processing_outcome=state.processing_outcome, failure_layer=state.failure_layer,
                candidate_scope=state.candidate_scope,
            )
            try:
                self.sink.emit(record)
            except Exception:
                logger.warning("RAG trace sink unavailable")