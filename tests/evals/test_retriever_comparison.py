"""Isolated A/B evaluation: python -m pytest tests/evals/test_retriever_comparison.py -q.

Reuses the frozen evaluator without changing its assertions or expected labels.
The vector provider below is evaluation-only TF-IDF, not a semantic model.
"""

import hashlib
import json
import math
import re
import socket
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from markdown_it import MarkdownIt

from agent.rag.contracts import DocumentChunk, RetrievalQuery, RetrievalResult
from agent.rag.hybrid_retriever import HybridRetriever, HybridRetrievalResult
from agent.rag.reranker import Reranker, RerankedResult, TokenJaccardScorer
from agent.rag.service import RAGService, STOP_WORDS
from agent.rag.vector_retriever import VectorRetriever
from agent.rag.vector_store import InMemoryVectorStore
from .test_rag_evals import CASES, TOP_K, EvaluationCase, _contains, _evaluate


ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "ci-results" / "retriever-comparison.md"


def _tokens(text: str) -> list[str]:
    return [term for term in re.findall(r"[^\W_]+", text.casefold()) if term not in STOP_WORDS]


class CorpusTfidfProvider:
    """Fixed corpus-only vocabulary and smoothed IDF; no queries or labels at fit time."""

    def __init__(self, texts: Sequence[str]) -> None:
        frequencies = Counter(term for text in texts for term in set(_tokens(text)))
        if not frequencies:
            raise ValueError("The comparison corpus must contain indexable terms")
        self.vocabulary = {term: index for index, term in enumerate(sorted(frequencies))}
        self.weights = {
            term: math.log((1 + len(texts)) / (1 + frequency)) + 1.0
            for term, frequency in frequencies.items()
        }
        self.dimensions = len(self.vocabulary) + 1

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for term, count in Counter(_tokens(text)).items():
            if term in self.vocabulary:
                vector[self.vocabulary[term]] = count * self.weights[term]
        if not any(vector):
            vector[-1] = 1.0
        return vector


class VectorEvaluationService(RAGService):
    """Test-only adapter retaining the same ingestion, generation and frozen checks."""

    def __init__(self, chunks: Sequence[DocumentChunk]) -> None:
        super().__init__()
        self.comparison_chunks = tuple(chunk.model_copy(deep=True) for chunk in chunks)
        provider = CorpusTfidfProvider([chunk.content for chunk in self.comparison_chunks])
        self.vector = VectorRetriever(provider, InMemoryVectorStore(provider.dimensions))

    def index_documents(self) -> int:
        count = super().index_documents()
        assert count == len(self.comparison_chunks)
        assert self.vector.index_chunks(self.comparison_chunks) == count
        return count

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        return self.vector.retrieve(query)


class HybridEvaluationService(VectorEvaluationService):
    """Keep both candidate retrievers independent of the shared answer generator."""

    def __init__(self, chunks: Sequence[DocumentChunk]) -> None:
        super().__init__(chunks)
        self.lexical = RAGService()
        self.hybrid = HybridRetriever(self.lexical, self.vector)

    def index_documents(self) -> int:
        count = super().index_documents()
        assert self.lexical.index_documents() == count
        return count

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        return self.hybrid.retrieve(query)

    def close(self) -> None:
        self.lexical.close()
        super().close()


class RerankedHybridEvaluationService(HybridEvaluationService):
    """Project active scores only at the frozen evaluator's result boundary."""

    def __init__(self, chunks: Sequence[DocumentChunk]) -> None:
        super().__init__(chunks)
        self.reranker = Reranker(TokenJaccardScorer(stop_words=STOP_WORDS))

    def candidate_results(self, query: RetrievalQuery) -> list[RetrievalResult]:
        return self.hybrid.retrieve(query.model_copy(update={
            "top_k": max(query.top_k, self.hybrid.candidate_top_k),
        }, deep=True))

    def reranked_results(self, query: RetrievalQuery) -> list[RerankedResult]:
        return self.reranker.rerank(query.query, self.candidate_results(query), top_k=query.top_k)

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        return [
            observation.result.model_copy(update={"score": observation.reranking_score}, deep=True)
            for observation in self.reranked_results(query)
        ]


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Retriever comparison must not access the network")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)


def _protected_hashes() -> dict[str, str]:
    paths = set((ROOT / "agent" / "rag").glob("*.py"))
    paths.update(path for path in (ROOT / "knowledge").rglob("*") if path.is_file())
    paths.update([
        ROOT / "tests/evals/rag-evals.json", ROOT / "tests/evals/test_rag_evals.py",
        ROOT / "api/main.py", ROOT / "api/rag.py", ROOT / "ci-results/rag-baseline.xml",
        ROOT / "ci-results/rag-ranking-candidate.xml",
    ])
    return {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def _expectations(case: EvaluationCase, report: dict[str, Any]) -> set[str]:
    satisfied = set()
    if bool(report["retrieved_chunk_ids"]) == case.should_retrieve:
        satisfied.add("should_retrieve")
    if report["insufficient_context_status"] == case.expected_insufficient_context:
        satisfied.add("insufficient_context")
    if report["actual_should_answer"] == case.should_answer:
        satisfied.add("should_answer")
    for source in case.expected_sources:
        if source in report["actual_sources"]:
            satisfied.add(f"source:{source}")
    for concept in case.expected_concepts:
        if concept not in report["missing_concepts"]:
            satisfied.add(f"concept:{concept}")
    for concept in case.forbidden_concepts:
        if concept not in report["forbidden_concepts_found"]:
            satisfied.add(f"forbidden-absent:{concept}")
    if not any(failure["layer"] in {"Retrieval", "Ranking"} for failure in report["failures"]):
        satisfied.add("retrieval_and_ranking")
    if not report["failures"]:
        satisfied.add("all_frozen_checks")
    return satisfied


def _better(lexical: set[str], vector: set[str]) -> str:
    if lexical > vector:
        return "Lexical"
    if vector > lexical:
        return "Vector"
    return "Tie on frozen expectations" if lexical == vector else "Mixed (trade-off)"


def _summary(cases: Sequence[EvaluationCase], reports: Sequence[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    source_pairs = [(case, report) for case, report in zip(cases, reports, strict=True) if case.expected_sources]
    concept_pairs = [(case, report) for case, report in zip(cases, reports, strict=True) if case.expected_concepts]
    count = len(cases)
    return {
        "Cases evaluated": (count, count),
        "Expected sources retrieved (all per case)": (
            sum(set(case.expected_sources) <= set(report["actual_sources"]) for case, report in source_pairs), len(source_pairs),
        ),
        "Expected concepts retrieved (all per case)": (
            sum(not report["missing_concepts"] for case, report in concept_pairs), len(concept_pairs),
        ),
        "Ranking success (frozen retrieval/ranking checks)": (
            sum(not any(failure["layer"] in {"Retrieval", "Ranking"} for failure in report["failures"]) for report in reports), count,
        ),
        "Insufficient-context correctness": (
            sum(case.expected_insufficient_context == report["insufficient_context_status"] for case, report in zip(cases, reports, strict=True)), count,
        ),
        "Retrieval failures (first failure layer)": (
            sum(report["likely_failure_layer"] == "Retrieval" for report in reports), count,
        ),
        "Ranking failures (first failure layer)": (
            sum(report["likely_failure_layer"] == "Ranking" for report in reports), count,
        ),
        "Pass rate (all frozen checks)": (sum(not report["failures"] for report in reports), count),
    }


def _text(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _items(values: Sequence[str]) -> str:
    return "; ".join(_text(value) for value in values) or "None"


def _render(
    cases: Sequence[EvaluationCase], lexical: Sequence[dict[str, Any]], vector: Sequence[dict[str, Any]],
    chunks: dict[str, DocumentChunk], source_by_chunk: dict[str, str], protected: dict[str, str],
    provider_dimensions: int,
) -> str:
    lines = [
        "# Retriever Strategy Comparison", "", "## Configuration", "",
        "- Dataset: `tests/evals/rag-evals.json`; every validated case is eligible; no category exclusions.",
        f"- Identical queries, expected sources/concepts, insufficient-context expectations, top_k={TOP_K}, filters={{}}.",
        f"- Same {len(chunks)} immutable chunks, identities and source mapping for both strategies.",
        "- A: current local lexical `RAGService.retrieve`, unchanged, positive weighted FTS5 BM25 scores.",
        f"- B: opt-in `VectorRetriever` + `InMemoryVectorStore`, {provider_dimensions} dimensions, unchanged.",
        "- B uses this harness's corpus-only TF-IDF provider: sorted vocabulary; raw term counts; IDF=log((1+N)/(1+df))+1; cosine similarity.",
        "- Tokenization: case-folded Unicode word tokens, existing STOP_WORDS removed; full rendered chunk text; no stemming, synonyms or label input.",
        "- Unknown query terms are ignored; an all-unknown/empty-token input uses a reserved coordinate absent from documents, giving zero cosine.",
        "- Vector minimum relevance remains its default 0.0 (inclusive), including orthogonal candidates. No cutoff calibration or tuning.",
        "- Both use the same unmodified extractive generator and frozen evaluator. Insufficiency is measured from its actual output, not inferred from golden labels.",
        "- Expected concepts use the frozen whole-phrase, case/whitespace-normalized checks. Extractive answers contain exactly selected chunk content, so answer concept coverage equals retrieved concept coverage.",
        "- Ranking success means all frozen Retrieval/Ranking checks pass, including source/concept coverage, relevance, positive scores, stable order and candidate-prefix checks.",
        "- Retrieval/Ranking failure totals use the frozen evaluator's first-failure label; other failure layers remain visible below.",
        "- Source/concept metrics exclude cases with no corresponding expectations from those denominators only; all cases remain in pass-rate/insufficiency metrics.",
        "- Better strategy uses strict set dominance over satisfied golden expectations/checks; crossed outcomes are mixed, equal outcomes tie. Scores never select a winner.",
        "- No network, credentials, model downloads, semantic model or production-default changes. TF-IDF is a lexical vector representation: this is not evidence about learned semantic retrieval.",
        "", "## Metrics", "", "| Metric | Lexical | Vector |", "| --- | ---: | ---: |",
    ]
    lexical_summary, vector_summary = _summary(cases, lexical), _summary(cases, vector)
    for metric, (lexical_count, denominator) in lexical_summary.items():
        vector_count, vector_denominator = vector_summary[metric]
        assert denominator == vector_denominator
        suffix_lexical = f" ({lexical_count / denominator:.2%})" if denominator else " (N/A)"
        suffix_vector = f" ({vector_count / denominator:.2%})" if denominator else " (N/A)"
        lines.append(f"| {metric} | {lexical_count}/{denominator}{suffix_lexical} | {vector_count}/{denominator}{suffix_vector} |")
    lines.extend(["", "## Per-Case Outcomes", "", "| Evaluation ID | Lexical | Vector | Better on expectations |", "| --- | --- | --- | --- |"])
    decisions = Counter()
    differing = []
    for case, lexical_report, vector_report in zip(cases, lexical, vector, strict=True):
        decision = _better(_expectations(case, lexical_report), _expectations(case, vector_report))
        decisions[decision] += 1
        lexical_status = lexical_report["likely_failure_layer"] or "PASS"
        vector_status = vector_report["likely_failure_layer"] or "PASS"
        lines.append(f"| {case.id} | {lexical_status} | {vector_status} | {decision} |")
        if lexical_report != vector_report:
            differing.append((case, lexical_report, vector_report, decision))
    lines.extend(["", "## Interpretation", ""])
    lines.append("Per-case golden-expectation dominance: " + "; ".join(f"{name}: {decisions[name]}" for name in ("Lexical", "Vector", "Tie on frozen expectations", "Mixed (trade-off)")) + ".")
    lines.extend([
        "Raw BM25 magnitudes and cosine scores are not comparable. These results describe only this frozen corpus, evaluator and explicit TF-IDF configuration; they do not establish an embedding-model winner.",
        "The existing context-free forbidden-phrase checks and all other assertions are retained, even for warnings quoting forbidden phrases. No failure changes the dataset, cutoff or implementations.",
        "", f"## Differing Results ({len(differing)} Cases)", "",
    ])
    for case, lexical_report, vector_report, decision in differing:
        lines.extend([
            f"### {case.id}", "", f"Question: {_text(case.question)}", "",
            f"Expected sources: {_items(case.expected_sources)}", "",
            f"Expected concepts: {_items(case.expected_concepts)}", "",
            f"Expected insufficient context: {case.expected_insufficient_context}", "",
            f"Better satisfies frozen expectations: **{decision}**", "",
            "| Rank | Lexical source / chunk | Lexical score | Vector source / chunk | Vector score |",
            "| ---: | --- | ---: | --- | ---: |",
        ])
        for position in range(max(len(lexical_report["retrieved_chunk_ids"]), len(vector_report["retrieved_chunk_ids"]))):
            cells = []
            for report in (lexical_report, vector_report):
                if position < len(report["retrieved_chunk_ids"]):
                    identifier = report["retrieved_chunk_ids"][position]
                    cells.extend([f"{_text(source_by_chunk[identifier])} / `{identifier}`", repr(report["retrieval_scores"][position])])
                else:
                    cells.extend(["None", "N/A"])
            lines.append(f"| {position + 1} | " + " | ".join(cells) + " |")
        for name, report in (("Lexical", lexical_report), ("Vector", vector_report)):
            lines.extend([
                "", f"{name} missing concepts: {_items(report['missing_concepts'])}", "",
                f"{name} actual insufficient context: {report['insufficient_context_status']}", "",
                f"{name} failures: " + (_items([f"{failure['layer']}: {failure['message']}" for failure in report["failures"]]) if report["failures"] else "None"),
            ])
        lines.append("")
    lines.extend(["## Integrity", "", f"All {len(protected)} protected files matched their pre-evaluation SHA-256 fingerprints after both runs:", "", "| Protected path | SHA-256 |", "| --- | --- |"])
    lines.extend(f"| {path} | `{digest}` |" for path, digest in protected.items())
    lines.extend(["", "Reproduce: `python -B -m pytest tests/evals/test_retriever_comparison.py -q -p no:cacheprovider`", ""])
    return "\n".join(lines)


def test_retriever_comparison() -> None:
    protected = _protected_hashes()
    cases = [EvaluationCase.model_validate(raw) for raw in CASES]
    assert cases and len({case.id for case in cases}) == len(cases)
    lexical = RAGService()
    vector = None
    try:
        documents = lexical.loader.load_documents()
        source_by_document = {document.document_id: str(document.metadata["relative_path"]) for document in documents}
        chunk_list = [chunk for document in documents for chunk in lexical.chunker.chunk_document(document)]
        chunks = {chunk.chunk_id: chunk for chunk in chunk_list}
        assert len(chunks) == len(chunk_list)
        assert all(set(case.expected_sources) <= set(source_by_document.values()) for case in cases)
        source_by_chunk = {chunk.chunk_id: source_by_document[chunk.document_id] for chunk in chunk_list}
        vector = VectorEvaluationService(chunk_list)
        lexical_reports = [_evaluate(case.model_dump(), lexical) for case in cases]
        vector_reports = [_evaluate(case.model_dump(), vector) for case in cases]
        for service, reports in ((lexical, lexical_reports), (vector, vector_reports)):
            assert len(reports) == len(cases)
            for case, report in zip(cases, reports, strict=True):
                assert "evaluation_error" not in report, report
                assert report["retrieval_executed"] and not report["retrieval_errors"], report
                assert report["success"] and report["error"] is None, report
                assert report["top_k"] == TOP_K
                assert all(identifier in chunks for identifier in report["retrieved_chunk_ids"])
                repeated = _evaluate(case.model_dump(), service)
                assert repeated == report, f"Nondeterministic comparison result: {case.id}"
        assert _protected_hashes() == protected, "The comparison modified a protected input or implementation"
        report_text = _render(cases, lexical_reports, vector_reports, chunks, source_by_chunk, protected, vector.vector.embedding_provider.dimensions)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report_text, encoding="utf-8")
    finally:
        lexical.close()
        if vector is not None:
            vector.close()
        assert _protected_hashes() == protected


def _render_hybrid_comparison(
    cases: Sequence[EvaluationCase], reports: dict[str, list[dict[str, Any]]],
    observations: dict[str, list[HybridRetrievalResult]], source_by_chunk: dict[str, str],
    protected: dict[str, str], hybrid: HybridRetriever,
) -> str:
    names = list(reports)
    lines = [
        "# Hybrid Retrieval Comparison", "", "## Configuration", "",
        "- Dataset: frozen `tests/evals/rag-evals.json`; every validated case is eligible; no exclusions or changed assertions.",
        f"- Same {len(cases)} cases, {len(source_by_chunk)} chunks, queries, source/concept expectations, insufficient-context expectations, top_k={TOP_K}, filters={{}}.",
        "- Lexical: unchanged weighted FTS5 BM25 `RAGService.retrieve`.",
        "- Vector: unchanged `VectorRetriever` with the original comparison's corpus-only TF-IDF provider and default minimum cosine 0.0.",
        "- TF-IDF: sorted corpus vocabulary, raw term counts, IDF=log((1+N)/(1+df))+1; existing STOP_WORDS; no query or golden-label fitting, synonyms, stemming or learned semantic embeddings.",
        "- Unknown query tokens are ignored; all-unknown inputs use the existing reserved orthogonal coordinate. Vector's inclusive default cutoff can return zero-score chunks.",
        f"- Hybrid fixed global configuration: lexical_weight={hybrid.lexical_weight}, vector_weight={hybrid.vector_weight}, candidate_top_k={hybrid.candidate_top_k}, min_relevance={hybrid.min_relevance} (inclusive).",
        "- Each strategy is independently queried for max(candidate_top_k, requested top_k) candidates; chunk-ID duplicates use the maximum raw score within each component.",
        "- Normalization: normalized_component=max(0, raw_score)/max_positive_score_in_that_component_pool; empty/nonpositive pools and missing components contribute zero.",
        "- Combined score is the weighted sum of normalized components, then cutoff, descending combined score / ascending chunk-ID ordering, then top_k.",
        "- No question-specific weights, score calibration, evaluation-ID/source special cases, external calls or production-default changes. All candidates originate from a child retriever.",
        "- All three use the same unchanged frozen evaluator and extractive generator. Insufficiency is the actual generated flag, not a golden-label-based override.",
        "- Source/concept success requires all expectations per case; denominators exclude empty expectation lists only for those two metrics. Concept checks use frozen whole phrases in the extractive evidence.",
        "- Ranking success means all frozen Retrieval/Ranking checks pass, including source/concept coverage, relevance, positive scores, stable ties and full-candidate prefix.",
        "- Pass rates require every frozen check, including grounding, citations and forbidden concepts. Harness checks passing do not mean golden cases all passed.",
        "- Weights/settings were fixed before this run; no retriever, dataset, knowledge document or expectation was optimized after observing results.",
        "", "## Metrics", "", "| Metric | " + " | ".join(names) + " |",
        "| --- | " + " | ".join("---:" for name in names) + " |",
    ]
    summaries = {name: _summary(cases, strategy_reports) for name, strategy_reports in reports.items()}
    for metric in summaries[names[0]]:
        cells = []
        for name in names:
            count, denominator = summaries[name][metric]
            cells.append(f"{count}/{denominator} ({count / denominator:.2%})" if denominator else "N/A")
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    layers = sorted({failure["layer"] for strategy_reports in reports.values() for report in strategy_reports for failure in report["failures"]})
    lines.extend(["", "## Failures by Layer", "", "First-failure counts are disjoint, preserving the frozen evaluator's check order:", "", "| Layer | " + " | ".join(names) + " |", "| --- | " + " | ".join("---:" for name in names) + " |"])
    for layer in layers:
        lines.append(f"| {layer} | " + " | ".join(str(sum(report["likely_failure_layer"] == layer for report in reports[name])) for name in names) + " |")
    lines.extend(["", "Cases with any failure in each layer (overlapping; not an additive total):", "", "| Layer | " + " | ".join(names) + " |", "| --- | " + " | ".join("---:" for name in names) + " |"])
    for layer in layers:
        lines.append(f"| {layer} | " + " | ".join(str(sum(any(failure["layer"] == layer for failure in report["failures"]) for report in reports[name])) for name in names) + " |")
    lines.extend(["", "## Interpretation", "", "Compare frozen expectations and pass rates, never raw BM25 versus cosine versus hybrid score magnitudes. This is a TF-IDF vector configuration, not a test of learned semantic embeddings.", "The inclusive zero cutoff can return zero-relevance candidates; the frozen positive-score and no-answer checks still fail where applicable. Existing extractive sufficiency behavior and context-free forbidden-phrase checks are unchanged.", "", "## Per-Case Outcomes", "", "| Evaluation ID | " + " | ".join(names) + " |", "| --- | " + " | ".join("---" for name in names) + " |"])
    for position, case in enumerate(cases):
        lines.append(f"| {case.id} | " + " | ".join(reports[name][position]["likely_failure_layer"] or "PASS" for name in names) + " |")
    lines.extend(["", "## Evidence and Component Scores", ""])
    for position, case in enumerate(cases):
        lines.extend([
            f"### {case.id}", "", f"Question: {_text(case.question)}", "",
            f"Expected sources: {_items(case.expected_sources)}", "",
            f"Expected concepts: {_items(case.expected_concepts)}", "",
            f"Expected insufficient context: {case.expected_insufficient_context}", "",
            "| Strategy | Actual sources | Scores in returned chunk order | Missing concepts | Actual insufficient context | Failures |",
            "| --- | --- | --- | --- | --- | --- |",
        ])
        for name in names:
            report = reports[name][position]
            sources = [source_by_chunk[identifier] for identifier in report["retrieved_chunk_ids"]]
            failures = [f"{failure['layer']}: {failure['message']}" for failure in report["failures"]]
            lines.append(f"| {name} | {_items(sources)} | {_items([repr(score) for score in report['retrieval_scores']])} | {_items(report['missing_concepts'])} | {report['insufficient_context_status']} | {_items(failures)} |")
        lines.extend(["", "Hybrid component scores for selected chunks (None means absent from that component pool):", "", "| Chunk ID | Lexical raw | Vector raw | Lexical normalized | Vector normalized | Lexical contribution | Vector contribution | Combined |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for observation in observations[case.id]:
            lines.append(f"| `{observation.result.chunk_id}` | {observation.lexical_raw_score!r} | {observation.vector_raw_score!r} | {observation.lexical_normalized_score!r} | {observation.vector_normalized_score!r} | {observation.lexical_contribution!r} | {observation.vector_contribution!r} | {observation.result.score!r} |")
        lines.append("")
    lines.extend(["## Integrity", "", f"All {len(protected)} protected fingerprints matched before and after the repeated comparison, including the original A/B report:", "", "| Protected path | SHA-256 |", "| --- | --- |"])
    lines.extend(f"| {path} | `{digest}` |" for path, digest in protected.items())
    lines.extend(["", "Reproduce after unit tests: `python -B -m pytest tests/evals/test_retriever_comparison.py::test_hybrid_retriever_comparison -q -p no:cacheprovider`", ""])
    return "\n".join(lines)


def test_hybrid_retriever_comparison() -> None:
    def fingerprints() -> dict[str, str]:
        return {
            **_protected_hashes(),
            REPORT_PATH.relative_to(ROOT).as_posix(): hashlib.sha256(REPORT_PATH.read_bytes()).hexdigest(),
        }

    protected = fingerprints()
    cases = [EvaluationCase.model_validate(raw) for raw in CASES]
    assert cases and len({case.id for case in cases}) == len(cases)
    lexical = RAGService()
    services: dict[str, RAGService] = {"Lexical": lexical}
    try:
        documents = lexical.loader.load_documents()
        source_by_document = {document.document_id: str(document.metadata["relative_path"]) for document in documents}
        chunk_list = [chunk for document in documents for chunk in lexical.chunker.chunk_document(document)]
        chunks = {chunk.chunk_id: chunk for chunk in chunk_list}
        assert len(chunks) == len(chunk_list)
        assert all(set(case.expected_sources) <= set(source_by_document.values()) for case in cases)
        source_by_chunk = {chunk.chunk_id: source_by_document[chunk.document_id] for chunk in chunk_list}
        vector, hybrid = VectorEvaluationService(chunk_list), HybridEvaluationService(chunk_list)
        services.update(Vector=vector, Hybrid=hybrid)
        reports: dict[str, list[dict[str, Any]]] = {}
        observations: dict[str, list[HybridRetrievalResult]] = {}
        for name, service in services.items():
            strategy_reports = []
            for case in cases:
                report = _evaluate(case.model_dump(), service)
                assert "evaluation_error" not in report, report
                assert report["retrieval_executed"] and not report["retrieval_errors"], report
                assert report["success"] and report["error"] is None, report
                assert report["top_k"] == TOP_K
                assert all(identifier in chunks for identifier in report["retrieved_chunk_ids"])
                assert _evaluate(case.model_dump(), service) == report, f"Nondeterministic {name} result: {case.id}"
                if name == "Hybrid":
                    request = RetrievalQuery(query=case.question, top_k=TOP_K, filters={})
                    selected = hybrid.hybrid.retrieve_with_scores(request)
                    assert [item.result.chunk_id for item in selected] == report["retrieved_chunk_ids"]
                    assert [item.result.score for item in selected] == report["retrieval_scores"]
                    candidate_request = request.model_copy(update={"top_k": max(TOP_K, hybrid.hybrid.candidate_top_k)})
                    candidates = hybrid.lexical.retrieve(candidate_request) + hybrid.vector.retrieve(candidate_request)
                    assert {item.result.chunk_id for item in selected} <= {item.chunk_id for item in candidates}
                    observations[case.id] = selected
                strategy_reports.append(report)
            assert len(strategy_reports) == len(cases)
            reports[name] = strategy_reports
        assert fingerprints() == protected
        report_text = _render_hybrid_comparison(cases, reports, observations, source_by_chunk, protected, hybrid.hybrid)
        path = ROOT / "ci-results/hybrid-retrieval-comparison.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report_text, encoding="utf-8")
    finally:
        for service in services.values():
            service.close()
        assert fingerprints() == protected


def _archived_hybrid_outcomes(cases: Sequence[EvaluationCase]) -> dict[str, str]:
    path = ROOT / "ci-results/hybrid-retrieval-comparison.md"
    tokens = MarkdownIt("commonmark").enable("table").parse(path.read_text(encoding="utf-8"))
    identifiers = {case.id for case in cases}
    outcomes = {}
    cells: list[str] = []
    for token in tokens:
        if token.type == "tr_open":
            cells = []
        elif token.type == "inline":
            cells.append(token.content)
        elif token.type == "tr_close" and cells and cells[0] in identifiers:
            assert cells[0] not in outcomes
            outcomes[cells[0]] = cells[-1]
    assert set(outcomes) == identifiers
    return outcomes


def _render_reranking_comparison(
    cases: Sequence[EvaluationCase], before: list[dict[str, Any]], after: list[dict[str, Any]],
    observations: dict[str, list[RerankedResult]], pools: dict[str, tuple[int, str]],
    source_by_chunk: dict[str, str], archived: dict[str, str], protected: dict[str, str],
) -> str:
    archived_passed = sum(status == "PASS" for status in archived.values())
    after_passed = sum(not report["failures"] for report in after)
    regressions = [case.id for case, previous, current in zip(cases, before, after, strict=True) if not previous["failures"] and current["failures"]]
    improvements = [case.id for case, previous, current in zip(cases, before, after, strict=True) if previous["failures"] and not current["failures"]]
    lines = [
        "# Reranking Comparison", "", "## Configuration", "",
        f"- Same {len(cases)} frozen cases from `tests/evals/rag-evals.json`; final top_k={TOP_K}, filters={{}}; no case exclusions or modified expectations/assertions.",
        f"- Previous archived Hybrid pass rate: **{archived_passed}/{len(cases)} ({archived_passed / len(cases):.2%})**. Current baseline reproduces every archived case's PASS/first-failure outcome.",
        f"- Reranked pass rate: **{after_passed}/{len(cases)} ({after_passed / len(cases):.2%})**.",
        "- Hybrid remains unchanged: equal 0.5/0.5 weights, candidate_top_k=50 per child, inclusive min_relevance=0.0; corpus-only TF-IDF vector provider, not a learned semantic model.",
        "- Both arms use the identical hybrid candidate pool (request top_k=max(50, final top_k)); baseline keeps its leading three, the other arm reranks that pool before keeping three. No additional retriever or expanded pool is used for reranking.",
        "- Fixed local scorer: Jaccard similarity between case-folded Unicode query/content token sets after existing STOP_WORDS removal; intersection size / union size; empty union scores zero.",
        "- No synonyms, stemming, heading/source boosts, label input, learned weights, threshold calibration, external services or tuning after results.",
        "- Reranker accepts only supplied RetrievalResult candidates; selected chunks retain identical IDs, source metadata, content and original hybrid scores. Ties use ascending chunk ID.",
        "- Original hybrid scores remain in RerankedResult.result.score / retrieval_score; reranking scores are separate. The test-only service projects reranking_score into the frozen evaluator's generic score field, so its unchanged descending-active-score/prefix checks evaluate the actual final ordering.",
        "- Both arms retain the same extractive generator, insufficient-context behavior, whole-phrase concepts, forbidden phrases and citation checks. No generator or abstention changes mask retrieval/grounding failures.",
        "- Network and DNS are blocked. Existing retrievers, knowledge, dataset, frozen evaluator, defaults and previous reports are fingerprint-protected. Results are repeated for determinism.",
        "", "## Metrics", "", "| Metric | Hybrid before | Hybrid plus reranking |", "| --- | ---: | ---: |",
    ]
    before_summary, after_summary = _summary(cases, before), _summary(cases, after)
    for metric, (count, denominator) in before_summary.items():
        after_count, after_denominator = after_summary[metric]
        assert denominator == after_denominator
        previous = f"{count}/{denominator} ({count / denominator:.2%})" if denominator else "N/A"
        current = f"{after_count}/{denominator} ({after_count / denominator:.2%})" if denominator else "N/A"
        lines.append(f"| {metric} | {previous} | {current} |")
    layers = sorted({"Ranking", "Retrieval", "Grounding"} | {
        failure["layer"] for report in before + after for failure in report["failures"]
    })
    lines.extend(["", "## Failures by Layer", "", "First-failure classification (disjoint, preserving frozen check order):", "", "| Layer | Before | After |", "| --- | ---: | ---: |"])
    for layer in layers:
        lines.append(f"| {layer} | {sum(report['likely_failure_layer'] == layer for report in before)} | {sum(report['likely_failure_layer'] == layer for report in after)} |")
    lines.extend(["", "Any failure in each layer (overlapping, not additive):", "", "| Layer | Before | After |", "| --- | ---: | ---: |"])
    for layer in layers:
        lines.append(f"| {layer} | {sum(any(failure['layer'] == layer for failure in report['failures']) for report in before)} | {sum(any(failure['layer'] == layer for failure in report['failures']) for report in after)} |")
    lines.extend([
        "", "## Regressions and Improvements", "",
        f"Pass-to-fail regressions ({len(regressions)}): {_items(regressions)}", "",
        f"Fail-to-pass improvements ({len(improvements)}): {_items(improvements)}", "",
        "Lost satisfied frozen expectations are also reported below, including cases already failing before reranking. A changed first-failure label alone is not treated as an improvement.", "",
        "| Evaluation ID | Before | After | Lost satisfied checks | Gained satisfied checks |",
        "| --- | --- | --- | --- | --- |",
    ])
    for case, previous, current in zip(cases, before, after, strict=True):
        previous_checks, current_checks = _expectations(case, previous), _expectations(case, current)
        lines.append(f"| {case.id} | {previous['likely_failure_layer'] or 'PASS'} | {current['likely_failure_layer'] or 'PASS'} | {_items(sorted(previous_checks - current_checks))} | {_items(sorted(current_checks - previous_checks))} |")
    lines.extend(["", "## Per-Case Evidence", ""])
    for case, previous, current in zip(cases, before, after, strict=True):
        count, digest = pools[case.id]
        lines.extend([
            f"### {case.id}", "", f"Question: {_text(case.question)}", "",
            f"Expected sources: {_items(case.expected_sources)}", "",
            f"Expected concepts: {_items(case.expected_concepts)}", "",
            f"Expected insufficient context: {case.expected_insufficient_context}", "",
            f"Identical candidate pool: {count} chunks; SHA-256 `{digest}` (ordered complete RetrievalResult JSON, including original scores).", "",
            "| Arm | Sources in returned order | Active scores | Missing concepts | Actual insufficient context | Failures |",
            "| --- | --- | --- | --- | --- | --- |",
        ])
        for name, report in (("Hybrid", previous), ("Hybrid + reranking", current)):
            failures = [f"{failure['layer']}: {failure['message']}" for failure in report["failures"]]
            sources = [source_by_chunk[identifier] for identifier in report["retrieved_chunk_ids"]]
            lines.append(f"| {name} | {_items(sources)} | {_items([repr(score) for score in report['retrieval_scores']])} | {_items(report['missing_concepts'])} | {report['insufficient_context_status']} | {_items(failures)} |")
        lines.extend(["", "| Selected chunk ID | Original hybrid retrieval score | Separate reranking score |", "| --- | ---: | ---: |"])
        for observation in observations[case.id]:
            lines.append(f"| `{observation.result.chunk_id}` | {observation.retrieval_score!r} | {observation.reranking_score!r} |")
        lines.append("")
    lines.extend(["## Interpretation", "", "Pass/fail and regressions are determined only by frozen expectations, not score magnitudes. This deterministic token-overlap reranker is a local baseline, not a semantic or cross-encoder model. No optimization was performed after observing these results.", "", "## Integrity", "", f"All {len(protected)} protected fingerprints matched before and after the repeated comparison:", "", "| Protected path | SHA-256 |", "| --- | --- |"])
    lines.extend(f"| {path} | `{digest}` |" for path, digest in protected.items())
    lines.extend(["", "Reproduce after unit tests: `python -B -m pytest tests/evals/test_retriever_comparison.py::test_reranking_comparison -q -p no:cacheprovider`", ""])
    return "\n".join(lines)


def test_reranking_comparison() -> None:
    def fingerprints() -> dict[str, str]:
        paths = [REPORT_PATH, ROOT / "ci-results/hybrid-retrieval-comparison.md"]
        return {
            **_protected_hashes(),
            **{path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        }

    protected = fingerprints()
    cases = [EvaluationCase.model_validate(raw) for raw in CASES]
    assert cases and len({case.id for case in cases}) == len(cases)
    archived = _archived_hybrid_outcomes(cases)
    loader_service = RAGService()
    services: list[RAGService] = [loader_service]
    try:
        documents = loader_service.loader.load_documents()
        source_by_document = {document.document_id: str(document.metadata["relative_path"]) for document in documents}
        chunk_list = [chunk for document in documents for chunk in loader_service.chunker.chunk_document(document)]
        chunks = {chunk.chunk_id: chunk for chunk in chunk_list}
        assert len(chunks) == len(chunk_list)
        assert all(set(case.expected_sources) <= set(source_by_document.values()) for case in cases)
        source_by_chunk = {chunk.chunk_id: source_by_document[chunk.document_id] for chunk in chunk_list}
        baseline, reranked = HybridEvaluationService(chunk_list), RerankedHybridEvaluationService(chunk_list)
        services.extend([baseline, reranked])
        before, after = [], []
        observations: dict[str, list[RerankedResult]] = {}
        pools: dict[str, tuple[int, str]] = {}
        for case in cases:
            reports = []
            for service in (baseline, reranked):
                report = _evaluate(case.model_dump(), service)
                assert "evaluation_error" not in report, report
                assert report["retrieval_executed"] and not report["retrieval_errors"], report
                assert report["success"] and report["error"] is None, report
                assert report["top_k"] == TOP_K
                assert _evaluate(case.model_dump(), service) == report, f"Nondeterministic result: {case.id}"
                reports.append(report)
            previous, current = reports
            assert (previous["likely_failure_layer"] or "PASS") == archived[case.id], f"Archived hybrid outcome changed: {case.id}"
            request = RetrievalQuery(query=case.question, top_k=TOP_K, filters={})
            candidate_request = request.model_copy(update={"top_k": max(TOP_K, baseline.hybrid.candidate_top_k)})
            candidates = baseline.hybrid.retrieve(candidate_request)
            assert candidates == reranked.candidate_results(request), f"Different candidate pools: {case.id}"
            selected = reranked.reranked_results(request)
            by_id = {candidate.chunk_id: candidate for candidate in candidates}
            assert [item.result.chunk_id for item in selected] == current["retrieved_chunk_ids"]
            assert [item.reranking_score for item in selected] == current["retrieval_scores"]
            assert all(item.result == by_id[item.result.chunk_id] for item in selected)
            assert all(item.retrieval_score == by_id[item.result.chunk_id].score for item in selected)
            assert [item.chunk_id for item in candidates[:TOP_K]] == previous["retrieved_chunk_ids"]
            payload = json.dumps([candidate.model_dump() for candidate in candidates], sort_keys=True, ensure_ascii=True).encode()
            pools[case.id] = (len(candidates), hashlib.sha256(payload).hexdigest())
            observations[case.id] = selected
            before.append(previous)
            after.append(current)
        assert fingerprints() == protected
        report_text = _render_reranking_comparison(cases, before, after, observations, pools, source_by_chunk, archived, protected)
        path = ROOT / "ci-results/reranking-comparison.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report_text, encoding="utf-8")
    finally:
        for service in services:
            service.close()
        assert fingerprints() == protected


@pytest.mark.parametrize(("lexical", "vector", "expected"), [
    ({"source:a", "concept:b"}, {"source:a"}, "Lexical"),
    ({"source:a"}, {"source:a", "concept:b"}, "Vector"),
    ({"source:a"}, {"concept:b"}, "Mixed (trade-off)"),
    ({"source:a"}, {"source:a"}, "Tie on frozen expectations"),
])
def test_better_strategy_depends_only_on_satisfied_expectations(
    lexical: set[str], vector: set[str], expected: str,
) -> None:
    assert _better(lexical, vector) == expected


def test_comparison_provider_is_deterministic_and_has_no_query_fit() -> None:
    provider = CorpusTfidfProvider(["clock clock water", "paper"])
    before = (dict(provider.vocabulary), dict(provider.weights), provider.dimensions)
    vectors = provider.embed_documents(["clock clock water", "paper"])
    assert vectors == provider.embed_documents(["clock clock water", "paper"])
    assert vectors[0] == provider.embed_query("CLOCK clock water")
    assert provider.embed_query("unseen vocabulary")[-1] == 1.0
    assert all(value == 0.0 for value in provider.embed_query("unseen vocabulary")[:-1])
    assert before == (provider.vocabulary, provider.weights, provider.dimensions)