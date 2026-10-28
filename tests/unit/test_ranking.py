import socket
from collections.abc import Iterator
from pathlib import Path

import pytest

from agent.rag.contracts import RetrievalQuery
from agent.rag.service import RAGService


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Ranking tests must remain completely local")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.fixture
def service(tmp_path: Path) -> Iterator[RAGService]:
    instance = RAGService(tmp_path)
    yield instance
    instance.close()


def write_document(directory: Path, name: str, content: str) -> None:
    (directory / name).write_text(content, encoding="utf-8")


def test_body_terms_are_indexed_once_not_duplicated_as_full_content(
    service: RAGService, tmp_path: Path,
) -> None:
    write_document(tmp_path, "device.md", "# Timing\n\n## Oscillators\n\nA quartz crystal supplies the signal.")
    service.index_documents()
    assert service._index is not None
    service._index.execute("CREATE VIRTUAL TABLE term_instances USING fts5vocab(chunks, 'instance')")

    occurrences = service._index.execute(
        "SELECT count(*) FROM term_instances WHERE term = 'quartz'",
    ).fetchone()[0]

    assert occurrences == 1
    result = service.retrieve(RetrievalQuery(query="quartz", top_k=1, filters={}))[0]
    assert result.content == "# Timing\n## Oscillators\n\nA quartz crystal supplies the signal."


def test_exact_terminology_does_not_match_a_word_prefix(service: RAGService, tmp_path: Path) -> None:
    write_document(tmp_path, "exact.md", "# Timing\n\nA quartz oscillator supplies the signal.")
    write_document(tmp_path, "prefix.md", "# Materials\n\nQuartzite is a metamorphic rock.")

    results = service.retrieve(RetrievalQuery(query="quartz", top_k=3, filters={}))

    assert [result.source_title for result in results] == ["Timing"]


def test_repeated_important_body_terms_improve_rank_with_equal_length(
    service: RAGService, tmp_path: Path,
) -> None:
    write_document(tmp_path, "repeated.md", "# Reference\n\nquartz quartz quartz quartz")
    write_document(tmp_path, "single.md", "# Reference\n\nquartz ceramic ceramic ceramic")

    results = service.retrieve(RetrievalQuery(query="quartz", top_k=3, filters={}))

    assert len(results) == 2
    assert results[0].content.endswith("quartz quartz quartz quartz")
    assert results[0].score > results[1].score
    assert service.retrieve(RetrievalQuery(query="quartz quartz quartz", top_k=3, filters={})) == results


@pytest.mark.parametrize("heading", ["# Quartz", "# Reference\n\n## Quartz"])
def test_source_and_section_titles_remain_relevant(
    service: RAGService, tmp_path: Path, heading: str,
) -> None:
    write_document(tmp_path, "relevant.md", heading + "\n\nMaintenance procedures apply here.")
    write_document(tmp_path, "unrelated.md", "# Ceramic\n\nMaintenance procedures apply here.")

    results = service.retrieve(RetrievalQuery(query="quartz", top_k=3, filters={}))

    assert len(results) == 1
    assert results[0].section == "Quartz"
    assert results[0].score > 0


def test_ancestor_heading_context_is_not_lost(service: RAGService, tmp_path: Path) -> None:
    write_document(
        tmp_path, "nested.md", "# Reference\n\n## Quartz\n\n### Maintenance\n\nInspect the device annually.",
    )

    results = service.retrieve(RetrievalQuery(query="quartz", top_k=1, filters={}))

    assert len(results) == 1
    assert results[0].source_title == "Reference"
    assert results[0].section == "Maintenance"
    assert "## Quartz" in results[0].content


def test_irrelevant_high_frequency_terms_do_not_outweigh_focused_evidence(
    service: RAGService, tmp_path: Path,
) -> None:
    write_document(tmp_path, "focused.md", "# Reference\n\nquartz oscillator timing")
    write_document(tmp_path, "verbose.md", "# Reference\n\nquartz oscillator " + "ceramic " * 100)
    write_document(tmp_path, "unrelated.md", "# Materials\n\n" + "ceramic " * 100)

    results = service.retrieve(RetrievalQuery(query="quartz oscillator", top_k=10, filters={}))

    assert len(results) == 2
    assert results[0].content.endswith("quartz oscillator timing")
    assert results[0].score > results[1].score


@pytest.mark.parametrize("top_k", [1, 2, 3, 10])
def test_top_k_and_stable_ties_survive_rebuilds(
    service: RAGService, tmp_path: Path, top_k: int,
) -> None:
    content = "# Timing\n\nA quartz oscillator supplies the signal."
    for name in ("third.md", "first.md", "second.md"):
        write_document(tmp_path, name, content)
    full = service.retrieve(RetrievalQuery(query="quartz", top_k=10, filters={}))
    query = RetrievalQuery(query="quartz", top_k=top_k, filters={})

    limited = service.retrieve(query)
    repeated = service.retrieve(query)
    service.index_documents()
    rebuilt = service.retrieve(query)

    assert len(full) == 3
    assert len({result.score for result in full}) == 1
    assert [result.chunk_id for result in full] == sorted(result.chunk_id for result in full)
    assert limited == repeated == rebuilt == full[:top_k]


def test_scores_preserve_exact_bm25_values_and_source_identity(
    service: RAGService, tmp_path: Path,
) -> None:
    write_document(tmp_path, "device.md", "# Timing\n\n## Oscillators\n\nA quartz crystal supplies the signal.")
    query = RetrievalQuery(query="quartz", top_k=3, filters={})
    results = service.retrieve(query)
    assert service._index is not None
    columns = service._index.execute("PRAGMA table_info(chunks)").fetchall()
    weights = {"source_title": 0.25, "section": 0.25, "content": 0.25, "body": 2.0, "heading_context": 0.25}
    arguments = ", ".join(str(weights.get(column[1], 0.0)) for column in columns)
    rows = service._index.execute(
        f"SELECT chunk_id, bm25(chunks, {arguments}) AS rank "
        "FROM chunks WHERE chunks MATCH ? ORDER BY rank, chunk_id", ('"quartz"',),
    ).fetchall()

    assert [(result.chunk_id, result.score) for result in results] == [
        (identifier, -float(rank)) for identifier, rank in rows if rank < 0
    ]
    for result in results:
        original = service._chunks[result.chunk_id]
        assert (result.source_title, result.section, result.content) == (
            original.source_title, original.section, original.content,
        )