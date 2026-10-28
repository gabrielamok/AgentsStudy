import math
from collections.abc import Sequence

import pytest

from agent.rag.embeddings import EmbeddingProvider, normalize_embedding, validate_dimensions


class LocalTestEmbeddingProvider:
    dimensions = 3

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(text.casefold().count(term)) for term in ("clock", "water", "paper")]


def test_local_provider_is_deterministic_and_preserves_batch_order() -> None:
    provider: EmbeddingProvider = LocalTestEmbeddingProvider()
    texts = ["clock water", "paper", "CLOCK clock"]
    expected = [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [2.0, 0.0, 0.0]]

    assert provider.embed_documents(texts) == expected
    assert provider.embed_documents(texts) == expected
    assert [provider.embed_query(text) for text in texts] == expected
    assert provider.embed_documents([]) == []


def test_normalization_is_deterministic_and_does_not_mutate_input() -> None:
    vector = [3.0, 4.0]

    assert normalize_embedding(vector, 2) == (0.6, 0.8)
    assert normalize_embedding(vector, 2) == normalize_embedding(vector, 2)
    assert vector == [3.0, 4.0]


@pytest.mark.parametrize("vector", [(1e308, 1e308), (1e-308, 1e-308), (3, 4)])
def test_normalization_handles_extreme_finite_magnitudes(vector: Sequence[float]) -> None:
    result = normalize_embedding(vector, 2)

    assert math.hypot(*result) == pytest.approx(1.0)
    assert all(math.isfinite(value) for value in result)


@pytest.mark.parametrize("vector", [
    [], [1.0], [1.0, 2.0, 3.0], [0.0, 0.0], [float("nan"), 1.0],
    [float("inf"), 1.0], [float("-inf"), 1.0], [True, 1.0], ["1", 2.0],
    [[1.0], 2.0], "12", b"12", None, 1.0, {0: 1.0, 1: 2.0},
])
def test_malformed_embeddings_are_rejected(vector: object) -> None:
    with pytest.raises(ValueError):
        normalize_embedding(vector, 2)


@pytest.mark.parametrize("dimensions", [0, -1, True, 2.0, "2", None])
def test_dimensions_must_be_a_positive_integer(dimensions: object) -> None:
    with pytest.raises(ValueError):
        validate_dimensions(dimensions)


def test_valid_dimensions_are_preserved() -> None:
    assert validate_dimensions(3) == 3