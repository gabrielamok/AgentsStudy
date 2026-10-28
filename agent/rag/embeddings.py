import math
from collections.abc import Sequence
from numbers import Real
from typing import Protocol


Embedding = tuple[float, ...]


class EmbeddingProvider(Protocol):
    """Explicitly supplied provider; document and query vectors must share a space."""

    @property
    def dimensions(self) -> int:
        ...

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one vector per text, in input order, without dropping entries."""
        ...

    def embed_query(self, text: str) -> Sequence[float]:
        ...


def validate_dimensions(dimensions: int) -> int:
    if type(dimensions) is not int or dimensions <= 0:
        raise ValueError("Embedding dimensions must be a positive integer")
    return dimensions


def normalize_embedding(vector: Sequence[float], dimensions: int) -> Embedding:
    """Validate and normalize a nonzero dense vector without overflowing its norm."""
    validate_dimensions(dimensions)
    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes, bytearray)):
        raise ValueError("Embedding must be a sequence of real numbers")
    if len(vector) != dimensions:
        raise ValueError("Embedding dimension mismatch")
    coordinates: list[float] = []
    for coordinate in vector:
        if isinstance(coordinate, bool) or not isinstance(coordinate, Real):
            raise ValueError("Embedding coordinates must be finite real numbers")
        try:
            value = float(coordinate)
        except (OverflowError, ValueError) as error:
            raise ValueError("Embedding coordinates must be finite real numbers") from error
        if not math.isfinite(value):
            raise ValueError("Embedding coordinates must be finite real numbers")
        coordinates.append(value)
    scale = max(abs(value) for value in coordinates)
    if scale == 0:
        raise ValueError("Embedding must have a nonzero norm")
    scaled = tuple(value / scale for value in coordinates)
    norm = math.hypot(*scaled)
    return tuple(value / norm for value in scaled)