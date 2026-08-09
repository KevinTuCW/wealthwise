from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from wealthwise.rag.embed import Embedder


@dataclass
class Doc:
    id: str
    text: str
    meta: dict = field(default_factory=dict)


@runtime_checkable
class Retriever(Protocol):
    def search(self, query: str, k: int = 3) -> list[Doc]: ...


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))          # both are L2-normalized


class InMemoryVectorStore:
    """Offline vector store: embeds docs on add, cosine-ranks on search."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._items: list[tuple[Doc, list[float]]] = []

    def add(self, docs: list[Doc]) -> None:
        for d in docs:
            self._items.append((d, self._embedder.embed(d.text)))

    def search(self, query: str, k: int = 3) -> list[Doc]:
        q = self._embedder.embed(query)
        ranked = sorted(self._items, key=lambda it: _cosine(q, it[1]), reverse=True)
        return [d for d, _ in ranked[:k]]
