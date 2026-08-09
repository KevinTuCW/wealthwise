import hashlib
import math
import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...


class LocalHashingEmbedder:
    """Dependency-free bag-of-words hashing embedder, L2-normalized (offline).

    Not semantic — tokens hash into fixed buckets — but needs no model or API,
    so the RAG pipeline runs and tests deterministically. Swap for a real
    embedder (SiliconFlow Qwen3) via config in production.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in re.findall(r"\w+", text.lower()):
            vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
