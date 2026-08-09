import json
from pathlib import Path

from wealthwise.rag.embed import Embedder
from wealthwise.rag.store import Doc, InMemoryVectorStore, Retriever


def load_policy_retriever(data_dir: str, embedder: Embedder) -> Retriever:
    """Load data/samples/policy.json into an in-memory vector store.

    Each entry is an investor-suitability or risk-disclosure clause (中文).
    """
    rows = json.loads((Path(data_dir) / "policy.json").read_text())
    store = InMemoryVectorStore(embedder)
    store.add([Doc(id=r["id"], text=r["text"], meta=r.get("meta", {})) for r in rows])
    return store


def load_research_retriever(data_dir: str, embedder: Embedder) -> Retriever:
    """Load data/samples/research.json into an in-memory vector store.

    Each entry is a sanitized/synthetic 标的研报/资讯 snippet (中文).
    """
    rows = json.loads((Path(data_dir) / "research.json").read_text())
    store = InMemoryVectorStore(embedder)
    store.add([Doc(id=r["id"], text=r["text"], meta=r.get("meta", {})) for r in rows])
    return store
