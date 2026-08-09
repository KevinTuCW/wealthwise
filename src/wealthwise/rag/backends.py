from wealthwise.config import Settings
from wealthwise.rag.embed import Embedder, LocalHashingEmbedder


class SiliconFlowEmbedder:
    """Real embedder over SiliconFlow's OpenAI-compatible embeddings endpoint.

    Uses the langfuse.openai drop-in so embeddings are traced. Lazily imported
    so the default offline build never needs openai/langfuse for RAG.
    """

    def __init__(self, model: str, base_url: str, api_key: str, dim: int) -> None:
        from langfuse.openai import OpenAI
        self.dim = dim
        self._model = model
        self._client = OpenAI(base_url=base_url or None, api_key=api_key or None)

    def embed(self, text: str) -> list[float]:
        r = self._client.embeddings.create(model=self._model, input=text, dimensions=self.dim)
        return r.data[0].embedding


def build_embedder(settings: Settings) -> Embedder:
    """Pick the embedder from config: offline hashing by default, real if set."""
    if settings.embed_provider == "siliconflow":
        return SiliconFlowEmbedder(model=settings.embed_model,
                                   base_url=settings.siliconflow_base_url,
                                   api_key=settings.siliconflow_api_key,
                                   dim=settings.embed_dim)
    return LocalHashingEmbedder(dim=settings.embed_dim)
