from wealthwise.config import Settings
from wealthwise.crosscheck import JuryResult, deliberate
from wealthwise.llm import ModelClient, OpenAICompatibleModelClient


def build_jury_clients(settings: Settings) -> list[ModelClient]:
    """Assemble the primary (GLM) + cross-check (SiliconFlow / DeepSeek) model clients."""
    return [
        OpenAICompatibleModelClient(**settings.primary_client_kwargs()),
        OpenAICompatibleModelClient(**settings.crosscheck_client_kwargs()),
    ]


def run_jury(settings: Settings, system: str, user: str,
             labels: list[str]) -> JuryResult:
    """Convenience: build the real jury and deliberate a single judgment."""
    return deliberate(build_jury_clients(settings), system, user, labels)
