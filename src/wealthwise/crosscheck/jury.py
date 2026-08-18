from wealthwise.config import Settings
from wealthwise.crosscheck import JuryResult, deliberate
from wealthwise.llm import ModelClient, OpenAICompatibleModelClient


def build_jury_clients(settings: Settings) -> list[ModelClient]:
    """Assemble the jury: GLM primary + two cross-lab SiliconFlow jurors.

    Three labs (Zhipu / DeepSeek / Ant) on purpose. Same-lab models share
    their failure modes, so cross-validation between them degrades into
    self-endorsement — and an *even* jury has no majority to speak of: two models
    either agree or tie. An odd jury restores the three outcomes the
    reconciliation logic was written for: unanimous, majority, no-majority.

    The jury stays advisory either way — `_stricter` means it can only tighten a
    compliance verdict, never soften one.

    `third_model=""` falls back to the original two-model pairing.
    """
    clients = [
        OpenAICompatibleModelClient(**settings.primary_client_kwargs()),
        OpenAICompatibleModelClient(**settings.crosscheck_client_kwargs()),
    ]
    if settings.third_model:
        clients.append(OpenAICompatibleModelClient(**settings.third_client_kwargs()))
    return clients


def run_jury(settings: Settings, system: str, user: str,
             labels: list[str]) -> JuryResult:
    """Convenience: build the real jury and deliberate a single judgment."""
    return deliberate(build_jury_clients(settings), system, user, labels)
