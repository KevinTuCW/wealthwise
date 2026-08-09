import json
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class Verdict(BaseModel):
    """One model's structured judgment: a label from an allowed set + why."""
    label: str
    rationale: str
    tokens: int = 0        # total tokens billed for this judgment (0 for offline clients)


@runtime_checkable
class ModelClient(Protocol):
    """A model that returns a structured Verdict for a judgment prompt.

    FakeModelClient is used in tests; OpenAICompatibleModelClient hits a real
    OpenAI-compatible endpoint (GLM via z.ai, DeepSeek via SiliconFlow). The
    jury depends only on this interface, so models are swappable without
    touching the reconciliation logic.
    """
    name: str

    def judge(self, system: str, user: str, labels: list[str]) -> Verdict: ...


class FakeModelClient:
    """Deterministic client returning a preset verdict (test double)."""

    def __init__(self, name: str, verdict: Verdict) -> None:
        self.name = name
        self._verdict = verdict

    def judge(self, system: str, user: str, labels: list[str]) -> Verdict:
        return self._verdict


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t[3:]
        if t[:4].lower() == "json":
            t = t[4:]
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


def parse_verdict(raw: str, labels: list[str], tokens: int = 0) -> Verdict:
    """Parse a model's raw reply into a Verdict, enforcing the allowed labels.

    Tolerates ```json fenced blocks. Raises ValueError on non-JSON output or a
    label outside `labels` — the caller treats that as a failed judgment.
    """
    try:
        data = json.loads(_strip_fences(raw))
    except json.JSONDecodeError as e:
        raise ValueError(f"model did not return JSON: {raw!r}") from e
    raw_label = str(data.get("label", "")).strip()
    match = next((l for l in labels if l.lower() == raw_label.lower()), None)
    if match is None:
        raise ValueError(f"label {raw_label!r} not in {labels}")
    return Verdict(label=match, rationale=str(data.get("rationale", "")), tokens=tokens)


class OpenAICompatibleModelClient:
    """Real client over any OpenAI-compatible endpoint (GLM via z.ai, DeepSeek
    via SiliconFlow). Uses the langfuse.openai drop-in so each judgment is
    traced as a `generation`. Temperature 0 for reproducible verdicts."""

    def __init__(self, name: str, model: str, base_url: str, api_key: str) -> None:
        from langfuse.openai import OpenAI
        self.name = name
        self._model = model
        self._client = OpenAI(base_url=base_url or None, api_key=api_key or None)

    def judge(self, system: str, user: str, labels: list[str]) -> Verdict:
        instruction = (
            f'{user}\n\nRespond with ONLY a JSON object: '
            f'{{"label": <one of {labels}>, "rationale": "<one sentence>"}}'
        )
        resp = self._client.chat.completions.create(
            model=self._model, temperature=0,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": instruction}],
        )
        content = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        tokens = getattr(usage, "total_tokens", 0) or 0
        return parse_verdict(content, labels, tokens=tokens)
