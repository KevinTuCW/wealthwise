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
    traced as a `generation`. Temperature 0 for reproducible verdicts.

    Parameters
    ----------
    extra_body:
        Vendor-specific request fields. Used to turn off extended reasoning:
        a juror picks one label from a fixed set, and left to think freely
        GLM-4.7 spent ~1,150 completion tokens and 30–60s reaching the same
        verdict it reaches in 42 tokens and 2.4s with thinking off. The reasoning
        was not buying accuracy on a closed-set classification, and it was the
        single largest contributor to advisory latency.
        The field name differs per vendor (`thinking` on z.ai, `enable_thinking`
        on SiliconFlow), so it is passed in per client rather than hard-coded.
    timeout:
        Per-request bound, and deliberately the *only* one. An earlier revision
        also capped `max_tokens`, reasoning that a juror ignoring `extra_body`
        should not run unbounded. Kimi-K3 is exactly such a juror, and capping it
        made things worse rather than safer: it spent the whole allowance on
        reasoning it would not stop producing and returned an empty `content`,
        so a cap meant to contain a slow juror turned it into a failed one that
        took the advisory run down with it. Wall-clock is the bound that actually
        holds regardless of how a model spends its tokens.
    max_retries:
        Pinned rather than left at the SDK default of 2, which silently turned
        `timeout` into three times itself: a juror that ran long was retried
        twice more, so a 60s bound produced a 180s node. One retry absorbs a
        transient error while keeping the worst case legible as 2 × timeout.
    """

    def __init__(self, name: str, model: str, base_url: str, api_key: str,
                 extra_body: dict | None = None, timeout: float = 60.0,
                 max_retries: int = 1) -> None:
        from langfuse.openai import OpenAI
        self.name = name
        self._model = model
        self._extra_body = extra_body or None
        self._client = OpenAI(base_url=base_url or None, api_key=api_key or None,
                              timeout=timeout, max_retries=max_retries)

    def judge(self, system: str, user: str, labels: list[str]) -> Verdict:
        instruction = (
            f'{user}\n\nRespond with ONLY a JSON object: '
            f'{{"label": <one of {labels}>, "rationale": "<one sentence>"}}'
        )
        kwargs: dict = {}
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body
        resp = self._client.chat.completions.create(
            model=self._model, temperature=0,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": instruction}],
            **kwargs,
        )
        content = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        tokens = getattr(usage, "total_tokens", 0) or 0
        return parse_verdict(content, labels, tokens=tokens)
