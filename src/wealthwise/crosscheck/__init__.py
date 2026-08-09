from collections import Counter

from pydantic import BaseModel

from wealthwise.llm import ModelClient, Verdict
from wealthwise.obs import traced


class JuryResult(BaseModel):
    """Reconciled multi-model judgment (cross-check layer), symmetric to SignalConsensus.

    `label` is None when models tie with no strict majority. `escalate` is the
    actionable flag — hand off to reflection or a human when it is True.
    """
    label: str | None
    confidence: float
    disagreement: bool
    escalate: bool
    verdicts: list[Verdict]
    sources: list[str]
    tokens: int = 0        # total tokens across the jury for this judgment


@traced("wealthwise.crosscheck.deliberate")
def deliberate(clients: list[ModelClient], system: str, user: str,
               labels: list[str], *, escalate_below: float = 0.66) -> JuryResult:
    """Query each model for a Verdict and reconcile them into a JuryResult.

    A lone model is capped at 0.5 confidence (it cannot be corroborated).
    With several models, confidence is the majority share; a tie yields no
    label; escalate fires whenever confidence falls below `escalate_below`.
    """
    if not clients:
        raise ValueError("deliberate() needs at least one model client")

    verdicts = [c.judge(system, user, labels) for c in clients]
    sources = [c.name for c in clients]
    counts = Counter(v.label for v in verdicts)
    top_label, top_count = counts.most_common(1)[0]
    n = len(verdicts)
    disagreement = len(counts) > 1
    is_tie = list(counts.values()).count(top_count) > 1

    if n == 1:
        label, confidence = top_label, 0.5
    elif is_tie:
        label, confidence = None, top_count / n
    else:
        label, confidence = top_label, top_count / n

    return JuryResult(label=label, confidence=confidence, disagreement=disagreement,
                      escalate=confidence < escalate_below, verdicts=verdicts,
                      sources=sources, tokens=sum(v.tokens for v in verdicts))
