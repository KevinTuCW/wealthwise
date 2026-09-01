from collections import Counter
from concurrent.futures import ThreadPoolExecutor

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


def jury_votes(result: "JuryResult", stage: str) -> list[dict]:
    """Per-juror labels, in the shape the trace and the workbench read.

    `deliberate()` already carries every juror's verdict, but until this landed
    nothing put them into `trace_events`, so the dashboard reconstructed the
    jury from node names — found none — and reported "no jury was convened" on
    runs where the jury had in fact decided the macro tilt. A pillar that cannot
    be seen in the trace cannot be audited, which for a cross-validation layer
    is most of the point of having one.
    """
    return [
        {"stage": stage, "source": source, "label": verdict.label,
         "rationale": (verdict.rationale or "")[:160]}
        for source, verdict in zip(result.sources, result.verdicts)
    ]


@traced("wealthwise.crosscheck.deliberate")
def deliberate(clients: list[ModelClient], system: str, user: str,
               labels: list[str], *, escalate_below: float = 0.66) -> JuryResult:
    """Query each model for a Verdict and reconcile them into a JuryResult.

    A lone model is capped at 0.5 confidence (it cannot be corroborated).
    With several models, confidence is the majority share; a tie yields no
    label; escalate fires whenever confidence falls below `escalate_below`.

    Jurors are polled concurrently. They are independent by construction — that
    independence is the whole premise of cross-validation — so querying them in
    sequence made a deliberation cost the *sum* of the jurors' latencies when it
    only ever needed to cost the slowest one. Results are still collected in
    client order, so `verdicts`, `sources` and the reconciliation below are
    unchanged; an exception from any juror still propagates, and does so from the
    first juror in client order that raised, rather than whichever failed soonest.
    """
    if not clients:
        raise ValueError("deliberate() needs at least one model client")

    if len(clients) == 1:
        verdicts = [clients[0].judge(system, user, labels)]
    else:
        with ThreadPoolExecutor(max_workers=len(clients)) as pool:
            futures = [pool.submit(c.judge, system, user, labels) for c in clients]
            verdicts = [f.result() for f in futures]   # in client order; re-raises
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
