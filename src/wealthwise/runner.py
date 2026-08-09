"""Advisory pipeline runner — thin wrapper around build_graph + invoke.

Mirror of shopscout.funnel.runner.run_funnel.
"""
from __future__ import annotations

from wealthwise.agents.deps import AdvisoryDeps
from wealthwise.agents.state import AdvisoryState, InvestorProfile
from wealthwise.agents.supervisor.graph import build_graph
from wealthwise.obs import traced


@traced("wealthwise.run_advisory")
def run_advisory(
    profile: InvestorProfile | None,
    deps: AdvisoryDeps,
) -> AdvisoryState:
    """Run the full advisory pipeline as a LangGraph state machine.

    Builds the graph with deps bound in, invokes with an initial
    AdvisoryState(profile=profile), and returns the final AdvisoryState.

    Parameters
    ----------
    profile:
        InvestorProfile to advise on.  May be None — the input guard node
        will catch it and short-circuit to GUARDRAIL_BLOCKED.
    deps:
        AdvisoryDeps — providers, jury clients, and guardrail parameters.

    Returns
    -------
    AdvisoryState
        The final state after all nodes have executed (or the pipeline was
        terminated early by a guardrail or budget check).
    """
    graph = build_graph(deps)
    initial = AdvisoryState(profile=profile)
    final = graph.invoke(initial)
    return AdvisoryState.model_validate(final)
