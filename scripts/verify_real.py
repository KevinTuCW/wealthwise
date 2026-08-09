"""Keyed real-data verification: real GLM+DeepSeek jury + real Langfuse tracing.

Market data uses the sample providers (the A-share spot host 82.push2.eastmoney.com
is blocked from this environment; funds/macro reach live). This exercises the real
multi-model cross-check + tracing + token accounting path end-to-end.

Run: SSL_CERT_FILE unset, proxy env inherited,
     ENABLE_LANGFUSE_TRACING=true PYTHONPATH=src .venv/bin/python scripts/verify_real.py
"""
import os
import time

os.environ.setdefault("ENABLE_LANGFUSE_TRACING", "true")

from dataclasses import replace

from wealthwise.config import Settings
from wealthwise.agents.deps import AdvisoryDeps
from wealthwise.agents.state import InvestorProfile
from wealthwise.bootstrap import build_sample_deps
from wealthwise.crosscheck.jury import build_jury_clients
from wealthwise.runner import run_advisory
import wealthwise.obs as obs


def build_real_jury_deps(settings: Settings) -> AdvisoryDeps:
    # sample market/macro/fx + offline retrievers, but REAL GLM + DeepSeek jury
    deps = build_sample_deps(settings)
    return replace(deps, jury_clients=build_jury_clients(settings))


PROFILES = {
    "C2-conservative": InvestorProfile(
        risk_level="C2", investable=300000, horizon_years=3,
        goals=["capital_preservation"], liquidity_min=0.3, accept_cross_border=False,
    ),
    "C4-balanced-crossborder": InvestorProfile(
        risk_level="C4", investable=1000000, horizon_years=8,
        goals=["retirement", "growth"], liquidity_min=0.1, accept_cross_border=True,
    ),
}


def jury_events(state):
    out = []
    for ev in state.trace_events:
        if any(k in str(ev).lower() for k in ("jury", "tilt", "escalat", "disagree", "verdict")):
            out.append(ev)
    return out


def main():
    settings = Settings()
    print("tracing_enabled:", settings.tracing_enabled)
    print("primary model:", settings.llm_model, "| crosscheck:", settings.crosscheck_model)
    print("obs tracing_enabled():", obs.tracing_enabled())
    print("=" * 70)
    for name, profile in PROFILES.items():
        deps = build_real_jury_deps(settings)
        t0 = time.time()
        state = run_advisory(profile, deps)
        dt = time.time() - t0
        print(f"\n### {name}")
        print(f"  status           : {state.status}")
        print(f"  decision         : {state.compliance.decision if state.compliance else None}")
        print(f"  portfolio_r_level: {state.portfolio.portfolio_r_level if state.portfolio else None}")
        print(f"  fx_exposure      : {state.portfolio.fx_exposure if state.portfolio else None}")
        print(f"  confidence       : {state.confidence}")
        print(f"  tokens_used      : {state.tokens_used}")
        print(f"  budget_spent     : {state.budget_spent}")
        print(f"  wall_clock_sec   : {dt:.1f}")
        print(f"  n_trace_events   : {len(state.trace_events)}")
        je = jury_events(state)
        print(f"  jury/tilt events : {len(je)}")
        for ev in je[:6]:
            print("     -", {k: ev[k] for k in list(ev)[:8]} if isinstance(ev, dict) else ev)
        if state.compliance and state.compliance.disclosures:
            print(f"  disclosures      : {len(state.compliance.disclosures)} items")
    # flush Langfuse traces
    try:
        client = obs.get_client() if hasattr(obs, "get_client") else None
        if client:
            client.flush()
            print("\n[langfuse] flushed")
    except Exception as e:
        print("\n[langfuse] flush note:", type(e).__name__, str(e)[:80])
    print("\nDONE")


if __name__ == "__main__":
    main()
