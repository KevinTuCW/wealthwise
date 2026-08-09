"""Input guardrail — validate InvestorProfile before it enters the pipeline.

Two concerns:
1. Field-range validation: catch obviously bogus values (negative investable
   assets, zero horizon, out-of-range liquidity ratio) that would propagate
   silently through the advisory graph and produce nonsense recommendations.
2. Injection screening: free-text goal strings are user-controlled; run each
   through detect_injection so an adversarial goal cannot hijack the LLM agents
   downstream.
"""
from __future__ import annotations

from wealthwise.agents.state import InvestorProfile
from wealthwise.security.sanitize import detect_injection

VALID_RISK_LEVELS = {"C1", "C2", "C3", "C4", "C5"}


def screen_profile(profile: InvestorProfile) -> tuple[bool, list[str]]:
    """Validate *profile* and check goal strings for prompt injection.

    Returns
    -------
    (ok, reasons)
        ok      — True if the profile is clean and safe to use.
        reasons — list of human-readable problem descriptions (empty when ok).

    The graph node that calls this should set ``status = "GUARDRAIL_BLOCKED"``
    and surface the reasons when ok is False.
    """
    reasons: list[str] = []

    # --- field range checks ------------------------------------------------
    if profile.investable <= 0:
        reasons.append(
            f"investable assets must be > 0 (got {profile.investable})"
        )

    if profile.horizon_years <= 0:
        reasons.append(
            f"horizon_years must be > 0 (got {profile.horizon_years})"
        )

    if not (0.0 <= profile.liquidity_min <= 1.0):
        reasons.append(
            f"liquidity_min must be in [0, 1] (got {profile.liquidity_min})"
        )

    # risk_level is a Literal validated by pydantic at construction time,
    # but guard here in case model_construct was used to bypass validation.
    if profile.risk_level not in VALID_RISK_LEVELS:
        reasons.append(
            f"risk_level must be one of {sorted(VALID_RISK_LEVELS)} "
            f"(got {profile.risk_level!r})"
        )

    # --- injection screening on free-text goals ----------------------------
    for goal in profile.goals:
        hit, category = detect_injection(goal)
        if hit:
            reasons.append(
                f"possible prompt injection in goal ({category}): {goal!r}"
            )

    ok = len(reasons) == 0
    return ok, reasons
