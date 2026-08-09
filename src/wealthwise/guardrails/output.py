"""Output guardrail — enforce disclosure completeness and content safety.

Four mandatory checks before an advisory state may be issued:
1. Suitability-match statement present (C-level × portfolio R-level match).
2. Risk disclosure present — AND if the portfolio holds cross-border (HK/US)
   assets, the disclosure MUST include FX/channel/tax risk wording.
3. "不构成投资建议" disclaimer present in the explanation.
4. Confidence > 0 (evidence of scoring is present).

If ANY check fails → status is set to "NEEDS_HUMAN_REVIEW" and a note is
appended; the state must NOT be issued as-is.

Additionally:
- compliance.decision is validated against the legal vocabulary.
- state.explanation is run through detect_misleading + neutralize so banned
  promotional phrases are stripped before the explanation reaches the user.
- All notes are PII-scrubbed via redact.

Status convention
-----------------
"NEEDS_HUMAN_REVIEW" — one or more output checks failed; a human must
    review and approve before issuance.
"GUARDRAIL_BLOCKED"  — (set by the input guardrail / graph node) profile
    validation or injection detected; pipeline should not proceed.

States whose status is not "done" are not subject to the disclosure checks
(they haven't produced a recommendation yet).
"""
from __future__ import annotations

from wealthwise.agents.state import AdvisoryState
from wealthwise.compliance.language import detect_misleading, neutralize
from wealthwise.security.redact import redact

_VALID_COMPLIANCE_DECISIONS = {"PASS", "DOWNGRADE", "REJECT"}

# Keywords that satisfy the FX/cross-border risk disclosure requirement.
# At least one of these must appear in the combined notes when a cross-border
# asset is present.
_FX_KEYWORDS = ("汇率", "fx", "currency", "渠道", "channel", "税", "tax")

# Keywords that signal a suitability-match statement is present.
_SUITABILITY_KEYWORDS = ("适当性", "suitability", "匹配", "match")

# Keywords that signal a risk disclosure statement is present.
_RISK_KEYWORDS = ("风险", "risk", "risk disclosure")

# The required disclaimer phrase.
_DISCLAIMER = "不构成投资建议"

# Status written when a check fails.
STATUS_NEEDS_REVIEW = "NEEDS_HUMAN_REVIEW"

# Statuses that represent a completed / issuing recommendation.
_ISSUING_STATUSES = {"done", "completed", "issued"}


def _has_cross_border_assets(state: AdvisoryState) -> bool:
    """Return True if the portfolio holds any HK or US market assets."""
    if state.portfolio is None:
        return False
    # We need market info from the candidates. Aggregate equity/fixed-income
    # candidate pools are stored on the state.
    all_candidates = [
        *state.equity_candidates,
        *state.fixedincome_candidates,
    ]
    symbol_market = {c.symbol: c.market for c in all_candidates}
    for symbol, weight in state.portfolio.weights.items():
        if weight <= 0:
            continue
        market = symbol_market.get(symbol, "A")
        if market in {"HK", "US"}:
            return True
    # Also check fx_exposure as a fallback: if it's non-zero, assume cross-border.
    if state.portfolio.fx_exposure > 0:
        return True
    return False


def has_complete_disclosures(state: AdvisoryState) -> tuple[bool, list[str]]:
    """Check whether *state* carries all required advisory disclosures.

    This helper is intentionally separated so tests can call it directly.

    Returns
    -------
    (ok, missing)
        ok      — True iff all four disclosure requirements are met.
        missing — human-readable descriptions of what is absent.
    """
    missing: list[str] = []
    notes_lower = " ".join(state.notes).lower()
    explanation_lower = (state.explanation or "").lower()

    # 1. Suitability-match statement
    if not any(kw in notes_lower for kw in _SUITABILITY_KEYWORDS):
        missing.append(
            "suitability match statement missing from notes "
            "(expected 适当性匹配 / C-level × R-level confirmation)"
        )

    # 2. Risk disclosure — generic
    if not any(kw in notes_lower for kw in _RISK_KEYWORDS):
        missing.append(
            "risk disclosure missing from notes "
            "(expected 投资有风险 or equivalent)"
        )
    else:
        # 2b. Cross-border FX/channel/tax addendum required
        if _has_cross_border_assets(state):
            if not any(kw in notes_lower for kw in _FX_KEYWORDS):
                missing.append(
                    "cross-border portfolio detected but FX/channel/tax risk "
                    "disclosure absent from notes (required for HK/US assets)"
                )

    # 3. "不构成投资建议" disclaimer
    if _DISCLAIMER not in explanation_lower and _DISCLAIMER not in notes_lower:
        missing.append(
            f"disclaimer '{_DISCLAIMER}' missing from explanation or notes"
        )

    # 4. Confidence > 0
    if state.confidence <= 0:
        missing.append(
            f"confidence is {state.confidence}; evidence of scoring required "
            "(must be > 0 before issuance)"
        )

    return len(missing) == 0, missing


def enforce_output(state: AdvisoryState) -> AdvisoryState:
    """Apply all output guardrail checks and content sanitization.

    Mutates *state* in-place (fields are set via model_copy to remain Pydantic-
    friendly) and returns it.

    States that are not in an issuing status are sanitized (misleading language
    and PII scrubbed) but are NOT subject to the disclosure completeness checks.
    """
    updates: dict = {}

    # --- 1. Compliance decision vocabulary check ---------------------------
    if (
        state.compliance is not None
        and state.compliance.decision not in _VALID_COMPLIANCE_DECISIONS
    ):
        notes = [
            *state.notes,
            f"output guardrail: invalid compliance decision "
            f"{state.compliance.decision!r}; flagging for human review",
        ]
        updates["notes"] = notes
        updates["status"] = STATUS_NEEDS_REVIEW

    # --- 1b. Compliance decision gate: REJECT/DOWNGRADE must never issue ---
    if (
        state.compliance is not None
        and state.compliance.decision in {"REJECT", "DOWNGRADE"}
        and "status" not in updates
    ):
        notes = [
            *state.notes,
            f"output guardrail: compliance decision is "
            f"{state.compliance.decision!r} — advisory must not be issued; "
            f"flagging for human review",
        ]
        updates["notes"] = notes
        updates["status"] = STATUS_NEEDS_REVIEW

    # --- 2. Disclosure completeness (only for issuing states) --------------
    if state.status in _ISSUING_STATUSES and "status" not in updates:
        ok, missing = has_complete_disclosures(state)
        if not ok:
            notes = [
                *state.notes,
                "output guardrail: disclosure checks failed — flagged for "
                "human review: " + "; ".join(missing),
            ]
            updates["notes"] = notes
            updates["status"] = STATUS_NEEDS_REVIEW

    # Apply status/notes updates before content sanitization so we sanitize
    # the potentially appended notes too.
    if updates:
        state = state.model_copy(update=updates)

    # --- 3. Misleading-language neutralization on explanation --------------
    if state.explanation:
        if detect_misleading(state.explanation):
            state = state.model_copy(update={"explanation": neutralize(state.explanation)})

    # --- 4. PII redaction of notes -----------------------------------------
    clean_notes = [redact(n) for n in state.notes]
    if clean_notes != state.notes:
        state = state.model_copy(update={"notes": clean_notes})

    return state
