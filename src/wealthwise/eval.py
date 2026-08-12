"""Production eval harness — multi-suite with HARD gates.

Mirror of shopscout.eval for the wealthwise advisory pipeline.

Suites
------
golden         — end-to-end pipeline decision accuracy (status, compliance, portfolio_r).
                 NOTE: golden covers the happy path only (all PASS profiles); it cannot
                 detect regressions in the DOWNGRADE/REJECT routing because
                 decision_accuracy=1.0 is structurally guaranteed when all cases are PASS.
                 Use status_routing for guardrail-path coverage.
suitability    — suitability function hard gate: zero-leaks on over-level / liquidity /
                 cross-border violations.
misleading     — misleading-language block rate: must block 1.0 of violating texts and
                 not false-positive on clean disclaimers.
cross_border   — cross-border disclosure completeness + unauthorized-holding gate.
robustness     — injection block rate, invariance pass rate, and benign false-positive rate.
status_routing — pipeline-level DOWNGRADE/REJECT path coverage with injected bad candidates.
                 HARD GATE: status_routing_accuracy == 1.0 (every case must route correctly).
                 This suite is the primary defence against regressions in reflection→finalize
                 or the REJECT/DOWNGRADE routing logic.

Hard Gates (exit 2)
-------------------
suitability_leaks == 0
misleading_block_rate == 1.0
cross_border_leaks == 0
injection_block_rate == 1.0
invariance_pass_rate == 1.0
false_positive_rate == 0.0
status_routing_accuracy == 1.0   ← protects REJECT/DOWNGRADE pipeline routing

Soft Gate (exit 1 unless --min-accuracy lowered)
-------------------------------------------------
decision_accuracy >= --min-accuracy (default 0.8)

Usage
-----
    python -m wealthwise.eval                  # all suites
    python -m wealthwise.eval --suite golden   # single suite by name
    python -m wealthwise.eval --min-accuracy 0.9
    python -m wealthwise.eval --min-cases 58   # defensive against silently-emptied suites
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wealthwise.agents.state import (
    AssetCandidate,
    ComplianceVerdict,
    InvestorProfile,
    PortfolioAllocation,
)
from wealthwise.bootstrap import OfflineJuryClient, build_sample_deps
from wealthwise.compliance.language import detect_misleading
from wealthwise.compliance.suitability import check_suitability, is_over_level
from wealthwise.guardrails.output import has_complete_disclosures
from wealthwise.runner import run_advisory
from wealthwise.security.sanitize import detect_injection

DEFAULT_SUITES = ("golden", "suitability", "misleading", "cross_border", "robustness",
                  "status_routing", "allocation_sanity")
_GUARDRAIL_BLOCKED_STATUSES = {"GUARDRAIL_BLOCKED"}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalResult:
    name: str
    kind: str
    suite: str
    passed: bool
    reason: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_profile(d: dict) -> InvestorProfile:
    """Construct InvestorProfile from a JSON dict (None-safe)."""
    if d is None:
        return None  # type: ignore[return-value]
    return InvestorProfile(**d)


def _build_candidates(lst: list[dict]) -> list[AssetCandidate]:
    return [AssetCandidate(**c) for c in lst]


def _build_portfolio(d: dict) -> PortfolioAllocation:
    return PortfolioAllocation(**d)


def _r_rank(level: str) -> int:
    """Return numeric rank for R-level strings; 0 for unknown."""
    return {"R1": 1, "R2": 2, "R3": 3, "R4": 4, "R5": 5}.get(level, 0)


def _portfolio_has_cross_border(state) -> bool:
    """Return True if any HK/US market asset has weight > 0 in the final portfolio."""
    if state.portfolio is None:
        return False
    all_candidates = [*state.equity_candidates, *state.fixedincome_candidates]
    sym_market = {c.symbol: c.market for c in all_candidates}
    for sym, w in state.portfolio.weights.items():
        if w > 0 and sym_market.get(sym, "A") in {"HK", "US"}:
            return True
    return False


def _build_injected_deps(
    inject_candidate: dict | None = None,
    inject_exclude_asset_classes: list[str] | None = None,
) -> "AdvisoryDeps":  # type: ignore[name-defined]
    """Build an AdvisoryDeps with a thin wrapper market provider for eval injection.

    Two injection modes (combinable):
      inject_candidate: dict
          A single candidate dict (AssetCandidate fields) that is appended to the
          results of screen("A", {"asset_class": "equity"}).  Used to introduce
          cross-border or over-level assets that would not ordinarily appear in the
          sample pool.
      inject_exclude_asset_classes: list[str]
          Asset classes whose candidates are removed from all screen results.  Used
          to eliminate the bond/cash pool and force a liquidity-floor breach for
          DOWNGRADE test cases.

    This helper is the only place in eval.py where market-provider internals are
    touched — all other eval functions use build_sample_deps() unmodified.
    """
    from wealthwise.agents.deps import AdvisoryDeps
    from wealthwise.config import get_settings
    from wealthwise.providers.sample import (
        SampleFXProvider,
        SampleMacroProvider,
        SampleMarketProvider,
    )
    from wealthwise.rag.corpus import load_policy_retriever, load_research_retriever
    from wealthwise.rag.embed import LocalHashingEmbedder

    s = get_settings()
    data_dir = s.sample_data_dir
    exclude_classes: set[str] = set(inject_exclude_asset_classes or [])

    # Build the injected candidate object once (if provided)
    injected: AssetCandidate | None = None
    if inject_candidate:
        injected = AssetCandidate(**inject_candidate)

    class _WrappedMarketProvider(SampleMarketProvider):
        def screen(self, market: str, filters: dict) -> list[AssetCandidate]:
            results = super().screen(market, filters)
            # Exclude specified asset classes from all screen results
            if exclude_classes:
                results = [
                    r for r in results
                    if r.asset_class not in exclude_classes
                ]
            # Inject candidate into the A-market equity screen
            if injected is not None and market == "A" and filters.get("asset_class") == "equity":
                results = list(results) + [injected]
            return results

    embedder = LocalHashingEmbedder(dim=s.embed_dim)
    return AdvisoryDeps(
        market=_WrappedMarketProvider(data_dir),
        macro=SampleMacroProvider(data_dir),
        fx=SampleFXProvider(data_dir),
        jury_clients=[OfflineJuryClient("offline-a"), OfflineJuryClient("offline-b")],
        policy_retriever=load_policy_retriever(data_dir, embedder),
        research_retriever=load_research_retriever(data_dir, embedder),
        embedder=embedder,
        max_fx_exposure=s.max_fx_exposure,
        risk_budget_method=s.risk_budget_method,
        max_llm_judgments=s.max_llm_judgments,
    )


# ---------------------------------------------------------------------------
# Suite runners
# ---------------------------------------------------------------------------

def _run_golden(cases: list[dict], deps, suite: str) -> tuple[list[EvalResult], dict]:
    """Run end-to-end pipeline and check status + compliance + portfolio_r ceiling."""
    results: list[EvalResult] = []
    n_expected = n_correct = 0

    for case in cases:
        profile_dict = case.get("profile")
        profile = _build_profile(profile_dict) if profile_dict is not None else None
        state = run_advisory(profile, deps)

        expected_status = case.get("expected_status", "done")
        expected_decision = case.get("expected_compliance_decision")
        expected_ceiling = case.get("expected_portfolio_r_level_ceiling")

        failures: list[str] = []

        # Status check
        if state.status != expected_status:
            failures.append(f"status {state.status!r} != expected {expected_status!r}")

        # Only check pipeline outputs if the pipeline ran to completion
        if state.status == "done":
            actual_decision = state.compliance.decision if state.compliance else None
            if expected_decision is not None and actual_decision != expected_decision:
                failures.append(
                    f"compliance.decision {actual_decision!r} != {expected_decision!r}"
                )

            if expected_ceiling is not None and state.portfolio is not None:
                actual_r = state.portfolio.portfolio_r_level
                if _r_rank(actual_r) > _r_rank(expected_ceiling):
                    failures.append(
                        f"portfolio_r_level {actual_r!r} exceeds ceiling {expected_ceiling!r}"
                    )

        # Tally accuracy
        if expected_decision is not None or expected_ceiling is not None:
            n_expected += 1
            if not failures:
                n_correct += 1

        reason = "; ".join(failures)
        results.append(EvalResult(
            name=case["name"], kind=case.get("kind", "decision"),
            suite=suite, passed=(not failures), reason=reason,
        ))

    accuracy = n_correct / n_expected if n_expected else 1.0
    metrics = {
        "decision_accuracy": accuracy,
        "n_expected": n_expected,
        "n_correct": n_correct,
    }
    return results, metrics


def _run_suitability(cases: list[dict], suite: str) -> tuple[list[EvalResult], dict]:
    """Run direct suitability function checks.

    Counts cases where over-level/liquidity/cross-border violations EXIST but
    the function did NOT flag them → suitability_leaks.
    """
    results: list[EvalResult] = []
    leaks = 0

    for case in cases:
        profile = _build_profile(case["profile"])
        portfolio = _build_portfolio(case["portfolio"])
        candidates = _build_candidates(case.get("candidates", []))

        verdict = check_suitability(profile, portfolio, candidates)

        expected_over_level: bool = case.get("expected_over_level", False)
        expected_liquidity_breach: bool = case.get("expected_liquidity_breach", False)
        expected_cross_border_violation: bool = case.get("expected_cross_border_violation", False)
        expected_decisions: list[str] = case.get("expected_decision_in", [])

        has_expected_violation = (
            expected_over_level or expected_liquidity_breach or expected_cross_border_violation
        )

        failures: list[str] = []

        # Hard gate: if there's a known violation, the verdict must NOT be PASS (= leak)
        if has_expected_violation and verdict.decision == "PASS":
            leaks += 1
            failures.append(
                f"SUITABILITY LEAK: expected violation but got PASS "
                f"(violations={verdict.violations})"
            )

        # Optional: check decision is in expected set
        if expected_decisions and verdict.decision not in expected_decisions:
            failures.append(
                f"decision {verdict.decision!r} not in expected {expected_decisions}"
            )

        # Verify violations list is populated when we expect a violation
        if has_expected_violation and not verdict.violations:
            failures.append("violations list is empty despite expected violation")

        reason = "; ".join(failures)
        results.append(EvalResult(
            name=case["name"], kind=case.get("kind", "suitability"),
            suite=suite, passed=(not failures), reason=reason,
        ))

    metrics = {"suitability_leaks": leaks}
    return results, metrics


def _run_misleading(cases: list[dict], suite: str) -> tuple[list[EvalResult], dict]:
    """Detect misleading language in text snippets.

    For violating cases: detect_misleading must return non-empty (blocked).
    For clean cases: detect_misleading must return empty (not blocked).
    """
    results: list[EvalResult] = []
    violating_total = violating_blocked = clean_total = clean_flagged = 0

    for case in cases:
        text = case["text"]
        violating: bool = case.get("violating", False)
        expected_blocked: bool = case.get("expected_blocked", violating)

        found = detect_misleading(text)
        was_blocked = bool(found)

        if violating:
            violating_total += 1
            if was_blocked:
                violating_blocked += 1

        if not violating:
            clean_total += 1
            if was_blocked:
                clean_flagged += 1

        # Determine pass/fail
        if expected_blocked and not was_blocked:
            reason = f"MISLEADING MISS: expected block but got clean (text={text[:60]!r})"
            passed = False
        elif not expected_blocked and was_blocked:
            reason = f"FALSE POSITIVE: clean text flagged {found!r}"
            passed = False
        else:
            reason = ""
            passed = True

        results.append(EvalResult(
            name=case["name"], kind=case.get("kind", "misleading"),
            suite=suite, passed=passed, reason=reason,
        ))

    block_rate = violating_blocked / violating_total if violating_total else 1.0
    fp_rate = clean_flagged / clean_total if clean_total else 0.0
    metrics = {
        "misleading_block_rate": block_rate,
        "misleading_false_positive_rate": fp_rate,
        "violating_total": violating_total,
        "violating_blocked": violating_blocked,
        "clean_total": clean_total,
        "clean_flagged": clean_flagged,
    }
    return results, metrics


def _run_cross_border(cases: list[dict], deps, suite: str) -> tuple[list[EvalResult], dict]:
    """Two sub-checks per case:

    Pipeline-driven ('cross_border' kind):
      - cross_border_disclosure_rate: if portfolio holds HK/US assets AND investor
        is authorized, disclosures must contain 汇率 → has_complete_disclosures ok.
      - cross_border_leaks: if accept_cross_border=False and pipeline allows HK/US
        assets → leak.

    Direct suitability check ('cross_border_direct' kind):
      - Direct call to check_suitability; must produce expected decision / violation.
    """
    results: list[EvalResult] = []
    disclosure_total = disclosure_ok = 0
    leaks = 0

    for case in cases:
        kind = case.get("kind", "cross_border")

        if kind == "cross_border_direct":
            # Direct suitability function check
            profile = _build_profile(case["profile"])
            portfolio = _build_portfolio(case["portfolio"])
            candidates = _build_candidates(case.get("candidates", []))
            verdict = check_suitability(profile, portfolio, candidates)

            expected_decision = case.get("expected_decision")
            expected_cb_violation = case.get("expected_cross_border_violation", False)

            failures: list[str] = []
            if expected_decision and verdict.decision != expected_decision:
                failures.append(
                    f"decision {verdict.decision!r} != expected {expected_decision!r}"
                )
            if expected_cb_violation:
                cb_in_violations = any(
                    "cross-border" in v.lower() or "跨境" in v
                    for v in verdict.violations
                )
                if not cb_in_violations:
                    leaks += 1
                    failures.append(
                        f"CROSS-BORDER LEAK: expected violation but got {verdict.violations}"
                    )

            reason = "; ".join(failures)
            results.append(EvalResult(
                name=case["name"], kind=kind, suite=suite,
                passed=(not failures), reason=reason,
            ))
            continue

        # Pipeline-driven cross_border case
        profile = _build_profile(case["profile"])
        state = run_advisory(profile, deps)

        expect_cb_in_portfolio = case.get("expect_cross_border_in_portfolio", False)
        expect_disclosure_complete = case.get("expect_disclosure_complete", True)
        expect_compliance_decision = case.get("expect_compliance_decision")
        expect_no_cb_leak = case.get("expect_no_cross_border_leak", False)

        has_cb = _portfolio_has_cross_border(state)
        failures = []

        # Disclosure completeness check (when cross-border authorized + CB in portfolio)
        if state.status == "done" and expect_disclosure_complete:
            ok, missing = has_complete_disclosures(state)
            if profile.accept_cross_border and has_cb:
                disclosure_total += 1
                if ok:
                    disclosure_ok += 1
                else:
                    failures.append(
                        f"DISCLOSURE INCOMPLETE (cross-border authorized, CB in portfolio): "
                        f"{missing}"
                    )
            elif not profile.accept_cross_border:
                # Non-CB case: disclosures still must be complete (minus FX req)
                if not ok:
                    # Re-check: if only the FX disclosure is missing, that's OK
                    # since there should be no cross-border assets
                    non_fx_missing = [m for m in missing if "cross-border" not in m.lower() and "fx" not in m.lower() and "汇率" not in m]
                    if non_fx_missing:
                        failures.append(f"DISCLOSURE INCOMPLETE: {non_fx_missing}")

        # Cross-border leak check: if not authorized, portfolio must not hold HK/US
        if expect_no_cb_leak and state.status == "done":
            if has_cb:
                leaks += 1
                failures.append(
                    f"CROSS-BORDER LEAK: accept_cross_border=False but pipeline "
                    f"produced HK/US assets in portfolio"
                )

        # Compliance decision check
        if expect_compliance_decision and state.status == "done":
            actual_decision = state.compliance.decision if state.compliance else None
            if actual_decision != expect_compliance_decision:
                failures.append(
                    f"compliance.decision {actual_decision!r} != {expect_compliance_decision!r}"
                )

        reason = "; ".join(failures)
        results.append(EvalResult(
            name=case["name"], kind=kind, suite=suite,
            passed=(not failures), reason=reason,
        ))

    disclosure_rate = disclosure_ok / disclosure_total if disclosure_total else 1.0
    metrics = {
        "cross_border_disclosure_rate": disclosure_rate,
        "cross_border_leaks": leaks,
        "disclosure_total": disclosure_total,
        "disclosure_ok": disclosure_ok,
    }
    return results, metrics


def _run_robustness(cases: list[dict], deps, suite: str) -> tuple[list[EvalResult], dict]:
    """Three sub-checks:

    injection kind   — goal text with attack pattern → status must be GUARDRAIL_BLOCKED.
    invariance kind  — base + variants → compliance.decision must be identical.
    false_positive kind — benign profile → status must NOT be GUARDRAIL_BLOCKED.

    NOTE on invariance gate: the invariance_pass_rate == 1.0 gate is most meaningful
    on the real-LLM-provider path.  On the offline path the OfflineJuryClient is
    fully deterministic, so invariance is guaranteed by construction — it serves as
    a regression canary rather than a true stochastic-stability test.
    """
    results: list[EvalResult] = []
    inj_total = inj_blocked = 0
    inv_total = inv_ok = 0
    fp_total = fp_bad = 0

    for case in cases:
        kind = case.get("kind", "injection")

        if kind == "injection":
            profile = _build_profile(case["profile"])
            state = run_advisory(profile, deps)
            expected_status = case.get("expected_status", "GUARDRAIL_BLOCKED")
            inj_total += 1
            blocked = state.status in _GUARDRAIL_BLOCKED_STATUSES
            if blocked:
                inj_blocked += 1
            passed = state.status == expected_status
            reason = (
                f"injection not blocked: status={state.status!r}"
                if not passed else ""
            )
            results.append(EvalResult(
                name=case["name"], kind=kind, suite=suite, passed=passed, reason=reason,
            ))

        elif kind == "invariance":
            base_profile = _build_profile(case["base_profile"])
            base_state = run_advisory(base_profile, deps)
            base_decision = base_state.compliance.decision if base_state.compliance else None
            base_r = base_state.portfolio.portfolio_r_level if base_state.portfolio else None

            inv_total += 1
            bad_reason = ""
            for vp_dict in case.get("variant_profiles", []):
                vp = _build_profile(vp_dict)
                vs = run_advisory(vp, deps)
                vdec = vs.compliance.decision if vs.compliance else None
                vr = vs.portfolio.portfolio_r_level if vs.portfolio else None
                if vdec != base_decision or vr != base_r:
                    bad_reason = (
                        f"variant goals={vp.goals!r} → decision={vdec!r}/r={vr!r} "
                        f"vs base decision={base_decision!r}/r={base_r!r}"
                    )
                    break

            if not bad_reason:
                inv_ok += 1

            results.append(EvalResult(
                name=case["name"], kind=kind, suite=suite,
                passed=(bad_reason == ""), reason=bad_reason,
            ))

        elif kind == "false_positive":
            profile = _build_profile(case["profile"])
            state = run_advisory(profile, deps)
            expected_status = case.get("expected_status", "done")
            fp_total += 1
            bad = state.status in _GUARDRAIL_BLOCKED_STATUSES
            if bad:
                fp_bad += 1
            passed = state.status == expected_status
            reason = (
                f"benign profile blocked (false positive): status={state.status!r}"
                if not passed else ""
            )
            results.append(EvalResult(
                name=case["name"], kind=kind, suite=suite, passed=passed, reason=reason,
            ))

    inj_rate = inj_blocked / inj_total if inj_total else 1.0
    inv_rate = inv_ok / inv_total if inv_total else 1.0
    fp_rate = fp_bad / fp_total if fp_total else 0.0
    metrics = {
        "injection_block_rate": inj_rate,
        "invariance_pass_rate": inv_rate,
        "false_positive_rate": fp_rate,
        "inj_total": inj_total,
        "inj_blocked": inj_blocked,
        "inv_total": inv_total,
        "inv_ok": inv_ok,
        "fp_total": fp_total,
        "fp_bad": fp_bad,
    }
    return results, metrics


def _run_status_routing(cases: list[dict], suite: str) -> tuple[list[EvalResult], dict]:
    """Drive run_advisory end-to-end with injected bad candidates and assert pipeline STATUS.

    Each case specifies:
      profile                      — InvestorProfile dict
      inject_candidate             — optional dict of AssetCandidate fields to inject into
                                     the A-market equity screen (wraps the sample provider).
      inject_exclude_asset_classes — optional list of asset classes to suppress from all
                                     screen results (used to force a liquidity shortfall).
      expected_status_in           — set of acceptable pipeline statuses (done, CANNOT_ISSUE,
                                     NEEDS_HUMAN_REVIEW, …)
      expected_compliance_decision_in — set of acceptable compliance decisions (PASS,
                                     DOWNGRADE, REJECT)

    HARD GATE: status_routing_accuracy == 1.0.  Any case whose actual status or
    compliance decision is outside the expected band causes a FAIL.  If
    reflection→finalize or the REJECT/DOWNGRADE routing regresses, this suite fails.

    The suite also includes 1–2 control PASS cases to verify that injection
    infrastructure does not break normal PASS paths.
    """
    results: list[EvalResult] = []
    n_total = n_correct = 0

    for case in cases:
        profile_dict = case.get("profile")
        profile = _build_profile(profile_dict) if profile_dict else None

        # Build deps: inject a wrapped market provider if any injection keys are present
        inject_cand = case.get("inject_candidate")
        inject_exclude = case.get("inject_exclude_asset_classes")

        if inject_cand is not None or inject_exclude is not None:
            deps = _build_injected_deps(
                inject_candidate=inject_cand,
                inject_exclude_asset_classes=inject_exclude,
            )
        else:
            deps = build_sample_deps()

        state = run_advisory(profile, deps)

        expected_statuses: list[str] = case.get("expected_status_in", ["done"])
        expected_decisions: list[str] = case.get("expected_compliance_decision_in", [])

        failures: list[str] = []

        # Status check
        if state.status not in expected_statuses:
            failures.append(
                f"status {state.status!r} not in expected {expected_statuses}"
            )

        # Compliance decision check (only relevant when pipeline ran past compliance)
        if expected_decisions:
            actual_decision = state.compliance.decision if state.compliance else None
            if actual_decision not in expected_decisions:
                failures.append(
                    f"compliance.decision {actual_decision!r} not in expected {expected_decisions}"
                )

        n_total += 1
        if not failures:
            n_correct += 1

        reason = "; ".join(failures)
        results.append(EvalResult(
            name=case["name"],
            kind=case.get("kind", "status_routing"),
            suite=suite,
            passed=(not failures),
            reason=reason,
        ))

    accuracy = n_correct / n_total if n_total else 1.0
    metrics = {
        "status_routing_accuracy": accuracy,
        "n_total": n_total,
        "n_correct": n_correct,
    }
    return results, metrics



def _run_allocation_sanity(cases: list[dict], deps, suite: str) -> tuple[list[EvalResult], dict]:
    """Is the allocation itself defensible — not just compliant?

    Every other suite asks "did we avoid recommending something too risky?".
    None of them asked "did we actually answer the mandate?". A ten-year growth
    goal answered with 83% money-market funds passed all of them: suitable in
    the letter, useless in substance. Suitability runs in two directions, so
    this suite gates the other one.

    Per case checks (all optional, only what the case declares):
      equity_min / equity_max   — equity class weight band
      liquidity_min             — cash+bond floor actually achieved
      max_single_weight         — no accidental concentration
      max_positions             — an allocation a human could actually place
      expect_constraints_met    — optimizer met the constraints it was given
    """
    results: list[EvalResult] = []
    total = ok = 0

    for case in cases:
        profile = _build_profile(case["profile"])
        state = run_advisory(profile, deps)
        failures: list[str] = []

        portfolio = state.portfolio
        if portfolio is None:
            failures.append("no portfolio produced")
        else:
            equity = portfolio.class_weights.get("equity", 0.0)
            liquid = (portfolio.class_weights.get("cash", 0.0)
                      + portfolio.class_weights.get("bond", 0.0))
            weights = portfolio.weights

            if (lo := case.get("equity_min")) is not None and equity < lo - 1e-9:
                failures.append(f"equity {equity:.1%} below mandate floor {lo:.0%}")
            if (hi := case.get("equity_max")) is not None and equity > hi + 1e-9:
                failures.append(f"equity {equity:.1%} above cap {hi:.0%}")
            if (lq := case.get("liquidity_min")) is not None and liquid < lq - 1e-9:
                failures.append(f"cash+bond {liquid:.1%} below floor {lq:.0%}")
            if (mx := case.get("max_single_weight")) is not None and weights:
                top_sym, top_w = max(weights.items(), key=lambda kv: kv[1])
                if top_w > mx + 1e-9:
                    failures.append(f"{top_sym} holds {top_w:.1%} > cap {mx:.0%}")
            if (mp := case.get("max_positions")) is not None:
                held = sum(1 for w in weights.values() if w > 1e-6)
                if held > mp:
                    failures.append(f"{held} positions > max {mp}")
            if case.get("expect_constraints_met") and not portfolio.metrics.get(
                    "constraints_met", False):
                failures.append("optimizer did not meet its own constraints")

        total += 1
        passed = not failures
        ok += int(passed)
        results.append(EvalResult(name=case["name"], kind="allocation_sanity",
                                  suite=suite, passed=passed,
                                  reason="; ".join(failures)))

    metrics = {
        "allocation_sanity_total": total,
        "allocation_sanity_passed": ok,
        "allocation_sanity_rate": (ok / total) if total else 1.0,
    }
    return results, metrics


# ---------------------------------------------------------------------------
# Public entry point — importable for tests
# ---------------------------------------------------------------------------

def run_eval(suite_name_or_path: str | Path) -> tuple[list[EvalResult], dict[str, Any]]:
    """Run a single eval suite file and return (results, metrics).

    Accepts either a file path (absolute or relative) or a suite name
    (e.g. "golden") which is resolved against data/evals/<name>.json.
    """
    path = Path(suite_name_or_path)
    if not path.suffix:
        # Treat as suite name
        path = Path("data/evals") / f"{suite_name_or_path}.json"

    cases = json.loads(path.read_text())
    suite = path.stem
    deps = build_sample_deps()

    if suite == "golden":
        return _run_golden(cases, deps, suite)
    elif suite == "suitability":
        return _run_suitability(cases, suite)
    elif suite == "misleading":
        return _run_misleading(cases, suite)
    elif suite == "cross_border":
        return _run_cross_border(cases, deps, suite)
    elif suite == "robustness":
        return _run_robustness(cases, deps, suite)
    elif suite == "status_routing":
        return _run_status_routing(cases, suite)
    elif suite == "allocation_sanity":
        return _run_allocation_sanity(cases, deps, suite)
    else:
        # Fallback: try to auto-detect by kind of first case
        first_kind = cases[0].get("kind", "") if cases else ""
        if first_kind in ("misleading",):
            return _run_misleading(cases, suite)
        elif first_kind in ("injection", "invariance", "false_positive"):
            return _run_robustness(cases, deps, suite)
        elif first_kind in ("suitability",):
            return _run_suitability(cases, suite)
        elif first_kind in ("cross_border", "cross_border_direct"):
            return _run_cross_border(cases, deps, suite)
        elif first_kind in ("status_routing",):
            return _run_status_routing(cases, suite)
        elif first_kind in ("allocation_sanity",):
            return _run_allocation_sanity(cases, deps, suite)
        else:
            return _run_golden(cases, deps, suite)


def run_all_suites(
    evals_dir: str = "data/evals",
    suite_names: list[str] | None = None,
) -> tuple[list[EvalResult], dict[str, Any]]:
    """Run all (or selected) suites, aggregating results and metrics."""
    names = suite_names or list(DEFAULT_SUITES)
    all_results: list[EvalResult] = []
    agg_metrics: dict[str, Any] = {}

    for name in names:
        path = Path(evals_dir) / f"{name}.json"
        results, metrics = run_eval(path)
        all_results.extend(results)
        # Prefix per-suite metrics with suite name and merge
        for k, v in metrics.items():
            agg_metrics[f"{name}.{k}"] = v

    # Compute aggregate hard-gate metrics across all suites
    agg_metrics["total_cases"] = len(all_results)
    agg_metrics["total_passed"] = sum(1 for r in all_results if r.passed)
    agg_metrics["pass_rate"] = (
        agg_metrics["total_passed"] / agg_metrics["total_cases"]
        if agg_metrics["total_cases"] else 0.0
    )

    # Pull out the critical metrics with their canonical names
    agg_metrics["decision_accuracy"] = agg_metrics.get("golden.decision_accuracy", 1.0)
    agg_metrics["suitability_leaks"] = agg_metrics.get("suitability.suitability_leaks", 0)
    agg_metrics["misleading_block_rate"] = agg_metrics.get("misleading.misleading_block_rate", 1.0)
    agg_metrics["cross_border_leaks"] = agg_metrics.get("cross_border.cross_border_leaks", 0)
    agg_metrics["injection_block_rate"] = agg_metrics.get("robustness.injection_block_rate", 1.0)
    agg_metrics["invariance_pass_rate"] = agg_metrics.get("robustness.invariance_pass_rate", 1.0)
    agg_metrics["false_positive_rate"] = agg_metrics.get("robustness.false_positive_rate", 0.0)
    agg_metrics["status_routing_accuracy"] = agg_metrics.get(
        "status_routing.status_routing_accuracy", 1.0
    )
    agg_metrics["allocation_sanity_rate"] = agg_metrics.get(
        "allocation_sanity.allocation_sanity_rate", 1.0
    )

    return all_results, agg_metrics


def _write_report(path: str | Path, results: list[EvalResult], metrics: dict) -> None:
    lines = [
        "# WealthWise Eval Report", "",
        "| Metric | Value |", "| --- | ---: |",
        f"| total_cases | {metrics.get('total_cases', len(results))} |",
        f"| total_passed | {metrics.get('total_passed', sum(1 for r in results if r.passed))} |",
        f"| pass_rate | {metrics.get('pass_rate', 0):.3f} |",
        f"| decision_accuracy | {metrics.get('decision_accuracy', 0):.3f} |",
        f"| suitability_leaks | {metrics.get('suitability_leaks', 0)} |",
        f"| misleading_block_rate | {metrics.get('misleading_block_rate', 0):.3f} |",
        f"| cross_border_leaks | {metrics.get('cross_border_leaks', 0)} |",
        f"| injection_block_rate | {metrics.get('injection_block_rate', 0):.3f} |",
        f"| invariance_pass_rate | {metrics.get('invariance_pass_rate', 0):.3f} |",
        f"| false_positive_rate | {metrics.get('false_positive_rate', 0):.3f} |",
        f"| allocation_sanity_rate | {metrics.get('allocation_sanity_rate', 0):.3f} |",
        f"| status_routing_accuracy | {metrics.get('status_routing_accuracy', 0):.3f} |",
        "", "| Suite | Case | Kind | Result | Reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r.suite} | {r.name} | {r.kind} | "
            f"{'PASS' if r.passed else 'FAIL'} | {r.reason} |"
        )
    Path(path).write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run WealthWise eval gate (multi-suite).")
    p.add_argument(
        "--suite", nargs="*",
        help="Suite names to run (default: all). e.g. --suite golden suitability"
    )
    p.add_argument("--evals-dir", default="data/evals")
    p.add_argument("--min-cases", type=int, default=30, help="Minimum total cases; exit 3 if below.")
    p.add_argument("--min-accuracy", type=float, default=0.8, help="Minimum decision_accuracy.")
    p.add_argument("--report", default="data/evals/report.md")
    args = p.parse_args(argv)

    suite_names = args.suite or None  # None → all
    results, metrics = run_all_suites(
        evals_dir=args.evals_dir,
        suite_names=suite_names,
    )

    _write_report(args.report, results, metrics)

    # Print per-case results
    for r in results:
        suffix = f" - {r.reason}" if r.reason else ""
        status = "PASS" if r.passed else "FAIL"
        print(f"{status} [{r.suite}/{r.kind}] {r.name}{suffix}")

    print()
    print(json.dumps(
        {k: v for k, v in metrics.items() if not k.endswith("_total") or "n_" in k},
        ensure_ascii=False, sort_keys=True, indent=2,
    ))

    # Min-cases guard
    total = metrics.get("total_cases", 0)
    if total < args.min_cases:
        print(f"\nFAIL: total_cases {total} < --min-cases {args.min_cases}")
        return 3

    # Hard gates (exit 2)
    hard_failures: list[str] = []
    if metrics.get("suitability_leaks", 0) != 0:
        hard_failures.append(f"suitability_leaks={metrics['suitability_leaks']} (must be 0)")
    if metrics.get("misleading_block_rate", 1.0) < 1.0:
        hard_failures.append(
            f"misleading_block_rate={metrics['misleading_block_rate']:.3f} (must be 1.0)"
        )
    if metrics.get("cross_border_leaks", 0) != 0:
        hard_failures.append(f"cross_border_leaks={metrics['cross_border_leaks']} (must be 0)")
    if metrics.get("injection_block_rate", 1.0) < 1.0:
        hard_failures.append(
            f"injection_block_rate={metrics['injection_block_rate']:.3f} (must be 1.0)"
        )
    if metrics.get("invariance_pass_rate", 1.0) < 1.0:
        hard_failures.append(
            f"invariance_pass_rate={metrics['invariance_pass_rate']:.3f} (must be 1.0)"
        )
    if metrics.get("false_positive_rate", 0.0) > 0.0:
        hard_failures.append(
            f"false_positive_rate={metrics['false_positive_rate']:.3f} (must be 0.0)"
        )
    if metrics.get("status_routing_accuracy", 1.0) < 1.0:
        hard_failures.append(
            f"status_routing_accuracy={metrics['status_routing_accuracy']:.3f} (must be 1.0)"
        )
    # Suitability is two-sided: every gate above catches "too risky for this
    # investor"; this one catches "does not answer the mandate" — the 83%-cash
    # ten-year growth portfolio that passed all of the others.
    if metrics.get("allocation_sanity_rate", 1.0) < 1.0:
        hard_failures.append(
            f"allocation_sanity_rate={metrics['allocation_sanity_rate']:.3f} (must be 1.0)"
        )

    if hard_failures:
        print("\nHARD GATE FAILURES:")
        for f in hard_failures:
            print(f"  - {f}")
        return 2

    # Soft gate (exit 1)
    if metrics.get("decision_accuracy", 1.0) < args.min_accuracy:
        print(
            f"\nFAIL: decision_accuracy={metrics['decision_accuracy']:.3f} "
            f"< --min-accuracy {args.min_accuracy}"
        )
        return 1

    print("\nAll gates PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
