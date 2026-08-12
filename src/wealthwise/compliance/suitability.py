"""Investor-suitability C-R hard gate.

China Investor Suitability Framework
-------------------------------------
Investor risk tolerance is classified C1 (most conservative) … C5 (most
aggressive).  Product/portfolio risk is rated R1 (lowest) … R5 (highest).

Core rule (越级 — over-level mismatch):
    A product is suitable only if its R-level ≤ investor's C-level.
    C3 → max acceptable product risk is R3.

Additional mandatory checks:
    Liquidity floor   — cash+bond class weight ≥ profile.liquidity_min.
    Cross-border gate — if investor has not authorized cross-border exposure
                        (accept_cross_border=False), the portfolio must not
                        hold any HK- or US-market assets.

Decision Policy
---------------
PASS
    No violations. matched=True.

DOWNGRADE
    Over-level at the individual-asset or portfolio-aggregate level, OR
    liquidity floor breached.  The portfolio *could* be fixed by de-risking
    or adding cash/bonds — no hard regulatory block, but advisory system must
    not present the allocation as-is.

REJECT
    Hard block that cannot be resolved by re-weighting alone:
      • Unauthorized cross-border exposure (regulatory authorization is
        binary — either the investor signed the cross-border agreement or not).
      • Any single asset exceeds C-level AND the portfolio_r_level also
        exceeds C-level — a compound violation indicating the optimizer failed
        to produce even an in-principle-fixable allocation.

    Note on reachability: `portfolio_r_level` is defined as the max R among held
    assets, so "one asset over-level" implies "portfolio over-level" and the
    compound branch is what actually fires. The asset-only DOWNGRADE case
    described above is therefore unreachable with the current aggregate
    definition, and the DOWNGRADE path is reached via the liquidity floor. Kept
    explicit here so the policy and the code do not drift apart; changing
    `portfolio_r_level` to a weighted measure would make it reachable again.

Confidence
----------
All rules are deterministic — confidence is always 1.0. (compliance_node
composes the *reported* confidence from jury agreement and whether the
optimizer met its constraints; this value is only the rule-layer input.)

Dependencies
------------
R_ORDER is imported from wealthwise.portfolio.metrics (single source of truth).
Do NOT redefine R-level ordering here.
"""
from __future__ import annotations

from wealthwise.agents.state import (
    AssetCandidate,
    ComplianceVerdict,
    InvestorProfile,
    PortfolioAllocation,
)
from wealthwise.portfolio.metrics import R_ORDER

# ---------------------------------------------------------------------------
# C-level → numeric index (mirrors R_ORDER; kept here for suitability logic)
# ---------------------------------------------------------------------------

C_ORDER: dict[str, int] = {
    "C1": 1,
    "C2": 2,
    "C3": 3,
    "C4": 4,
    "C5": 5,
}

# Tolerance for weight comparisons — weights are floats normalized by division,
# so exact-equality boundaries need a little slack.
_FLOAT_TOL: float = 1e-9

# ---------------------------------------------------------------------------
# Pure helper — importable for direct zero-miss testing
# ---------------------------------------------------------------------------


def is_over_level(r_level: str, c_level: str) -> bool:
    """Return True iff product R-level strictly exceeds investor C-level.

    Parameters
    ----------
    r_level:
        Product/asset risk classification, e.g. "R4".
    c_level:
        Investor risk tolerance classification, e.g. "C3".

    Returns
    -------
    bool
        True  → over-level mismatch (越级); the product is NOT suitable.
        False → within tolerance (equal or below); the product IS suitable.
    """
    return R_ORDER.get(r_level, 0) > C_ORDER.get(c_level, 0)


# ---------------------------------------------------------------------------
# Main gate
# ---------------------------------------------------------------------------


def check_suitability(
    profile: InvestorProfile,
    portfolio: PortfolioAllocation,
    candidates: list[AssetCandidate],
) -> ComplianceVerdict:
    """Evaluate whether *portfolio* is suitable for *profile*.

    Parameters
    ----------
    profile:
        Investor suitability profile (risk_level C1–C5, liquidity_min,
        accept_cross_border).
    portfolio:
        Proposed allocation (weights, class_weights, portfolio_r_level).
    candidates:
        Full candidate list used to look up per-asset metadata (r_level,
        market).  Assets with weight > 0 in portfolio.weights that are not
        found in candidates default to R5/A-market (fail closed: unknown
        ratings treated as highest risk R5).

    Returns
    -------
    ComplianceVerdict
        decision: PASS | DOWNGRADE | REJECT
        matched: True iff no suitability violations found
        violations: human-readable violation strings
        disclosures: mandatory disclosure statements
        confidence: 1.0 (deterministic)

    Decision policy is documented in the module docstring.
    """
    c_level = profile.risk_level
    violations: list[str] = []

    # Build lookup maps from candidates
    symbol_r: dict[str, str] = {c.symbol: c.r_level for c in candidates}
    symbol_market: dict[str, str] = {c.symbol: c.market for c in candidates}

    # ------------------------------------------------------------------
    # 1. Per-asset over-level check
    # ------------------------------------------------------------------
    asset_over_level_found = False
    for symbol, weight in portfolio.weights.items():
        if weight <= 0:
            continue
        r = symbol_r.get(symbol, "R5")  # fail closed: unknown ratings treated as highest risk R5
        if is_over_level(r, c_level):
            asset_over_level_found = True
            violations.append(
                f"{symbol}: R-level {r} exceeds investor {c_level}"
            )

    # ------------------------------------------------------------------
    # 2. Portfolio-aggregate over-level check
    # ------------------------------------------------------------------
    port_r = portfolio.portfolio_r_level
    portfolio_over_level = is_over_level(port_r, c_level)
    if portfolio_over_level:
        violations.append(
            f"Portfolio aggregate {port_r} exceeds investor {c_level}"
        )

    # ------------------------------------------------------------------
    # 3. Liquidity floor check
    # ------------------------------------------------------------------
    liquid_weight = (
        portfolio.class_weights.get("cash", 0.0)
        + portfolio.class_weights.get("bond", 0.0)
    )
    # Tolerance matters here: an optimizer that lands exactly on the floor
    # produces 0.19999999999999998, and a bare `<` then reports a shortfall of
    # "20.00% < required 20.00%" — a violation that exists only in binary.
    if liquid_weight < profile.liquidity_min - _FLOAT_TOL:
        violations.append(
            f"Liquidity shortfall: cash+bond weight {liquid_weight:.2%} "
            f"< required {profile.liquidity_min:.2%}"
        )

    # ------------------------------------------------------------------
    # 4. Cross-border authorization gate
    # ------------------------------------------------------------------
    cross_border_violation = False
    if not profile.accept_cross_border:
        for symbol, weight in portfolio.weights.items():
            if weight <= 0:
                continue
            market = symbol_market.get(symbol, "A")
            if market in {"HK", "US"}:
                cross_border_violation = True
                violations.append(
                    f"Cross-border unauthorized: {symbol} ({market}) held "
                    f"but investor has not authorized cross-border exposure"
                )

    # ------------------------------------------------------------------
    # Decision policy
    # ------------------------------------------------------------------
    if not violations:
        decision = "PASS"
        matched = True
    elif cross_border_violation:
        # Hard block: regulatory authorization is binary
        decision = "REJECT"
        matched = False
    elif asset_over_level_found and portfolio_over_level:
        # Compound over-level: optimizer produced a fundamentally unsuitable
        # allocation — REJECT rather than expecting a re-weight to fix it
        decision = "REJECT"
        matched = False
    else:
        # Over-level at asset level only, or liquidity breach — de-risking
        # or adding cash/bonds could fix it
        decision = "DOWNGRADE"
        matched = False

    # ------------------------------------------------------------------
    # Disclosures (always generated — four mandatory kinds)
    # ------------------------------------------------------------------
    # 1. Suitability-match statement naming C-level and R-level
    suitability_stmt = (
        f"投资者风险等级 {c_level}，组合风险等级 {port_r}，"
        + ("符合适当性匹配要求。" if matched else "不符合适当性匹配要求，存在违规项目，请审查。")
    )
    disclosures: list[str] = [suitability_stmt]

    # 2. Risk disclosure
    disclosures.append("投资有风险，入市须谨慎，过往业绩不代表未来表现。")

    # 3. Disclaimer
    disclosures.append("本内容不构成投资建议，仅供参考，请结合自身情况审慎决策。")

    # 4. Cross-border / FX disclosure. Keyed on the *actual* holdings first —
    #    the authorization flag says what the investor allowed, not what the
    #    portfolio ended up holding, and the disclosure has to describe the
    #    latter. (compliance_node quantifies the exposure on top of this.)
    if portfolio.fx_exposure > 0:
        disclosures.append("跨境标的涉及汇率波动、通道（港股通/QDII）与税收风险。")
    elif profile.accept_cross_border:
        disclosures.append("本组合当前未持有境外资产；如后续加入，将涉及汇率波动、通道与税收风险。")
    else:
        disclosures.append("投资者未授权跨境投资，组合不得持有境外资产。")

    return ComplianceVerdict(
        decision=decision,
        matched=matched,
        violations=violations,
        disclosures=disclosures,
        confidence=1.0,
    )
