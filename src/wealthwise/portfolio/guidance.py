"""Execution guidance — how to put the plan on, and how to keep it there.

A holdings list answers "what to buy" and stops. The questions that follow it are
the ones a retail investor actually gets wrong: all at once or staged, how often
to rebalance, through which account, and whether they are even eligible to hold
the cross-border sleeve.

All of it is rule-derived. None of these answers depend on a model, and routing
them through one would add latency and a hallucination surface to arithmetic.
"""
from __future__ import annotations

from wealthwise.agents.state import InvestorProfile, PortfolioAllocation
from wealthwise.portfolio.execution import ExecutionPlan

# Staging exists to blunt entry-point risk, which scales with the equity sleeve —
# a bond-and-cash book has little to average into.
_STAGE_BANDS = ((0.20, 1), (0.50, 3), (1.01, 4))
_STAGE_INTERVAL_WEEKS = 4

# Shorter horizons need tighter drift control; longer ones can let winners run
# rather than paying turnover to trim them every quarter.
_REBALANCE_BY_HORIZON = ((3, "每季度"), (7, "每半年"), (999, "每年"))

# Class drift that justifies trading. Below this, costs outweigh the correction.
_DRIFT_TRIGGER_PP = 5.0

# Northbound (港股通) eligibility: 20 trading days' average daily assets of at
# least 500,000 CNY across securities and cash.
_HK_CONNECT_THRESHOLD = 500_000.0


def _entry_plan(equity_weight: float) -> dict:
    stages = next(n for ceiling, n in _STAGE_BANDS if equity_weight < ceiling)
    if stages == 1:
        detail = "权益占比低，择时收益有限，可一次性建仓。"
    else:
        detail = (
            f"权益占比 {equity_weight:.0%}，建议分 {stages} 期建仓，"
            f"每期间隔约 {_STAGE_INTERVAL_WEEKS} 周、等额买入，"
            "以摊薄单一入场点的择时风险。固收与现金部分可一次性到位。"
        )
    return {"stages": stages, "interval_weeks": _STAGE_INTERVAL_WEEKS,
            "detail": detail}


def _rebalance_plan(horizon_years: int) -> dict:
    cadence = next(label for ceiling, label in _REBALANCE_BY_HORIZON
                   if horizon_years <= ceiling)
    return {
        "cadence": cadence,
        "drift_trigger_pp": _DRIFT_TRIGGER_PP,
        "detail": (
            f"{cadence}检视一次；任一大类实际占比偏离目标超过 "
            f"{_DRIFT_TRIGGER_PP:.0f} 个百分点时提前再平衡。"
            "再平衡只在大类之间做，个股不因短期涨跌调整。"
        ),
    }


def _channels(plan: ExecutionPlan, profile: InvestorProfile) -> list[str]:
    markets = {p.market for p in plan.positions}
    classes = {p.asset_class for p in plan.positions}
    notes: list[str] = []

    if "A" in markets:
        notes.append("A 股与场内 ETF：通过证券账户按手（100 股/份）买入，T+1 交收。")
    if "bond" in classes or "cash" in classes:
        notes.append(
            "债券 ETF 与货币 ETF 均为场内品种，可用同一证券账户买卖；"
            "货币 ETF 支持 T+0，是流动性仓位的落点。")
    if "HK" in markets:
        eligible = profile.investable >= _HK_CONNECT_THRESHOLD
        gate = ("账户资产已满足港股通 50 万门槛，可走港股通"
                if eligible else
                f"账户资产 {profile.investable:,.0f} 元未达港股通 50 万门槛，"
                "需通过 QDII 基金或合规跨境券商参与")
        notes.append(f"港股：{gate}；注意每手股数各不相同（本方案已按实际每手计算）。")
    if "US" in markets:
        notes.append(
            "美股：内地投资者需通过 QDII 额度或合规跨境券商买入，"
            "留意额度开放情况与汇兑成本。")
    return notes


def build_guidance(profile: InvestorProfile, portfolio: PortfolioAllocation,
                   plan: ExecutionPlan) -> dict:
    """Derive entry pacing, rebalancing rules and account routing for `plan`."""
    equity_weight = portfolio.class_weights.get("equity", 0.0)
    return {
        "entry": _entry_plan(equity_weight),
        "rebalance": _rebalance_plan(profile.horizon_years),
        "channels": _channels(plan, profile),
    }
