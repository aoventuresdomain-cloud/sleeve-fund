"""Risk profiles and the sleeve-level guard (plan section 6, layer 2).

The guard is a pure function so it can be tested exhaustively and later moved
out of the strategy process into an independent risk service.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta


OWN_LEVERAGE = "max_leverage"  # B-35 (a): a Strategy's own leverage cap, under its risk profile's


@dataclass(frozen=True)
class RiskProfile:
    name: str
    max_drawdown: float  # from peak equity: flatten and halt; only the PM restarts
    daily_loss: float  # from the UTC day's opening equity: flatten and pause 24 hours
    max_position_pct: float  # largest position as a share of sleeve equity
    max_open_positions: int
    # Strategies that may go short trade a perpetual on margin (sleeve_fund.markets). Their gross exposure
    # stays under this many times equity, a stop sits at most stop_to_liquidation of the way to the
    # liquidation price, and an open position is cut back once the price comes within
    # min_liquidation_distance of liquidation (long/short verdict, 4 Oct 2026: risk limits for both sides).
    max_leverage: float = 1.0
    stop_to_liquidation: float = 0.5
    min_liquidation_distance: float = 0.10


PROFILES: dict[str, RiskProfile] = {
    p.name: p
    for p in (
        RiskProfile("conservative", max_drawdown=0.10, daily_loss=0.03, max_position_pct=0.20, max_open_positions=3,
                    max_leverage=1.0, stop_to_liquidation=0.33, min_liquidation_distance=0.15),
        RiskProfile("balanced", max_drawdown=0.20, daily_loss=0.05, max_position_pct=0.33, max_open_positions=3,
                    max_leverage=2.0, stop_to_liquidation=0.5, min_liquidation_distance=0.10),
        RiskProfile("aggressive", max_drawdown=0.35, daily_loss=0.08, max_position_pct=0.50, max_open_positions=2,
                    max_leverage=3.0, stop_to_liquidation=0.5, min_liquidation_distance=0.07),
    )
}


@dataclass(frozen=True)
class PortfolioProfile:
    """The limits across the whole book (v2 P2-2), each a share of the fund's marked equity. The defaults are the
    balanced ones the PM accepted on 6 Oct 2026. version: the profile a decision was taken under, journaled with it."""
    version: int = 1
    gross: float = 1.5  # sum of every position's notional
    net_instrument: float = 0.5  # long less short notional in one underlying, across venues and contract types
    margin: float = 0.5  # posted isolated margin, spot at its full notional
    open_risk: float = 0.05  # loss if every stop is hit, stopless positions at their measured move
    drawdown: float = 0.15  # from the high-water mark: flatten and halt every strategy; only the PM restarts
    daily_loss: float = 0.03  # from the 00:00 UTC book: no new entries until the next 00:00 UTC
    # Bridge Drop A (Advisor COMPLIANCE-5): loss if every stop is hit, per asset (T1) and per asset group (T2), on new
    # entries only; None = switched off, as a profile from before the compliance tests is. The PM's seeded values are
    # portfolio.compliance.SEEDED. groups: {underlying: group}, the PM's map, versioned with the tests; with no map at
    # all T2 has nothing to judge, with one an asset left out of it is refused while T2 is on (Advisor BRIDGE-A-FE-2).
    asset_open_risk: float | None = None
    group_open_risk: float | None = None
    groups: Mapping[str, str] | None = field(default=None, compare=False)


PORTFOLIO = PortfolioProfile()


def profile(name: str) -> RiskProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown risk profile {name!r}; known: {sorted(PROFILES)}") from None


@dataclass(frozen=True)
class Breach:
    action: str  # "halt" | "pause_day"
    reason: str


def trading_day(ts: datetime) -> date:
    """The UTC day a mark belongs to, for the daily-loss guard. A mark at exactly 00:00 is the day before's
    last: a backtest's bar that closes at midnight is stamped then, so the new day opens at its equity, as
    paper's day opens at its last mark before midnight (coordinator, 5 Oct: it opened an hour early)."""
    return (ts - timedelta(microseconds=1)).date()


def check(profile: RiskProfile, equity: float, peak: float, day_open: float, halts: bool = True) -> Breach | None:
    """Return the breach to act on, worst first, or None. Bad inputs fail closed. halts=False (an ablation's 1x pair
    only, CR F235-5) leaves out the drawdown halt and the daily-loss pause; bad inputs still halt."""
    if not all(x == x and x > 0 for x in (equity, peak, day_open)):  # NaN or non-positive
        return Breach("halt", f"invalid equity data (equity={equity}, peak={peak}, day_open={day_open})")
    if not halts:
        return None
    drawdown = 1 - equity / max(peak, equity)
    if drawdown >= profile.max_drawdown:
        return Breach("halt", f"drawdown {drawdown:.1%} hit the {profile.max_drawdown:.0%} limit")
    day_loss = 1 - equity / day_open
    if day_loss >= profile.daily_loss:
        return Breach("pause_day", f"daily loss {day_loss:.1%} hit the {profile.daily_loss:.0%} limit")
    return None


def own_leverage(params: dict | None) -> float | None:
    """B-35 (a): a Strategy's own leverage cap (params["max_leverage"]), or None when it has none or it isn't a number
    of at least 1x to 2 dp (the model's config refuses such a value on save and Start, check_own_leverage). Below
    1x is refused, never a cap (Advisor): the venue's minimum is 1x; smaller exposure is size share, not leverage."""
    v = (params or {}).get(OWN_LEVERAGE)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 1:
        return None
    if round(v, 2) != v:  # Advisor B-35 (a): to 2 dp, the venue's leverage step
        return None
    return float(v)


def check_own_leverage(v) -> None:
    """B-35 (a): raises ValueError for a Strategy's own leverage cap that isn't a number of at least 1x to 2 dp."""
    if v is not None and own_leverage({OWN_LEVERAGE: v}) is None:
        raise ValueError(f"{OWN_LEVERAGE} {v!r} must be a number of times, minimum 1x, to 2 decimal places (the "
                         "venue's step): minimum 1x; use size share to trade smaller. It can lower the risk profile's "
                         "leverage cap, never raise it")


def capped_leverage(p: RiskProfile, params: dict | None = None) -> float:
    """B-35 (a): the profile's leverage cap, lowered by the Strategy's own (own_leverage), never raised by it."""
    own = own_leverage(params)
    return p.max_leverage if own is None else min(own, p.max_leverage)


def leverage_cap(p: RiskProfile, params: dict | None = None) -> float:
    """The leverage a position is margined at: the profile's cap lowered by the Strategy's own (capped_leverage, B-35
    (a)), or a two-leg Strategy's perp leg's own setting (params["leg"]["leverage"]) when lower, never above 2x (P2-8,
    Advisor 05:35 UK: min(leg, 2x, profile)). The one figure for its margin, liquidation price and the gate's book
    (LongFlatStrategy._max_leverage, fund_holdings)."""
    cap = capped_leverage(p, params)
    leg = (params or {}).get("leg") or {}
    if leg.get("leverage") is None:
        return cap
    return min(float(leg["leverage"]), 2.0, cap)


def position_cap(p: RiskProfile, params: dict | None = None) -> float:
    """The largest position's notional as a multiple of equity. On spot, the profile's position cap. On a
    perpetual the same cap applies to the margin, and the notional is that margin times the leverage cap
    (PM, 5 Oct 2026): balanced puts at most 33% of equity up as margin, at 2x a notional of 66% (at the
    Strategy's own lower cap, B-35 (a), that cap's). The
    stop-to-liquidation and liquidation-distance guards still bound it."""
    from sleeve_fund import markets

    return p.max_position_pct * capped_leverage(p, params) if markets.is_perp(params) else p.max_position_pct
