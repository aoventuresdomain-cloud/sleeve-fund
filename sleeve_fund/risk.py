"""Risk profiles and the sleeve-level guard (plan section 6, layer 2).

The guard is a pure function so it can be tested exhaustively and later moved
out of the strategy process into an independent risk service.
"""

from __future__ import annotations

from dataclasses import dataclass


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


def profile(name: str) -> RiskProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown risk profile {name!r}; known: {sorted(PROFILES)}") from None


@dataclass(frozen=True)
class Breach:
    action: str  # "halt" | "pause_day"
    reason: str


def check(profile: RiskProfile, equity: float, peak: float, day_open: float) -> Breach | None:
    """Return the breach to act on, worst first, or None. Bad inputs fail closed."""
    if not all(x == x and x > 0 for x in (equity, peak, day_open)):  # NaN or non-positive
        return Breach("halt", f"invalid equity data (equity={equity}, peak={peak}, day_open={day_open})")
    drawdown = 1 - equity / max(peak, equity)
    if drawdown >= profile.max_drawdown:
        return Breach("halt", f"drawdown {drawdown:.1%} hit the {profile.max_drawdown:.0%} limit")
    day_loss = 1 - equity / day_open
    if day_loss >= profile.daily_loss:
        return Breach("pause_day", f"daily loss {day_loss:.1%} hit the {profile.daily_loss:.0%} limit")
    return None
