"""P2-1b's diagnostic (Advisor, 6 Oct 17:09, ruling 3): a converted variant against its legacy weight version on spot."""

from __future__ import annotations

from sleeve_fund.research.metrics import turnover_per_year, years_covered

# The params that make a converted variant (P2-1b): central sizing, and the band variant's band.
CONVERSION_KEYS = ("sizing", "resize_band")

GAP_RELATIVE = 0.30  # |converted - legacy| / |legacy| past this flags the pair
GAP_FLOOR = 0.05  # under this |legacy| return, an absolute gap instead
GAP_POINTS = 0.015


def gross_gap_flag(converted: float, legacy: float) -> bool:
    """Whether the gross returns differ enough to investigate the conversion before trusting either result: more than
    30% of the legacy return, or 1.5 points when the legacy return is under 5% either way. A flag, never a gate."""
    gap = abs(converted - legacy)
    return bool(gap > GAP_POINTS if abs(legacy) < GAP_FLOOR else gap > GAP_RELATIVE * abs(legacy))


def cost_profile(res) -> dict:
    """A run's turnover (traded notional a year over average equity), fee drag (fees a year over average equity) and
    gross return (the return with the fees paid added back), as the tearsheet reckons them."""
    equity = res.equity
    if equity is None or not len(equity):
        return {"turnover": 0.0, "fee_drag": 0.0, "gross_return": 0.0}
    years = years_covered(equity) or 1.0
    mean = float(equity.mean()) or 1.0
    start = float(res.starting_capital)
    return {"turnover": turnover_per_year(res.fills, equity), "fee_drag": float(res.fees_paid) / mean / years,
            "gross_return": (float(equity.iloc[-1]) + float(res.fees_paid)) / start - 1}
