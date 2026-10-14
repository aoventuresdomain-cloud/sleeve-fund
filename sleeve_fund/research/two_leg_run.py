"""P2-8 two-leg backtest: one two-leg Strategy (spot + perp carry, or a pair) in ONE NautilusTrader run, each leg a
strategy of its own on its own clone of the venue (a cash account for spot, a margin account for a perp), with its own
fees, funding and journal, coupled by a Pair (strategies.pair). The Strategy's balance is split between the legs by
what each needs (two_legs.split_capital), as Store.create_two_legs splits it in paper.

The lead leg's model decides; the other leg (LegFollower) trades only with it. The result keeps each leg's own record,
as run_backtest gives it, and the Strategy's: equity summed over both legs, fees and funding per leg and in total."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from decimal import Decimal

import pandas as pd

from sleeve_fund import two_legs
from sleeve_fund.research.portfolio import clone_instrument, clone_venue
from sleeve_fund.research.runner import (
    BacktestResult,
    add_leg,
    feed_legs,
    leg_result,
    new_engine,
    plan_run,
    release_leg,
)
from sleeve_fund.strategies.base import LongFlatConfig
from sleeve_fund.strategies.pair import LegFollower, Pair


@dataclass(frozen=True)
class LegRun:
    """One leg of a two-leg backtest: the leg as Store.create_two_legs takes it (instrument pair, params, venue, side,
    ratio, leverage) with the engine instrument it trades and its bars (decision bars stamped at their close, and
    optional shorter exec bars). Without bars of its own a leg trades the other leg's (a perp simulated on spot)."""

    leg: dict
    instrument: object
    prices: pd.DataFrame | None = None
    exec_prices: pd.DataFrame | None = None


@dataclass
class TwoLegResult:
    lead: BacktestResult
    follow: BacktestResult
    shares: tuple[Decimal, Decimal]  # each leg's opening balance
    pair_log: list  # the Pair's steps: (ts ns, what, detail)
    trades: int  # pairs opened
    equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))  # both legs summed, at each close

    @property
    def fees_paid(self) -> float:
        return self.lead.fees_paid + self.follow.fees_paid

    @property
    def funding_paid(self) -> float:
        """Funding received (+) or paid (-), over both legs: only a perp leg has any."""
        return sum(f["amount"] for r in (self.lead, self.follow) for f in r.funding)


def run_two_legs(strategy: str, first: LegRun, second: LegRun, starting_capital: float = 10_000.0, *,
                 params: dict | None = None, bar_minutes: int = 1440, exec_minutes: int = 1,
                 risk_profile: str | None = None, warmup_prices: pd.DataFrame | None = None,
                 first_out: str | None = None, log_level: str = "ERROR") -> TwoLegResult:
    """Backtest the two-leg Strategy whose first leg runs `strategy` with `params` (its model's settings, merged over
    the first leg's own market params) and whose second leg follows it. Each leg is checked and the balance split as
    Store.create_two_legs does (a short spot leg, a perp above 2x and a pair across venues are refused)."""
    lead_leg, follow_leg = two_legs.check_legs(first.leg, second.leg)
    shares = two_legs.split_capital(starting_capital, (lead_leg, follow_leg))
    prices = first.prices
    if prices is None or prices.empty:
        raise ValueError("the first leg needs its decision bars")
    lead_plan = plan_run(strategy, prices, first.instrument, {**(lead_leg.get("params") or {}), **(params or {})},
                         float(shares[0]), bar_minutes=bar_minutes, exec_prices=first.exec_prices,
                         exec_minutes=exec_minutes, risk_profile=risk_profile, warmup_prices=warmup_prices)
    f_prices = second.prices if second.prices is not None else prices
    f_exec = second.exec_prices if second.prices is not None else first.exec_prices
    # The follower has no model: planned as the benchmark (no weight sizing to refuse on a perp), run as a LegFollower.
    follow_plan = plan_run("buy_and_hold", f_prices, second.instrument, dict(follow_leg.get("params") or {}),
                           float(shares[1]), bar_minutes=bar_minutes, exec_prices=f_exec, exec_minutes=exec_minutes,
                           risk_profile=risk_profile)
    follow_plan = dataclasses.replace(follow_plan, strategy_name="two_leg_follower", strategy_cls=LegFollower,
                                      config_cls=LongFlatConfig)
    engine = new_engine(log_level)
    legs = []
    try:
        for n, (run, plan) in enumerate(((first, lead_plan), (second, follow_plan)), start=1):
            clone = clone_instrument(run.instrument, clone_venue(run.instrument.id.venue, n))
            legs.append(add_leg(engine, plan, clone, order_id_tag=f"{n:03d}"))
        pair = Pair(legs[0].strategy, legs[1].strategy, lead_side=lead_leg["side"], follow_side=follow_leg["side"],
                    ratio=follow_leg["ratio"], first_out=first_out, name="two_leg")
        feed_legs(engine, legs)
        lead, follow = leg_result(engine, legs[0]), leg_result(engine, legs[1])
    finally:
        for leg in legs:
            release_leg(leg)
        engine.dispose()
    equity = lead.equity.add(follow.equity, fill_value=None).dropna()
    return TwoLegResult(lead, follow, shares, list(pair.log), pair.trades, equity)
