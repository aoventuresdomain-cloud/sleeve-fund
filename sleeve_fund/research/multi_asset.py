"""A multi-asset Strategy's backtest (multi-asset spec §2, §4, §6; Advisor MA-SPEC): a portfolio run of its legs, the
parent's first asset and one "<name>~<ASSET>" leg per other, each its own idea sized off the fund (each leg asks for its
full size; the compliance tests trim it, as on paper), gated in one pass per close strongest idea first, then by the
asset's place in the list. Results per asset and pooled, with the fixed-list warning on every report."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from sleeve_fund import markets
from sleeve_fund.research.metrics import round_trips
from sleeve_fund.research.portfolio_run import LegSpec, PortfolioResult, run_portfolio
from sleeve_fund.risk import PORTFOLIO, PortfolioProfile

FIXED_LIST = "Fixed list chosen on {date}; history may favour assets that survived."


@dataclass(frozen=True)
class AssetData:
    """One asset's bars for the run: as LegSpec takes them."""

    instrument: Any
    prices: pd.DataFrame
    exec_prices: pd.DataFrame
    warmup_prices: pd.DataFrame


def underlying(instrument) -> str:
    return str(getattr(getattr(instrument, "id", instrument), "symbol", instrument)).split("/")[0].upper()


def leg_names(name: str, instruments: list) -> list[str]:
    """The run's names: the Strategy's own for its first asset, "<name>~<UNDERLYING>" for each other (Store.create_legs)."""
    return [name] + [f"{name}~{underlying(i)}" for i in instruments[1:]]


def check_assets(assets: list[AssetData]) -> None:
    """Every asset on the same bar clock over the whole window, so no leg trades through a gap the others don't have; one
    that fails refuses the whole Strategy and names the asset (spec §2)."""
    if len(assets) < 2:
        raise ValueError("a multi-asset Strategy lists at least two assets")
    seen = [underlying(a.instrument) for a in assets]
    if len(set(seen)) != len(seen):
        raise ValueError(f"an asset is listed twice: {seen}")
    clock = assets[0].prices.index
    for a in assets[1:]:
        if not a.prices.index.equals(clock):
            missing = clock.difference(a.prices.index)
            raise ValueError(f"{underlying(a.instrument)}'s bars aren't on the first asset's clock over the window"
                             + (f" (missing {len(missing)}, first {missing[0]})" if len(missing) else ""))


@dataclass
class MultiAssetResult:
    name: str
    universe_date: str
    run: PortfolioResult
    per_asset: list[dict] = field(default_factory=list)  # {asset, leg, trades, pnl, fees, share}
    pooled: dict = field(default_factory=dict)  # {trades, pnl, fees}

    @property
    def warning(self) -> str:
        return FIXED_LIST.format(date=self.universe_date)


def run_multi_asset(name: str, strategy: str, assets: list[AssetData], params: dict, universe_date: str,
                    capital: float = 10_000.0, profile: PortfolioProfile = PORTFOLIO, **leg) -> MultiAssetResult:
    """Run the Strategy over its fixed list. capital: the fund's, split evenly across the legs' accounts (each leg's
    account only posts margin; every leg sizes off the fund's book, run_portfolio(fund_sized=True))."""
    check_assets(assets)
    names = leg_names(name, [a.instrument for a in assets])
    specs = [LegSpec(n, strategy, a.instrument, a.prices, a.exec_prices, a.warmup_prices, params=dict(params),
                     capital=capital / len(assets), **leg) for n, a in zip(names, assets)]
    run = run_portfolio(specs, profile, fund_sized=True)
    out = MultiAssetResult(name, str(universe_date), run)
    total = 0.0
    for n, a in zip(names, assets):
        r = run.legs[n]
        pnl = float(r.equity.iloc[-1]) - float(r.starting_capital) if len(r.equity) else 0.0
        trades = len(round_trips(r.fills, shorts=markets.is_perp(params))) if r.fills is not None and not r.fills.empty else 0
        out.per_asset.append({"asset": underlying(a.instrument), "leg": n, "trades": trades, "pnl": pnl,
                              "fees": float(r.fees_paid)})
        total += pnl
    for row in out.per_asset:
        row["share"] = row["pnl"] / total if total else 0.0
    out.pooled = {"trades": sum(r["trades"] for r in out.per_asset), "pnl": total,
                  "fees": sum(r["fees"] for r in out.per_asset)}
    return out
