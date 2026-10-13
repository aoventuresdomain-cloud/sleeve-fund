"""STOP-LOOSEN-1 (QD's finding, Independent Quant Advisor 17:10 UK, 13 Oct): the half-liquidation rule is a standing cap
on every stop level for the life of a margin position, not an entry check. A level that loosens past half the way to
the CURRENT liquidation price (isolated margin moves with funding) is clamped there, never flattened; tightening is
unchanged. Each clamp is journaled, the first of a position alerts, open risk reads the clamped stop and the tear
sheet counts the trades clamped. One engine path for backtest and paper (strategies.base._capped_stop)."""

import dataclasses

import numpy as np
import pandas as pd
import pytest
import test_hub_146_qa as qa
from test_hub_146_qa import _probe  # noqa: F401 - the autouse fixture that registers the probe strategy

from sleeve_fund import markets
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.store import Store
from sleeve_fund.strategies.definitions import to_params
from sleeve_fund.strategies.rules import Rules

PERP = {"market": "perp", "allow_short": True}
LEVERAGE = {"balanced": 2.0, "aggressive": 3.0}
# Gapped 1-minute bars (each opens, trades and closes at one price), so nothing trades through a stop inside a bar.
# A 3-bar average entered with a high close still in its window (a 11% stop), which loosens to 29% on the next bar
# as that close leaves it: past half the way to liquidation at 2x (about 25%) and at 3x (about 16.5%). The entry
# fills at 150 on that bar; on the bar after, the average catches up with the close and the level exit ends the leg.
PATHS = {1: ([300, 300, 230, 20, 150, 150, 150, 150], {"long": {"entry": {"left": "close", "op": ">", "right": 100}}}),
         -1: ([40, 40, 70, 280, 150, 150, 150, 150], {"short": {"entry": {"left": "close", "op": "<", "right": 200}}})}
LOOSENED = {1: (20 + 150 + 150) / 3, -1: (280 + 150 + 150) / 3}
MAINTENANCE = markets.LOW_FEE_PERP.maintenance_margin


def _bars(closes: list[float], start: str = "2025-10-03 00:01", minutes: int = 1) -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(closes), freq=f"{minutes}min", tz="UTC")
    c = np.array(closes, float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e3}, index=idx)


def _defn(legs: dict, blocks: dict, exits: dict) -> dict:
    return {"version": 1, "reason": "A test case, written before it is run.", "blocks": blocks, **legs, "exits": exits}


def _spy_stops(monkeypatch) -> list[dict]:
    """The stop each rules candle holds once its levels have moved: (entry, side, stop_frac)."""
    seen = []
    real = Rules._move_levels

    def spy(self):
        out = real(self)
        if self._entry_px is not None:
            seen.append({"entry": self._entry_px, "side": self._entry_side, "frac": self._stop_frac})
        return out

    monkeypatch.setattr(Rules, "_move_levels", spy)
    return seen


def _entry(res) -> float:
    """The entry's fill price (the venue's fee schedule can move it off the close by a tick's worth)."""
    entry = next(o for o in res.fills.sort_values("ts_last").index if res.decisions[o]["intent"] == "entry")
    return float(res.fills.loc[entry, "avg_px"])


def _liq(entry: float, side: int, lev: float) -> float:
    """The isolated liquidation price at this leverage (margin.isolated_liquidation): the margin is the notional over
    the leverage, whatever the size, so the price is the entry's and the leverage's alone."""
    return entry * (1 - side / lev) / (1 - side * MAINTENANCE)


def _held(seen: list[dict], entry: float) -> list[float]:
    return [entry * (1 - s["side"] * s["frac"]) for s in seen if s["entry"] == pytest.approx(entry, rel=1e-12)]


def _events(res, kind: str) -> list[dict]:
    return [e for e in res.journal.events_ if e["kind"] == kind]


@pytest.mark.parametrize("profile", ["balanced", "aggressive"])
@pytest.mark.parametrize("side", [1, -1])
def test_a_level_loosening_past_half_way_to_liquidation_is_held_at_half_way(side, profile, instrument, monkeypatch):
    closes, legs = PATHS[side]
    seen = _spy_stops(monkeypatch)
    defn = _defn(legs, {"ma": {"kind": "sma", "period": 3}}, {"exit_at_level": {"level": "ma"}})
    res = run_backtest("rules", _bars(closes), instrument, params={**to_params(defn), **PERP}, bar_minutes=1,
                       half_spread=0, risk_profile=profile)
    assert not res.handler_errors, res.handler_errors
    entry = _entry(res)
    liq = _liq(entry, side, LEVERAGE[profile])
    cap = entry - side * 0.5 * abs(entry - liq)  # half the way from the entry to the liquidation price
    assert (LOOSENED[side] - cap) * side < 0, "the path must loosen past the cap, or the pin proves nothing"
    held = _held(seen, entry)
    assert held and all(h == pytest.approx(cap, rel=1e-9) for h in held)  # held there, not at the average's level
    (alert,) = _events(res, "stop_clamped_liquidation")  # the first clamp of the position alerts, once
    assert alert["level"] == "warning" and f"{liq:,.6g}" in alert["message"]
    assert res.stops_clamped == 1


def test_a_stop_that_tightens_is_unchanged_on_margin(instrument, monkeypatch):
    """A ratcheted trail only ever tightens: no clamp, no event, nothing counted."""
    seen = _spy_stops(monkeypatch)
    legs = {"long": {"entry": {"left": "close", "op": ">", "right": 103}}}
    defn = _defn(legs, {"dc": {"kind": "donchian", "period": 3}}, {"trail": {"level": "dc.lower"}})
    res = run_backtest("rules", _bars([100, 100, 104, 108, 112, 116, 120, 124]), instrument, params={**to_params(defn), **PERP}, bar_minutes=1,
                       half_spread=0, risk_profile="aggressive")
    assert not res.handler_errors, res.handler_errors
    held = _held(seen, _entry(res))
    assert len(held) > 2 and held == sorted(held) and held[-1] > held[0]  # it tightened, as the trail moved it
    assert not _events(res, "stop_clamped_liquidation") and not _events(res, "stop_clamped")
    assert res.stops_clamped == 0


def test_funding_paid_moves_the_liquidation_price_and_a_fixed_stop_is_pulled_in(instrument, monkeypatch):
    """A long at 3x with a stop just inside half way at entry. 1% funding paid at the 08:00 UTC settlement comes out of
    its isolated margin and lifts the liquidation price by about 1.5, so half way moves above the stop: the stop is
    pulled in to half way to the new price on that candle, and open risk counts it from there."""
    monkeypatch.setattr(markets, "LOW_FEE_PERP", dataclasses.replace(markets.LOW_FEE_PERP, funding_rate=0.01))
    seen = _spy_stops(monkeypatch)
    legs = {"long": {"entry": {"left": "close", "op": ">", "right": 140}}}
    defn = _defn(legs, {"dc": {"kind": "donchian", "period": 3}}, {"stop": {"level": "dc.lower"}})
    closes = [200, 125.5, 150, 150, 150, 150, 150, 150]  # hourly, from 03:00 UTC: held across 08:00
    res = run_backtest("rules", _bars(closes, "2025-10-03 03:00", 60), instrument,
                       params={**to_params(defn), **PERP}, bar_minutes=60, half_spread=0, risk_profile="aggressive")
    assert not res.handler_errors, res.handler_errors
    paid = [f for f in res.funding if f["amount"] < 0]
    assert paid, "no funding was paid while the position was held"
    entry = _entry(res)
    liq = _liq(entry, 1, 3.0)
    assert 125.5 > entry - 0.5 * (entry - liq)  # inside half way at entry, so it opened
    held = _held(seen, entry)
    assert held[0] == pytest.approx(125.5)  # before the settlement: the stop as it opened
    assert held[-1] > 125.5 + 0.5  # after it: pulled in towards the price
    (alert,) = _events(res, "stop_clamped_liquidation")
    moved = float(alert["message"].split("liquidation price ")[1].split(" ")[0].replace(",", ""))
    assert moved > liq + 1.0  # the price it was held against is the moved one, not the entry's
    assert held[-1] == pytest.approx(entry - 0.5 * (entry - moved), rel=1e-5)
    assert res.stops_clamped == 1


@pytest.mark.parametrize("paid", [0.0, -500.0], ids=["no-funding", "funding-paid"])
def test_after_a_restart_the_funding_paid_since_the_entry_still_pulls_the_stop_in(paid, monkeypatch):
    """Paper, restarted holding a 3x long of 0.25 at 60,000 with a 16% stop (half way to liquidation is 16.5%). 500 of
    funding paid before the restart lifts the liquidation price by about 2,010, so half way is now about 14.8%: the
    restarted strategy holds the stop there from its first price, and a 15.5% fall stops it out. Without that funding
    the same fall stays above the 16% stop
    (the daily-loss pause may still close it)."""
    real = Store.funding

    def with_paid(self, sleeve, limit=1000):
        rows = real(self, sleeve, limit)
        if limit is None and paid:
            entry_ts = pd.Timestamp(qa.START + 5 * qa.M, unit="ns", tz="UTC").to_pydatetime()
            rows = [{"ts": entry_ts + pd.Timedelta(seconds=30), "amount": paid}, *rows]
        return rows

    monkeypatch.setattr(Store, "funding", with_paid)
    record = Store.record_order

    def with_liq(self, sleeve, **kw):  # the price paper journals on the entry's fill (_set_liq), 3x long
        if kw.get("order_id") == "O-held":
            kw["signal"] = {**kw["signal"], "position_liquidation_px": qa.BASE * (2 / 3) / (1 - MAINTENANCE)}
        return record(self, sleeve, **kw)

    monkeypatch.setattr(Store, "record_order", with_liq)
    p = qa.shape(qa.flat_prices(30), 14.0, 30, qa.adverse(1, 0.155))
    run = qa.restart(p, 6, 12, side=1, perp=True, profile="aggressive", stop=0.16, qty=0.25)
    seq = run.sequence()
    exits = [r for r in seq if r[0] != "entry"]
    if paid:
        assert exits and exits[0][0] == "stop_loss", seq
        assert [e for e in run.events if e["kind"] == "stop_clamped_liquidation"]
    else:  # the daily-loss pause may close it, but no stop fires
        assert not [r for r in exits if r[0] == "stop_loss"], seq
