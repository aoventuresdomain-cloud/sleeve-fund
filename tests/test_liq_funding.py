"""LIQ-FUND-1 (Independent Quant Advisor, 13 Oct 17:20 UK: a MUST before any perp Strategy runs on paper). The liquidation
guard read the liquidation price fixed at the fill and ignored the funding booked since, so a position whose margin
funding had drained survived past the price the venue would take it at. Every check now reads _liq_now: the price fixed
at the fill moved by the funding booked to the position's isolated margin since, p' = p + f / (mm x |q| - q). A payment
brings it closer for a long and a short alike; a credit moves it away.

Hourly settlements from 07:50 UTC, so one falls at 08:00 inside the run; a large fixed rate makes the move plain. The
price then gaps to a level between the fixed and the moved liquidation price: the moved one decides."""

import dataclasses

import pandas as pd
import pytest
import test_hub_146_qa as qa
from test_hub_146_full_round import _funding_rate_stub
from test_hub_146_qa import (  # noqa: F401 - _probe registers the probe
    LIQUIDATION_MECHANICS,
    _probe,
    flat_prices,
)

MM = 0.005
SETTLED = pd.Timestamp("2025-10-03 08:00", tz="UTC")
pytestmark = list(LIQUIDATION_MECHANICS)  # a stopless 3x position on purpose: the guard is the subject


@pytest.fixture
def hourly(monkeypatch):
    """Hourly settlements, the run from 07:50, and the aggressive profile's drawdown and daily-loss limits lifted:
    the funding drained here would pause it first, and the liquidation guard is the subject."""
    from sleeve_fund import markets, risk

    aggressive = risk.PROFILES["aggressive"]
    monkeypatch.setitem(risk.PROFILES, "aggressive", dataclasses.replace(aggressive, max_drawdown=0.99, daily_loss=0.99))

    monkeypatch.setattr(markets, "LOW_FEE_PERP", dataclasses.replace(markets.LOW_FEE_PERP, funding_hours=tuple(range(24))))
    monkeypatch.setattr(qa, "START", int(pd.Timestamp("2025-10-03 07:50", tz="UTC").value))


def _rate(monkeypatch, rate):
    from sleeve_fund.strategies.base import LongFlatStrategy

    monkeypatch.setattr(LongFlatStrategy, "_funding_rate", _funding_rate_stub(LongFlatStrategy, rate))


def moved(fixed: float, qty: float, funded: float) -> float:
    return fixed + funded / (MM * abs(qty) - qty)


def _prices(side: int, depth: float, n: int = 40):
    """Flat to minute 20, then a gap `depth` against the position held to the end."""
    return qa.shape(flat_prices(n), 20, n, qa.adverse(side, depth))



def _backtest(prices, side):
    """qa.backtest on the same minutes, returning the result itself: the journal's orders with their signals."""
    from sleeve_fund.instruments import BOOK_SHARE
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.venues import venue

    inst = venue("KRAKEN").instrument("BTC", "USD", price_precision=1)
    params = {"enter": 5, "leave": 35, "side": side, "stop_loss": None, "market": "perp", "allow_short": True}
    m = qa.minutes_of(prices)
    bars = m.set_axis(m.index + pd.Timedelta(minutes=1))
    bars["volume"] = 60 / BOOK_SHARE
    return run_backtest("probe", bars, inst, params=params, starting_capital=10_000, risk_profile="aggressive",
                        bar_minutes=1, half_spread=qa.SPREAD / 2 / qa.BASE)


def _paper(prices, side):
    return qa.paper(prices, side=side, perp=True, profile="aggressive", stop=None, leave=35)


def _book(where, prices, side):
    """(orders oldest first with intent and signal, the entry's fixed liquidation price, qty signed, funding since)."""
    from sleeve_fund import markets

    if where == "paper":
        run = _paper(prices, side)
        orders = run.orders
        entry = next(o for o in orders if o["intent"] == "entry")
        fixed = entry["signal"]["position_liquidation_px"]
        fill = next(f for f in run.fills if f["order_id"] == entry["order_id"])
        funded = sum(float(r["amount"]) for r in run.store.funding("q146"))
    else:
        res = _backtest(prices, side)
        fills = sorted(res.journal.fills_, key=lambda f: f["ts"])
        orders = [{"intent": res.decisions[f["order_id"]]["intent"], **res.decisions[f["order_id"]]} for f in fills]
        fill = fills[0]
        assert orders[0]["intent"] == "entry", orders
        cash = 10_000 - side * fill["qty"] * fill["price"] - fill["fee"]
        fixed = markets.isolated_liquidation(cash, side * fill["qty"], fill["price"], 3.0, MM)
        funded = sum(r["amount"] for r in res.funding if pd.Timestamp(r["ts"]) > pd.Timestamp(fill["ts"]))
    return orders, fixed, side * fill["qty"], funded


# A payment brings the price closer: a gap to a level the fixed price is well clear of (past the 7% cut too) is through
# the moved one, and the position is liquidated there. Long pays a positive rate, short a negative one.
PAYS = [pytest.param(1, 0.10, 0.25, id="long"), pytest.param(-1, -0.12, 0.225, id="short")]


@pytest.mark.parametrize("where", ["backtest", "paper"])
@pytest.mark.parametrize(("side", "rate", "depth"), PAYS)
def test_funding_paid_moves_the_liquidation_price_closer_and_the_position_is_liquidated_there(hourly, monkeypatch,
                                                                                            where, side, rate, depth):
    _rate(monkeypatch, rate)
    p = _prices(side, depth)
    orders, fixed, qty, funded = _book(where, p, side)
    assert funded < 0, ("set-up: the 08:00 settlement was paid", funded)
    now = moved(fixed, qty, funded)
    gap = float(p[-1])
    assert (now - fixed) * side > 0, "a payment moves it towards the price"
    assert (gap - fixed) * side / gap > 0.07, "set-up: the fixed price is clear of the gap and of the 7% cut"
    assert (gap - now) * side <= 0, "set-up: the gap is through the moved price"
    liq = [o for o in orders if o["intent"] == "liquidation"]
    assert liq, [o["intent"] for o in orders]
    assert liq[0]["signal"]["liquidation_px"] == pytest.approx(now, rel=1e-6)
    assert liq[0]["reason"].startswith("Liquidated"), liq[0]["reason"]


# A credit moves it away: a gap through the fixed price that stays clear of the moved one (and of the cut) is survived.
CREDITS = [pytest.param(1, -0.10, 0.3417, id="long"), pytest.param(-1, 0.12, 0.3416, id="short")]


@pytest.mark.parametrize("where", ["backtest", "paper"])
@pytest.mark.parametrize(("side", "rate", "depth"), CREDITS)
def test_a_funding_credit_moves_the_liquidation_price_away(hourly, monkeypatch, where, side, rate, depth):
    _rate(monkeypatch, rate)
    p = _prices(side, depth)
    orders, fixed, qty, funded = _book(where, p, side)
    assert funded > 0, ("set-up: the 08:00 settlement was a credit", funded)
    now = moved(fixed, qty, funded)
    gap = float(p[-1])
    assert (fixed - now) * side > 0, "a credit moves it away from the price"
    assert (gap - fixed) * side <= 0, "set-up: the gap is through the fixed price"
    assert (gap - now) * side / gap > 0.07, "set-up: the moved price is clear of the gap and of the 7% cut"
    assert not [o for o in orders if o["intent"] in ("liquidation", "liquidation_cut")], [o["intent"] for o in orders]


def test_a_restart_reads_the_funding_since_the_fill_back_from_the_journal(hourly, monkeypatch):
    """Paper restarted at 08:06 holding a long entered at 07:55, its liquidation price journaled on the entry and the
    08:00 settlement booked by the previous process. The new process moves the price by that funding, read from the
    journal (it never sees the settlement itself), so the same gap liquidates it at the moved price."""
    from sleeve_fund import markets
    from sleeve_fund.store import Store

    monkeypatch.setattr(qa, "START", int(pd.Timestamp("2025-10-03 08:06", tz="UTC").value))
    _rate(monkeypatch, 0.0)
    qty, px = 0.25, qa.BASE
    fee = qty * px * 0.0005
    fixed = markets.isolated_liquidation(10_000 - qty * px - fee, qty, px, 3.0, MM)
    funded = -1500.0
    record_order, create = Store.record_order, Store.create_sleeve

    def journaled(self, sleeve, **kw):  # the previous process journaled the entry's liquidation price (_set_liq)
        if kw.get("order_id") == "O-held":
            kw["signal"] = {**(kw.get("signal") or {}), "position_liquidation_px": round(fixed, 8)}
        return record_order(self, sleeve, **kw)

    def with_funding(self, *a, **kw):
        out = create(self, *a, **kw)
        self.record_funding("q146", qty=qty, price=px, rate=0.1, amount=funded, ts=SETTLED.to_pydatetime())
        return out

    monkeypatch.setattr(Store, "record_order", journaled)
    monkeypatch.setattr(Store, "create_sleeve", with_funding)
    now = moved(fixed, qty, funded)
    gap = now * 0.995  # through the moved price, and clear of the fixed price's 7% cut
    p = qa.shape(flat_prices(20), 4, 20, gap / qa.BASE)
    assert (gap - fixed) / gap > 0.07
    held = (qty, px, int(pd.Timestamp("2025-10-03 07:55", tz="UTC").value))
    run = qa.paper(p, side=1, perp=True, profile="aggressive", stop=None, enter=0, leave=19, held=held)
    liq = [o for o in run.orders if o["intent"] == "liquidation"]
    assert liq, [(o["intent"], o["reason"]) for o in run.orders]
    assert liq[0]["signal"]["liquidation_px"] == pytest.approx(now, rel=1e-6)


def _stub(qty: float, fixed: float, funded: float, since: int | None = 0):
    from types import SimpleNamespace

    from sleeve_fund.strategies.base import LongFlatStrategy

    class S:
        _liq_now, _fund_margin = LongFlatStrategy._liq_now, LongFlatStrategy._fund_margin
        _cfg = SimpleNamespace(perp=SimpleNamespace(maintenance_margin=MM))

        def __init__(self):
            self._liq_px, self._liq_funded, self._liq_from = fixed, funded, since

        def _liq(self, cash, qty):
            return self._liq_px if qty else None

    return S()


@pytest.mark.parametrize(("qty", "fixed", "funded", "expected"), [
    pytest.param(1.0, 50.0, -9.95, 60.0, id="long-pays-closer"),
    pytest.param(1.0, 50.0, 9.95, 40.0, id="long-credit-away"),
    pytest.param(-1.0, 150.0, -10.05, 140.0, id="short-pays-closer"),
    pytest.param(-1.0, 150.0, 10.05, 160.0, id="short-credit-away"),
    pytest.param(1.0, 50.0, 0.0, 50.0, id="none-booked"),
    pytest.param(1.0, 5.0, 9.95, None, id="long-credited-past-zero-has-none"),
])
def test_the_moved_price_is_where_equity_meets_the_maintenance_margin(qty, fixed, funded, expected):
    """mm 0.5%: a long's denominator is 0.005 - 1 = -0.995, a short's 0.005 + 1 = 1.005."""
    got = _stub(qty, fixed, funded)._liq_now(0.0, qty)
    assert got == (pytest.approx(expected) if expected is not None else None)


def test_only_funding_settled_after_the_fill_moves_it():
    """A reversal, or a late booking, of a settlement before the fill was the position's before it: not counted."""
    s = _stub(1.0, 50.0, 0.0, since=int(SETTLED.value))
    s._fund_margin(SETTLED.to_pydatetime(), -5.0)  # at the fill's instant: the position before it held that settlement
    s._fund_margin(SETTLED - pd.Timedelta(hours=1), 3.0)
    assert s._liq_funded == 0.0
    s._fund_margin(SETTLED + pd.Timedelta(hours=1), -5.0)
    assert s._liq_funded == -5.0
    s._liq_px = None  # flat: nothing is the margin's
    s._fund_margin(SETTLED + pd.Timedelta(hours=2), -5.0)
    assert s._liq_funded == -5.0


@pytest.fixture
def hourly_guarded(monkeypatch):
    """Hourly settlements from 07:50 with the aggressive profile's own limits on (8% daily loss, 35% drawdown)."""
    from sleeve_fund import markets

    monkeypatch.setattr(markets, "LOW_FEE_PERP", dataclasses.replace(markets.LOW_FEE_PERP, funding_hours=tuple(range(24))))
    monkeypatch.setattr(qa, "START", int(pd.Timestamp("2025-10-03 07:50", tz="UTC").value))



# The same with the aggressive profile's own limits on (the Advisor's worst case: 3x, held through settlements). Which
# fires first: a drain past the 8% daily loss pauses it at the settlement, before any liquidation; a drain inside it
# leaves the position open, and a gap through the moved price then liquidates it there, where the fixed price would
# only have cut it (inside 7% of it) at the old level.
@pytest.mark.parametrize("where", ["backtest", "paper"])
@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_guarded_a_drain_past_the_daily_loss_pauses_it_before_any_liquidation(hourly_guarded, monkeypatch, where,
                                                                               side):
    _rate(monkeypatch, 0.10 * side)  # 15% of equity paid at 08:00
    orders, _, _, funded = _book(where, _prices(side, 0.25 if side > 0 else 0.22), side)
    assert funded < -800
    intents = [o["intent"] for o in orders]
    assert intents == ["entry", "risk_pause"], [(o["intent"], o["reason"]) for o in orders]
    assert "daily loss" in orders[1]["reason"].lower()


@pytest.mark.parametrize("where", ["backtest", "paper"])
@pytest.mark.parametrize(("side", "depth"), [pytest.param(1, 0.2834, id="long"), pytest.param(-1, 0.282, id="short")])
def test_guarded_a_drain_inside_the_limits_then_a_gap_liquidates_at_the_moved_price(hourly_guarded, monkeypatch,
                                                                                     where, side, depth):
    _rate(monkeypatch, 0.05 * side)  # 7.5% of equity paid at 08:00: inside the 8% daily loss
    p = _prices(side, depth)
    orders, fixed, qty, funded = _book(where, p, side)
    now, gap = moved(fixed, qty, funded), float(p[-1])
    assert -800 < funded < 0
    assert (gap - now) * side <= 0 < (gap - fixed) * side, "set-up: the gap is through the moved price only"
    assert [o["intent"] for o in orders] == ["entry", "liquidation"], [(o["intent"], o["reason"]) for o in orders]
    assert orders[1]["signal"]["liquidation_px"] == pytest.approx(now, rel=1e-6)
