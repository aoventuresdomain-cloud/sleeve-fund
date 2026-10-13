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
        _liq_now = LongFlatStrategy._liq_now
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
    from sleeve_fund.strategies.base import fund_margin

    s = _stub(1.0, 50.0, 0.0, since=int(SETTLED.value))
    fund_margin(s, SETTLED.to_pydatetime(), -5.0)  # at the fill's instant: the position before it held that settlement
    fund_margin(s, SETTLED - pd.Timedelta(hours=1), 3.0)
    assert s._liq_funded == 0.0
    fund_margin(s, SETTLED + pd.Timedelta(hours=1), -5.0)
    assert s._liq_funded == -5.0
    s._liq_px = None  # flat: nothing is the margin's
    fund_margin(s, SETTLED + pd.Timedelta(hours=2), -5.0)
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


def _ledger(where, prices, side):
    """After a run that liquidated: (end cash, margin, entry fee, liquidation fee, funding, liquidation fill's P&L)."""
    from sleeve_fund import markets

    if where == "paper":
        run = _paper(prices, side)
        intents = {o["order_id"]: o["intent"] for o in run.orders}
        fills = [{**f, "intent": intents[f["order_id"]]} for f in run.fills]
        funding = sum(float(r["amount"]) for r in run.store.funding("q146"))
        cash = float(run.store.journal_book("q146", 10_000)["cash"])
    else:
        res = _backtest(prices, side)
        fills = [{**f, "intent": res.decisions[f["order_id"]]["intent"]} for f in sorted(res.journal.fills_,
                                                                                        key=lambda f: f["ts"])]
        funding = sum(r["amount"] for r in res.funding)
        cash = float(res.equity.iloc[-1])
    assert [f["intent"] for f in fills] == ["entry", "liquidation"], [(f["intent"], f["price"]) for f in fills]
    entry, liq = fills
    margin = markets.isolated_margin(entry["qty"], entry["price"], 3.0)
    pnl = side * liq["qty"] * (liq["price"] - entry["price"])
    return cash, margin, entry["fee"], liq["fee"], funding, pnl


# Advisor 17:45 UK (13 Oct), MUST: bankruptcy moves by the same funding, b' = b - f / q, so a liquidation loses exactly
# the isolated margin plus the entry and liquidation fees, the funding inside that, never on top (#155). The funding
# lines and the liquidation's line sum to it; a credit is taken with the margin.
LOSSES = [pytest.param(1, 0.05, 0.2834, id="long-paid"), pytest.param(-1, -0.05, 0.282, id="short-paid"),
          pytest.param(1, -0.05, 0.40, id="long-credited"), pytest.param(-1, 0.05, 0.40, id="short-credited")]


@pytest.mark.parametrize("where", ["backtest", "paper"])
@pytest.mark.parametrize(("side", "rate", "depth"), LOSSES)
def test_a_liquidation_loses_exactly_the_margin_and_fees_with_the_funding_inside_it(hourly_guarded, monkeypatch,
                                                                                    where, side, rate, depth):
    _rate(monkeypatch, rate)
    cash, margin, entry_fee, liq_fee, funding, pnl = _ledger(where, _prices(side, depth), side)
    assert abs(funding) > 100, ("set-up: the 08:00 settlement moved the margin", funding)
    lost = margin + entry_fee + liq_fee
    assert 10_000 - cash == pytest.approx(lost, abs=0.02)
    assert funding + pnl - entry_fee - liq_fee == pytest.approx(-lost, abs=0.02)


# F242-1 (CR, HoE MUST FIX 13 Oct): the funding a position's margin has taken stays the position's when it is added to,
# as the venue's isolated margin keeps it, and _liq_now moves the combined position's price by it: p' = p + F / (mm x |q|
# - q) and b' = b - F / q, F all the funding since it opened and q all of it after the add. Only a position opened from
# flat, or through flat, counts afresh. Here 0.125 is entered at 07:55, 0.10 x its value paid (or credited) at 08:00,
# 0.125 more added at 08:02 (a rebalance), then the price gaps at 08:10 between the price fixed at the add and the one
# moved by the 08:00 funding.
ADD_QTY = "0.125"


def _add_probe_classes():
    from decimal import Decimal

    from nautilus_trader.model import OrderSide

    Probe, ProbeConfig = qa._probe_classes()

    class AddProbe(Probe):
        """Fixed sizes, so the combined position stays well inside its isolated margin's balance: an entry at `enter`,
        an add (a rebalance, OPENING_INTENTS) at `add_at`, flat at `leave`."""

        def _on_bar_sided(self, bar) -> None:
            k, close = self._k(bar), bar.close.as_double()
            if self._busy() or self._pending_exit is not None:
                return
            if k in (self.enter, self.add_at) and self._pos_side() in (0, self.side):
                side = OrderSide.BUY if self.side > 0 else OrderSide.SELL
                self._submit(side, Decimal(ADD_QTY), "entry" if k == self.enter else "rebalance", "pin",
                             {"close": close})
            elif k == self.leave and self._pos_side():
                self._sell_all("exit", "pin", {"close": close})

    return AddProbe, ProbeConfig


@pytest.fixture
def adds(hourly, monkeypatch):
    from sleeve_fund.strategies import REGISTRY

    monkeypatch.setitem(REGISTRY, "probe", _add_probe_classes())


def _add_run(where, prices, side):
    """(intents oldest first with their signals, the add's fixed liquidation price, qty after the add signed, funding
    since the entry, fills oldest first with intents, the run's end cash)."""
    from sleeve_fund import markets

    if where == "paper":
        run = qa.paper(prices, side=side, perp=True, profile="aggressive", stop=None, enter=5, leave=35,
                       extra={"add_at": 12})
        intents = {o["order_id"]: o for o in run.orders}
        fills = [{**f, "intent": intents[f["order_id"]]["intent"]} for f in run.fills]
        orders = list(run.orders)
        funded = sum(float(r["amount"]) for r in run.store.funding("q146"))
        cash = float(run.store.journal_book("q146", 10_000)["cash"])
    else:
        from sleeve_fund.instruments import BOOK_SHARE
        from sleeve_fund.research.runner import run_backtest
        from sleeve_fund.venues import venue

        inst = venue("KRAKEN").instrument("BTC", "USD", price_precision=1)
        params = {"enter": 5, "leave": 35, "side": side, "stop_loss": None, "market": "perp", "allow_short": True,
                  "add_at": 12}
        m = qa.minutes_of(prices)
        bars = m.set_axis(m.index + pd.Timedelta(minutes=1))
        bars["volume"] = 60 / BOOK_SHARE
        res = run_backtest("probe", bars, inst, params=params, starting_capital=10_000, risk_profile="aggressive",
                           bar_minutes=1, half_spread=qa.SPREAD / 2 / qa.BASE)
        fills = [{**f, "intent": res.decisions[f["order_id"]]["intent"]}
                 for f in sorted(res.journal.fills_, key=lambda f: f["ts"])]
        orders = [{**res.decisions[f["order_id"]]} for f in fills]
        funded = sum(r["amount"] for r in res.funding if pd.Timestamp(r["ts"]) > pd.Timestamp(fills[0]["ts"]))
        cash = float(res.equity.iloc[-1])
    opened = [f for f in fills if f["intent"] in ("entry", "rebalance")]
    assert [f["intent"] for f in opened] == ["entry", "rebalance"], [(f["intent"], f["qty"]) for f in fills]
    q = sum(f["qty"] for f in opened)
    avg = sum(f["qty"] * f["price"] for f in opened) / q
    fixed = markets.isolated_liquidation(10_000.0 - side * q * avg, side * q, avg, 3.0, MM)  # balance clear of margin
    return orders, fixed, side * q, funded, fills, cash


ADDS = [pytest.param(1, 0.10, 0.30, id="long"), pytest.param(-1, -0.10, 0.30, id="short")]


@pytest.mark.parametrize("where", ["backtest", "paper"])
@pytest.mark.parametrize(("side", "rate", "depth"), ADDS)
def test_an_add_keeps_the_funding_taken_before_it_and_the_combined_position_is_liquidated_there(adds, monkeypatch,
                                                                                               where, side, rate,
                                                                                               depth):
    _rate(monkeypatch, rate)
    p = _prices(side, depth)
    orders, fixed, qty, funded, fills, _ = _add_run(where, p, side)
    assert funded < 0 and all(pd.Timestamp(f["ts"]) > SETTLED for f in fills if f["intent"] == "rebalance"), \
        ("set-up: the 08:00 settlement was paid before the add", funded)
    now, gap = moved(fixed, qty, funded), float(p[-1])
    assert (gap - now) * side <= 0 < (gap - fixed) * side, "set-up: the gap is through the moved price only"
    liq = [o for o in orders if o["intent"] == "liquidation"]
    assert liq, [o["intent"] for o in orders]
    assert liq[0]["signal"]["liquidation_px"] == pytest.approx(now, rel=1e-6)


@pytest.mark.parametrize("where", ["backtest", "paper"])
@pytest.mark.parametrize(("side", "rate", "depth"), ADDS)
def test_across_an_add_a_liquidation_still_loses_exactly_the_margin_and_fees(adds, monkeypatch, where, side, rate,
                                                                             depth):
    from sleeve_fund import markets

    _rate(monkeypatch, rate)
    _, _, qty, funded, fills, cash = _add_run(where, _prices(side, depth), side)
    opened = [f for f in fills if f["intent"] in ("entry", "rebalance")]
    (liq,) = [f for f in fills if f["intent"] == "liquidation"]
    margin = sum(markets.isolated_margin(f["qty"], f["price"], 3.0) for f in opened)
    lost = margin + sum(f["fee"] for f in opened) + liq["fee"]
    assert abs(funded) > 100 and abs(liq["qty"]) == pytest.approx(abs(qty))
    assert 10_000 - cash == pytest.approx(lost, abs=0.02)


def _book_stub(post: float, entry_side: int, funded: float, since: int | None):
    """The perp entry book (_track_entry, _set_liq) on a position of `post` after the fill, its funding so far."""
    from decimal import Decimal
    from types import SimpleNamespace

    from sleeve_fund.strategies.base import LongFlatStrategy

    class S:
        _track_entry, _set_liq = LongFlatStrategy._track_entry, LongFlatStrategy._set_liq
        _cfg = SimpleNamespace(perp=SimpleNamespace(maintenance_margin=MM))
        runtime, _backtest = None, True

        def __init__(self):
            self._entry_px, self._entry_qty, self._entry_side = 100.0, 1.0, entry_side
            self._liq_px, self._liq_funded, self._liq_from = 50.0, funded, since
            self._stop_frac = self._tp_frac = self._funding_skip = self._replan_pending = self._plan_entry = None

        def _signed_qty(self):
            return Decimal(str(post))

        def _lot(self):
            return 0.001

        def _mark(self):
            return 0.0, 10_000.0, post, 0.0

        def _liq_from_entry(self, cash, qty):
            return 60.0 if qty else None

    return S()


@pytest.mark.parametrize(("post", "sign", "filled", "kept"), [
    pytest.param(2.0, 1, 1.0, True, id="add-keeps"),
    pytest.param(0.5, -1, 0.5, True, id="partial-reduce-keeps"),
    pytest.param(0.0, -1, 1.0, False, id="flat-resets"),
    pytest.param(-1.0, -1, 2.0, False, id="flip-resets"),
])
def test_only_flat_or_a_flip_starts_the_funding_afresh(post, sign, filled, kept):
    from decimal import Decimal

    s = _book_stub(post, 1, -7.5, int(SETTLED.value))
    opening = s._track_entry(sign, Decimal(str(filled)), filled, 100.0)
    if opening:
        s._set_liq("O-1", int(SETTLED.value) + 10**9)
    if kept:
        assert (s._liq_funded, s._liq_from) == (-7.5, int(SETTLED.value))
    else:
        assert s._liq_funded == 0.0 and s._liq_from in (None, int(SETTLED.value) + 10**9)


@pytest.mark.parametrize(("fills", "opened"), [
    pytest.param([("BUY", 1, 1), ("BUY", 1, 3), ("SELL", 0.5, 5)], 1, id="entry-add-reduce"),
    pytest.param([("BUY", 1, 1), ("SELL", 1, 2), ("SELL", 1, 3), ("SELL", 1, 4)], 3, id="closed-then-short-added"),
    pytest.param([("BUY", 1, 1), ("SELL", 2, 2), ("SELL", 1, 4)], 2, id="through-flat-then-added"),
])
def test_a_restart_counts_the_funding_from_the_positions_first_opening_fill(fills, opened):
    """The CR's read-back: from the first fill since the journal was last flat or went through flat, not the add's."""
    from sleeve_fund.strategies.base import LongFlatStrategy

    s = _book_stub(0.0, 0, 0.0, None)
    rows = [{"side": side, "qty": q, "ts": SETTLED + pd.Timedelta(minutes=m)} for side, q, m in fills]
    got = LongFlatStrategy._position_opened(s, list(reversed(rows)))
    assert got == int((SETTLED + pd.Timedelta(minutes=opened)).value)


def test_a_restart_after_an_add_reads_back_the_funding_since_the_first_fill(hourly, monkeypatch):
    """As the restart above, the long held entered as 0.125 at 06:55 and 0.125 added at 07:55 (the add's liquidation
    price journaled), 0.0125 x 60,000 paid at 07:00 and again at 08:00: both settlements move the price."""
    from sleeve_fund import markets
    from sleeve_fund.store import Store

    monkeypatch.setattr(qa, "START", int(pd.Timestamp("2025-10-03 08:06", tz="UTC").value))
    _rate(monkeypatch, 0.0)
    qty, px, half = 0.25, qa.BASE, 0.125
    fee = qty * px * 0.0005
    fixed = markets.isolated_liquidation(10_000 - qty * px - fee, qty, px, 3.0, MM)
    record_order, create = Store.record_order, Store.create_sleeve

    def journaled(self, sleeve, **kw):
        if kw.get("order_id") == "O-held":
            kw["signal"] = {**(kw.get("signal") or {}), "position_liquidation_px": round(fixed, 8)}
        return record_order(self, sleeve, **kw)

    def with_first_fill_and_funding(self, *a, **kw):
        out = create(self, *a, **kw)
        first = (SETTLED - pd.Timedelta(minutes=65)).to_pydatetime()
        self.record_order("q146", order_id="O-first", side="BUY", qty=half, intent="entry", reason="opened",
                          signal={}, ts=first)
        self.record_fill("q146", side="BUY", qty=half, price=px, fee=half * px * 0.0005, order_id="O-first",
                         trade_id="T-first", ts=first)
        for at in (SETTLED - pd.Timedelta(hours=1), SETTLED):
            self.record_funding("q146", qty=half if at < SETTLED else qty, price=px, rate=0.1, amount=-750.0,
                                ts=at.to_pydatetime())
        return out

    monkeypatch.setattr(Store, "record_order", journaled)
    monkeypatch.setattr(Store, "create_sleeve", with_first_fill_and_funding)
    now = moved(fixed, qty, -1500.0)
    gap = now * 0.995  # through the price moved by both settlements, above the one moved by 08:00 alone
    assert gap > moved(fixed, qty, -750.0)
    p = qa.shape(flat_prices(20), 4, 20, gap / qa.BASE)
    held = (half, px, int(pd.Timestamp("2025-10-03 07:55", tz="UTC").value))
    run = qa.paper(p, side=1, perp=True, profile="aggressive", stop=None, enter=0, leave=19, held=held)
    liq = [o for o in run.orders if o["intent"] == "liquidation"]
    assert liq, [(o["intent"], o["reason"]) for o in run.orders]
    assert liq[0]["signal"]["liquidation_px"] == pytest.approx(now, rel=1e-6)
