"""B-36, the Advisor's ruling B36-STRESS-CAP: on a fund-held store an isolated perp's stress loss from the mark is
capped at its remaining margin, the posted margin plus its SIGNED unrealised P&L (a winner can lose its gain too),
plus the taker fee at the liquidation price the engine works out (margin.isolated_liquidation on cash + the fund's
funding). Never 0 while a position is open, and the fund's total is the sum over its positions. Expected values are
worked here independently of the dashboard. On Postgres too (TEST_DATABASE_URL)."""

import pytest
from test_fe_b20_fund_held_wording import _engine_liquidation, _hold, _summaries
from test_halt_ref import (
    store,  # noqa: F401 - the journal, on Postgres when TEST_DATABASE_URL is set
)

from sleeve_fund import markets
from sleeve_fund.venues import venue as venue_profile

PERP = {"market": "perp", "allow_short": True}
QTY, ENTRY, FEE = 0.1, 60_000.0, 3.3


def _long(store, name, mark, qty=QTY):  # noqa: F811
    """A fund-held 2x perp long (made with 0, a 2% stop so it may use 2x; the stress ignores stops) of `qty` at 60,000 marked at `mark`, journaled as the engine does."""
    store.create_sleeve(name=name, strategy="ping_pong", instrument="BTC/USDT", venue="binance",
                        bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=0, desired_state="running",
                        params={"rise": 0.01, "dip": 0.005, "stop_loss": 0.02, **PERP})
    store.record_order(name, order_id=f"{name}-1", side="BUY", qty=qty, intent="entry", reason="dipped 0.5%")
    store.update_order(f"{name}-1", fill_qty=qty, fill_px=ENTRY, fee=FEE)
    store.record_fill(name, side="BUY", qty=qty, price=ENTRY, fee=FEE, order_id=f"{name}-1", trade_id=f"{name}-T")
    cash = -qty * ENTRY - FEE
    store.record_equity(name, equity=cash + qty * mark, cash=cash, qty=qty, price=mark, benchmark=0)
    return store.sleeve(name)


def _view(store):  # noqa: F811
    from sleeve_fund.dashboard import book as bookm
    from sleeve_fund.dashboard import riskops

    summaries, frames = _summaries(store)
    return riskops.risk_view(store, summaries, bookm.book_view(store, summaries, frames)), summaries


def _expected(store, s, mark):  # noqa: F811
    liq, posted = _engine_liquidation(store, s)
    taker = float(markets.fees_for(s.params, venue_profile(s.venue).fees, s.venue).taker)
    return posted + QTY * (mark - ENTRY) + taker * QTY * liq, posted, liq


def _down50(view):
    return next(sc for sc in view["scenarios"] if sc["shock"] == -0.50)


def test_a_winner_can_lose_its_gain_too(store):  # noqa: F811
    """(a) 2x long at +10%: what it has at stake is its margin, its gain and the fee. A 50% fall from 66,000 ends at
    33,000, above its liquidation, so that row books the move itself."""
    from sleeve_fund.dashboard.riskops import most_it_can_lose

    _hold(store)
    mark = ENTRY * 1.10
    s = _long(store, "win", mark)
    cap, posted, liq = _expected(store, s, mark)
    view, summaries = _view(store)
    assert posted == pytest.approx(QTY * ENTRY / 2)  # 2x
    assert most_it_can_lose(summaries[0], 1e12) == pytest.approx(cap)
    assert cap > posted + QTY * (mark - ENTRY)  # the signed gain is in it, with the fee on top
    assert mark * 0.5 > liq
    assert _down50(view)["loss"] == pytest.approx(QTY * mark * 0.5)


def test_a_loser_loses_its_remaining_margin_and_the_fee_never_0(store):  # noqa: F811
    """(b) 2x long at -20%, then a 50% fall past its liquidation: the remaining margin (3,000 - 1,200) and the fee."""
    _hold(store)
    mark = ENTRY * 0.80
    s = _long(store, "lose", mark)
    cap, posted, liq = _expected(store, s, mark)
    assert mark * 0.5 < liq
    view, _ = _view(store)
    loss = _down50(view)["loss"]
    assert loss == pytest.approx(cap)
    assert loss > posted - QTY * ENTRY * 0.20 > 0  # the remaining 1,800 and the fee, never 0


def test_the_funds_total_is_the_sum_over_its_positions(store):  # noqa: F811
    """(c) Two Strategies long the same asset: each position counted once, the fund's row their sum."""
    _hold(store)
    mark = ENTRY * 0.80
    a = _long(store, "a", mark)
    b = _long(store, "b", mark)
    view, _ = _view(store)
    caps = [_expected(store, s, mark)[0] for s in (a, b)]
    rows = {r["x"]["sleeve"].name: r for r in view["rows"]}
    assert sorted(rows) == ["a", "b"]
    for name, cap in zip(("a", "b"), caps, strict=True):
        assert rows[name]["shocks"][0]["loss"] == pytest.approx(cap)
    assert _down50(view)["loss"] == pytest.approx(sum(caps))
