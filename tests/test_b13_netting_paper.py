"""B-13 (Head 6): netting across Strategies in shadow on paper (PM 13 Oct "shadow first"; Advisor specs-p2-3 §2).
Read from the paper journal: what one account per venue would have sent had each decision bar's intents in an
instrument gone out as one net order. It writes nothing and moves no Strategy's figures.

Every expected figure is worked by hand below from the rows written, never read back from the shadow."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

import pytest
from sqlalchemy import func, select

from sleeve_fund.portfolio import netting_paper as npr
from sleeve_fund.store import Store

TAKER = D("0.00055")
T1 = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)  # a decision bar's close
T2 = T1 + timedelta(minutes=1)
INST = "BTC/USDT-PERP"


def _ns(t):
    return int(t.timestamp()) * 1_000_000_000


@pytest.fixture(params=["sqlite", "postgres"])
def store(request):
    """Both engines: SQLite in memory, and Postgres where CI (or a local box) sets TEST_DATABASE_URL."""
    if request.param == "sqlite":
        store = Store.in_memory()
    else:
        import os

        from sleeve_fund.store import make_engine, metadata

        url = os.environ.get("TEST_DATABASE_URL")
        if not url:
            pytest.skip("Postgres: CI runs these against it (TEST_DATABASE_URL)")
        engine = make_engine(url)
        metadata.drop_all(engine)
        store = Store(engine=engine)
    for name in ("a", "b", "c"):
        store.create_sleeve(name=name, strategy="ping_pong", instrument=INST, bar_spec="1-MINUTE-LAST-INTERNAL",
                            starting_balance=0, params={"market": "perp", "allow_short": True})
    yield store
    store.engine.dispose()


def _trade(store, sleeve, oid, side, qty, px, close=None, bar=None, at=None, intent="entry"):
    qty, px = D(qty), D(px)
    fee = (qty * px * TAKER).quantize(D("1e-8"))
    at = at or (bar or T1) + timedelta(seconds=2)
    timing = {"bar_close": _ns(bar), "bar_recv": _ns(bar), "decided": _ns(bar)} if bar is not None else None
    store.record_order(sleeve, order_id=oid, side=side, qty=float(qty), intent=intent, reason="test",
                       signal={"close": float(close)} if close is not None else {}, ts=at, timing=timing)
    store.record_fill(sleeve, side=side, qty=qty, price=px, fee=fee, order_id=oid, trade_id=f"t-{oid}", ts=at)
    return fee


def _run(store, members=lambda n: n in ("a", "b"), equity=D("100000")):
    fills, funding = npr.journal_rows(store, members)
    return npr.shadow(fills, funding, lambda venue, inst: TAKER, equity)


def test_b13a_opposite_intents_on_one_bar_net_to_one_order_saving_the_crossed_spread_and_fee(store):
    fa = _trade(store, "a", "a1", "BUY", "0.3", "60030", close="60000", bar=T1)  # a buys 0.3, half spread 0.05% up
    fb = _trade(store, "b", "b1", "SELL", "0.1", "59970", close="60000", bar=T1)  # b sells 0.1, 0.05% down
    [venue] = (out := _run(store))
    n = out[venue]
    # Alone: -0.3 x 60,030 + 0.1 x 59,970 = -12,012.00 in price; fees 9.904950 + 3.298350 = 13.203300.
    assert fa + fb == D("13.2033")
    # One account: net +0.2 at the mark moved by that bar's half spread, 60,000 x 1.0005 = 60,030; fee 6.6033.
    # Crossed saving: -12,006 - (-12,012) = 6; fee saving 13.2033 - 6.6033 = 6.60.
    assert n["saving"] == {"crossed": D("6"), "fee": D("6.6")}
    assert n["account"] == D("-12012.6033") and n["sub_ledger_total"] == D("-12025.2033")
    assert n["sub_ledgers"] == {"a": D("-18018.90495"), "b": D("5993.70165")}  # each its own step A, untouched
    assert n["residual"]["amount"] == 0 and n["residual"]["block"] is False
    assert (n["orders_sent"], n["orders_alone"], n["bars_shared"]) == (1, 2, 1)
    assert n["crossed_notional"] == D("6000")  # 0.1 matched inside the account at the 60,000 mark


def test_b13b_intents_on_different_bars_or_one_way_save_nothing(store):
    _trade(store, "a", "a1", "BUY", "0.3", "60030", close="60000", bar=T1)
    _trade(store, "b", "b1", "SELL", "0.1", "59970", close="60000", bar=T2)  # the next bar: nothing to cross
    _trade(store, "b", "b2", "BUY", "0.2", "60030", close="60000", bar=T1 + timedelta(minutes=5))
    _trade(store, "a", "a2", "BUY", "0.1", "60030", close="60000", bar=T1 + timedelta(minutes=5))  # same side
    [n] = _run(store).values()
    assert n["saving"] == {} and n["orders_sent"] == 4 - 1  # the same-side bar is one order, at step A's price
    assert n["account"] == n["sub_ledger_total"] and n["residual"]["amount"] == 0


def test_b13c_a_watched_exit_with_no_decision_bar_is_its_own_order(store):
    _trade(store, "a", "a1", "BUY", "0.3", "60030", close="60000", bar=T1)
    # b's target fired on a trade two seconds after T1's close: no decision bar, so it nets with nothing
    _trade(store, "b", "b1", "SELL", "0.1", "59970", at=T1 + timedelta(seconds=2), intent="take_profit")
    [n] = _run(store).values()
    assert n["saving"] == {} and n["orders_sent"] == 2


def test_b13d_fully_crossed_intents_send_nothing(store):
    _trade(store, "a", "a1", "BUY", "0.2", "60030", close="60000", bar=T1)
    _trade(store, "b", "b1", "SELL", "0.2", "59970", close="60000", bar=T1)
    [n] = _run(store).values()
    # Nothing goes out: the account moves 0; alone they paid 0.2 x 60 in spread (12) and 6.6033 + 6.5967 in fees.
    assert n["orders_sent"] == 0 and n["account"] == 0
    assert n["saving"] == {"crossed": D("12"), "fee": D("13.2")}


def test_b13e_only_the_funds_strategies_net_together(store):
    _trade(store, "a", "a1", "BUY", "0.3", "60030", close="60000", bar=T1)
    _trade(store, "c", "c1", "SELL", "0.1", "59970", close="60000", bar=T1)  # c is outside the fund's book here
    [n] = _run(store).values()
    assert n["saving"] == {} and set(n["sub_ledgers"]) == {"a"}


def test_b13f_opposed_exposure_per_pair(store):
    _trade(store, "a", "a1", "BUY", "0.3", "60030", close="60000", bar=T1)
    _trade(store, "b", "b1", "SELL", "0.1", "59970", close="60000", bar=T1)
    [n] = _run(store).values()
    # Just after T1: a +0.3, b -0.1; cancelling 2 x 0.1 of a gross 0.4 = 0.5 (prices cancel out)
    assert n["opposed_exposure"] == {("a", "b"): pytest.approx(0.5)}


def test_b13g_it_writes_nothing(store):
    _trade(store, "a", "a1", "BUY", "0.3", "60030", close="60000", bar=T1)
    _trade(store, "b", "b1", "SELL", "0.1", "59970", close="60000", bar=T1)
    from sleeve_fund.store import metadata

    def counts():
        with store.engine.connect() as c:
            return {t.name: c.execute(select(func.count()).select_from(t)).scalar() for t in metadata.sorted_tables}

    before, books = counts(), {n: store.journal_book(n, 0) for n in ("a", "b")}
    _run(store)
    assert counts() == before and {n: store.journal_book(n, 0) for n in ("a", "b")} == books


def test_b13h_a_taker_mismatch_on_one_account_is_refused(store):
    _trade(store, "a", "a1", "BUY", "0.3", "60030", close="60000", bar=T1)
    fills, funding = npr.journal_rows(store, lambda n: True)
    fills.append({**fills[0], "instrument": "ETH/USDT-PERP", "order_id": "x"})
    with pytest.raises(ValueError, match="same taker fee"):
        npr.shadow(fills, funding, lambda v, i: TAKER if i == INST else D("0.0006"), D("100000"))
