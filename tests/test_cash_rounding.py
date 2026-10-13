"""CASH-3 (Advisor 8 Oct 04:10 UK ruling 1; 13 Oct 16:15 UK): each fill's fee and cash leg count at 8 dp, rounded half
up, on their own, with no carry, wherever the quote currency holds 8 places (USDT: the paper venue and the golden
fixtures), so a restart can't move a fee. A currency held to fewer places (USD, test Strategies only) keeps the
carry, so its fees stay within a cent of the schedule in total."""

from decimal import Decimal as D

import pytest
from nautilus_trader.model import Currency

from sleeve_fund.instruments import FeeSchedule, ScheduleFeeModel
from sleeve_fund.store import fill_cash, per_fill, replay_book

USDT, USD = Currency.from_str("USDT"), Currency.from_str("USD")
EXACT = [D("0.155"), D("0.1549"), D("0.000000125"), D("0.000000135"), D("0.0000000149"), D("0.1651")]


def _model():
    return ScheduleFeeModel(FeeSchedule(D("0.0002"), D("0.0005")))


def test_a_usdt_fee_is_the_same_with_or_without_a_restart_before_it():
    one_run = _model()
    run = [one_run._charge(x, USDT).as_decimal() for x in EXACT]
    restarted = [_model()._charge(x, USDT).as_decimal() for x in EXACT]
    assert run == restarted == [D("0.155"), D("0.1549"), D("0.00000013"), D("0.00000014"), D("0.00000001"),
                                D("0.1651")]


def test_a_usd_fee_keeps_the_carry_so_small_fills_still_pay_within_a_cent_in_total():
    m = _model()
    charged = [m._charge(D("0.004"), USD).as_decimal() for _ in range(10)]
    assert sum(charged) == D("0.04")  # no fill pays nothing in the end: rounding alone would charge 0.00 ten times


@pytest.mark.parametrize("side, sign", [("BUY", -1), ("SELL", 1)])
def test_a_fills_cash_leg_and_fee_count_at_8_dp_half_up_each_on_its_own(side, sign):
    # leg 0.333 x 0.123456785 = 0.041111109405 -> 0.04111111; fee 0.000000125 -> 0.00000013 (half up, not even)
    assert fill_cash(side, D("0.333"), D("0.123456785"), D("0.000000125")) == sign * D("0.04111111") - D("0.00000013")
    assert per_fill(D("0.000000135")) == D("0.00000014") and per_fill(D("-0.000000125")) == D("-0.00000013")


def test_the_book_replays_the_same_rounding_as_the_one_reader():
    fills = [dict(side="BUY", qty=D("0.333"), price=D("0.123456785"), fee=D("0.000000125")),
             dict(side="SELL", qty=D("0.333"), price=D("0.123456785"), fee=D("0.000000135"))]
    book = replay_book(fills, 0)
    assert book["cash"] == sum((fill_cash(f["side"], f["qty"], f["price"], f["fee"]) for f in fills), D(0))
    assert book["cash"] == -(D("0.00000013") + D("0.00000014"))


def test_the_store_reads_each_fill_at_8_dp_half_up_on_either_database(tmp_path):
    """Postgres sums with round(numeric, 8) (half away from zero), SQLite row by row with per_fill: the same figure."""
    import os
    from datetime import datetime, timezone

    from sqlalchemy import text

    from sleeve_fund.store import Store, make_engine, metadata

    eng = make_engine(os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path}/t.db")
    with eng.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS alembic_version"))
    metadata.drop_all(eng)
    try:
        s = Store(engine=eng)
        s.create_sleeve(name="s", strategy="buy_and_hold", instrument="BTC/USDT-PERP",
                        bar_spec="1-HOUR-LAST-INTERNAL", starting_balance=1_000)
        t = datetime(2026, 10, 13, 9, tzinfo=timezone.utc)
        s.record_fill("s", side="BUY", qty=D("0.333"), price=D("0.123456785"), fee=D("0.000000125"),
                      order_id="O-1", trade_id="T-1", ts=t)
        s.record_fill("s", side="SELL", qty=D("0.333"), price=D("0.123456785"), fee=D("0.000000135"),
                      order_id="O-2", trade_id="T-2", ts=t)
        assert s.strategy_cash_pnl("s") == -(D("0.00000013") + D("0.00000014"))
        assert s.fill_totals("s") == (2, D("0.00000027"))
        assert D(str(s.journal_book("s", 1_000)["cash"])) - 1_000 == s.strategy_cash_pnl("s")
    finally:
        metadata.drop_all(eng)
