"""Long and short on a perpetual (plan L1-L3, 4 Oct 2026): the margin account, short and flipping trades,
funding, the liquidation price and guard, trade pairing, and a paper restart that carries a short."""

from datetime import datetime, timezone

import pytest

from sleeve_fund import markets
from sleeve_fund.research.metrics import trades
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.store import replay_book
from test_backtest import _path

PERP = {"market": "perp", "allow_short": True}


# --- markets ------------------------------------------------------------------


def test_liquidation_price_long_and_short():
    # A short of 1 at 70,000 cash (35,000 equity + 35,000 proceeds): liquidated where cash + q*p = 0.5% of q*p.
    assert markets.liquidation_price(70_000, -1, 0.005) == pytest.approx(70_000 / 1.005)
    # A long bought with 50,000 borrowed (cash -50,000): liquidated just above the debt.
    assert markets.liquidation_price(-50_000, 1, 0.005) == pytest.approx(50_000 / 0.995)
    # A long fully paid for can't be liquidated; nor can no position.
    assert markets.liquidation_price(1_000, 1, 0.005) is None
    assert markets.liquidation_price(1_000, 0, 0.005) is None


def test_funding_times_are_the_8_hourly_marks_in_the_window():
    t = lambda h, m=0: datetime(2026, 10, 4, h, m, tzinfo=timezone.utc)  # noqa: E731
    assert markets.funding_times(t(7, 59), t(16), (0, 8, 16)) == [t(8), t(16)]
    assert markets.funding_times(t(8), t(15, 59), (0, 8, 16)) == []  # a mark at the start was already settled
    assert len(markets.funding_times(t(0), datetime(2026, 10, 7, tzinfo=timezone.utc), (0, 8, 16))) == 9


def test_markets_fees_and_validation():
    from decimal import Decimal

    from sleeve_fund.instruments import FeeSchedule

    venue = FeeSchedule(Decimal("0.004"), Decimal("0.008"))
    assert markets.fees_for({}, venue) == venue
    assert markets.fees_for({"market": "perp"}, venue).taker == Decimal("0.0005")
    assert markets.fees_for({"market": "perp-venue-fees"}, venue) == venue  # the stress case
    with pytest.raises(ValueError):
        markets.market_of({"market": "options"})


def test_shorts_need_a_perp(prices, instrument):
    with pytest.raises(ValueError, match="perp"):
        run_backtest("ping_pong", prices.iloc[:20], instrument, {"allow_short": True})


# --- the journal and trade pairing ---------------------------------------------


def _fill(side, qty, price, fee=0.0, oid=None):
    return {"side": side, "qty": qty, "price": price, "fee": fee, "order_id": oid or f"{side}{price}", "ts": None}


def test_replay_book_carries_a_short():
    book = replay_book([_fill("SELL", 2, 100.0, 0.1)], 1_000.0, funding=-0.5)
    assert book["qty"] == pytest.approx(-2)
    assert book["entry_px"] == pytest.approx(100.0)
    assert book["cash"] == pytest.approx(1_000 + 200 - 0.1 - 0.5)
    # Covering half keeps the entry; going through flat to long starts a new one at that fill.
    book = replay_book([_fill("SELL", 2, 100.0), _fill("BUY", 1, 90.0), _fill("BUY", 3, 95.0)], 1_000.0)
    assert book["qty"] == pytest.approx(2)
    assert book["entry_px"] == pytest.approx(95.0)


def test_a_short_trip_makes_money_on_a_fall():
    rows = [_fill("SELL", 1, 100.0, 0.05), _fill("BUY", 1, 95.0, 0.05)]
    (t,) = trades(rows, shorts=True)
    assert t["side"] == -1 and t["pnl"] == pytest.approx(4.9) and t["ret"] == pytest.approx(0.049)
    assert t["entry_px"] == 100.0 and t["exit_px"] == 95.0
    # On spot a sell from flat is a journal read from mid-trip, and skipped as before.
    assert trades(rows) == []


def test_one_fill_through_flat_closes_the_trip_and_opens_the_next():
    rows = [_fill("BUY", 1, 100.0, 0.1), _fill("SELL", 3, 110.0, 0.3), _fill("BUY", 2, 105.0, 0.2)]
    long_, short = trades(rows, shorts=True)
    assert (long_["side"], short["side"]) == (1, -1)
    assert long_["pnl"] == pytest.approx(10 - 0.1 - 0.1)  # the sell's fee split by quantity: 0.1 of 0.3
    assert short["pnl"] == pytest.approx(2 * 5 - 0.2 - 0.2)


def test_open_lot_and_allocation_show_a_short():
    from sleeve_fund.dashboard import book, trading

    fills = [_fill("SELL", 1, 100.0, oid="a"), _fill("BUY", 1, 90.0, oid="b"), _fill("SELL", 2, 95.0, oid="c")]
    assert trading.open_lot(list(reversed(fills)), shorts=True)["order_id"] == "c"
    assert trading.open_lot(list(reversed(fills))) is None  # spot reading

    class S:
        instrument = "BTC/USD"

    rows = book.allocation([{"sleeve": S, "position_value": -3_000.0, "cash": 13_000.0}], 10_000.0)
    assert [(r["name"], r["share"]) for r in rows] == [("BTC short", -0.3), ("Cash", 1.3)]
    assert sum(r["bar"] for r in rows) == pytest.approx(1.0) and all(r["bar"] >= 0 for r in rows)


# --- backtests -----------------------------------------------------------------


def test_ping_pong_on_a_perp_shorts_its_short_leg(prices, instrument):
    # Long from 100; +1% at 101.5 sells and shorts; the 0.5% dip at 100.9 covers and goes long; +1% at 102 again.
    closes = [100.0, 100.5, 100.8, 101.5, 101.3, 101.2, 100.9, 101.0, 101.5, 102.0, 102.0]
    res = run_backtest("ping_pong", _path(prices, closes), instrument, PERP, half_spread=0)
    fills = res.fills.sort_values("ts_last")
    assert list(fills["side"]) == ["BUY", "SELL", "SELL", "BUY", "BUY", "SELL", "SELL"]
    intents = [res.decisions[o]["intent"] for o in fills.index]
    assert intents == ["entry", "exit", "entry", "exit", "entry", "exit", "entry"]
    long1, short, long2 = trades([*map(dict, _rows(res))], shorts=True)
    assert [t["side"] for t in (long1, short, long2)] == [1, -1, 1]
    assert short["entry_px"] == pytest.approx(101.5) and short["exit_px"] == pytest.approx(100.9)
    assert short["pnl"] == pytest.approx(short["qty"] * 0.6 - short["fees"])
    # The low-fee perp schedule: 0.05% taker, not Kraken spot's 0.80%.
    first = fills.iloc[0]
    fee = float(str(first["commissions"][0]).split()[0])
    assert fee == pytest.approx(float(first["filled_qty"]) * 100 * 0.0005, abs=0.01)
    # Ends short from 102: equity is the cash flows plus funding, marked at the close.
    assert res.exposure.iloc[-1] < 0


def _rows(res):
    from sleeve_fund.research.metrics import fills_to_rows

    return fills_to_rows(res.fills)


def test_backtest_equity_adds_up_on_a_perp(prices, instrument):
    closes = [100.0, 100.5, 100.8, 101.5, 101.3, 101.2, 100.9, 101.0, 101.5, 102.0, 101.0]
    res = run_backtest("ping_pong", _path(prices, closes), instrument, PERP, half_spread=0)
    rows = _rows(res)
    cash = 10_000.0 - sum((r["qty"] * r["price"] if r["side"] == "BUY" else -r["qty"] * r["price"]) + r["fee"]
                          for r in rows)
    qty = sum(r["qty"] if r["side"] == "BUY" else -r["qty"] for r in rows)
    funding = res.equity.iloc[-1] - (cash + qty * closes[-1])
    # Daily bars cross three funding marks a day; what is left over is funding alone, and small.
    assert funding != 0 and abs(funding) < 0.01 * 10_000


def test_funding_is_charged_to_a_held_long(prices, instrument):
    # Bought on the first close and held flat at 100 for 4 more days: 12 funding marks at 0.01% each.
    closes = [100.0] * 5
    res = run_backtest("ping_pong", _path(prices, closes), instrument, {"market": "perp"}, half_spread=0)
    qty = float(res.fills.iloc[0]["filled_qty"])
    fee = float(str(res.fills.iloc[0]["commissions"][0]).split()[0])
    drag = 10_000 - fee - res.equity.iloc[-1]
    assert drag == pytest.approx(12 * qty * 100 * 0.0001, rel=0.01)


def test_long_only_perp_holds_the_short_leg_flat(prices, instrument):
    closes = [100.0, 100.5, 100.8, 101.5, 101.3, 101.2, 100.9, 101.0]
    res = run_backtest("ping_pong", _path(prices, closes), instrument, {"market": "perp"}, half_spread=0)
    assert list(res.fills.sort_values("ts_last")["side"]) == ["BUY", "SELL", "BUY"]


def test_rsi_bands_shorts_a_rally_on_a_perp(prices, instrument):
    # A slide pushes RSI under 30 (long), a rebound past 55 sells; the rally past 70 now opens a short.
    # (A chop to start, not a flat line: RSI of a flat line reads as overbought.)
    closes = ([100.0 + (0.3 if i % 2 else -0.3) for i in range(20)] + [100 - i for i in range(1, 12)]
              + [89 + 1.5 * i for i in range(1, 15)] + [110.0] * 5)
    res = run_backtest("rsi_bands", _path(prices, closes), instrument, {"rsi_period": 5, **PERP}, half_spread=0)
    fills = res.fills.sort_values("ts_last")
    assert [(f["side"], res.decisions[o]["intent"]) for o, f in fills.iterrows()] == [
        ("BUY", "entry"), ("SELL", "exit"), ("SELL", "entry")]
    assert "short until it falls to 50" in res.decisions[fills.index[2]]["reason"]
    assert res.exposure.iloc[-1] < 0


def test_a_short_stop_rests_above_the_entry(prices, instrument):
    # Short on the rise, then the price rallies 3% through a 2% stop: the stop buys back above the entry.
    closes = [100.0, 100.5, 100.8, 101.5, 102.0, 103.0, 104.6, 104.6]
    res = run_backtest("ping_pong", _path(prices, closes), instrument, {**PERP, "stop_loss": 0.02}, half_spread=0)
    fills = res.fills.sort_values("ts_last")
    intents = [res.decisions[o]["intent"] for o in fills.index]
    assert intents[:3] == ["entry", "exit", "entry"]
    stop = [o for o in fills.index if res.decisions[o]["intent"] == "stop_loss"]
    assert stop, intents
    assert fills.loc[stop[0], "side"] == "BUY"
    assert float(fills.loc[stop[0], "avg_px"]) == pytest.approx(101.5 * 1.02, rel=0.002)


def test_entry_refused_when_its_stop_sits_too_near_liquidation():
    from sleeve_fund.strategies.base import entry_liquidation

    # 3x long of 30,000 on 10,000 equity: liquidated about 33% below; a 20% stop is past half of that.
    liq, distance = entry_liquidation(cash=10_000, qty=0.3, close=100_000, side=1, fee=0.0005, maintenance=0.005)
    assert 0.32 < distance < 0.34
    assert liq == pytest.approx(100_000 * (1 - distance))
    # A 1x short is liquidated only near double the price.
    _, distance = entry_liquidation(cash=10_000, qty=0.1, close=100_000, side=-1, fee=0.0005, maintenance=0.005)
    assert distance > 0.95


# --- paper: the restore order on a restart -------------------------------------


def _record(path, meta, legs, px=60_000.0):
    """A recorded paper session: quotes and trades every second, the price moving by each leg's share
    over its minutes."""
    from nautilus_trader.model import AggressorSide, Price, Quantity, QuoteTick, TradeId, TradeTick

    from sleeve_fund.paper.recorder import Recorder
    from sleeve_fund.venues import venue
    from test_replay import START

    inst = venue("KRAKEN").instrument("BTC", "USD", price_precision=1)
    rec = Recorder(path)
    rec.meta = meta
    rec.start(inst)
    s = 0
    for minutes, move in legs:
        step = (1 + move) ** (1 / (minutes * 60))
        for _ in range(minutes * 60):
            px *= step
            t = START + s * 1_000_000_000
            rec.quote(QuoteTick(inst.id, Price(px - 0.5, 1), Price(px + 0.5, 1), Quantity(1, 8), Quantity(1, 8),
                                t, t + 1000))
            rec.trade(TradeTick(inst.id, Price(px, 1), Quantity(0.05, 8),
                                AggressorSide.BUY if s % 2 else AggressorSide.SELL, TradeId(str(s)), t + 2000, t + 3000))
            s += 1
    rec.close()


def _meta(balance, params):
    return {"balances": [f"{balance:.2f} USD"],
            "sleeve": {"name": "ping-pong-test", "strategy": "ping_pong", "instrument": "BTC/USD",
                       "bar_spec": "1-MINUTE-LAST-INTERNAL", "starting_balance": 10_000,
                       "risk_profile": "balanced", "params": params,
                       "maker_fee": "0.0002", "taker_fee": "0.0005", "tick_seconds": 30}}


def test_ping_pong_shorts_in_the_paper_runtime(tmp_path):
    from sleeve_fund.research.replay import replay

    path = tmp_path / "pp.jsonl.gz"
    _record(path, _meta(10_000, {"rise": 0.01, "dip": 0.005, **PERP}),
            [(5, 0.0), (20, 0.015), (20, -0.012), (20, 0.015)])
    orders, fills = replay(path, with_fills=True)
    assert [(o["side"], o["intent"]) for o in orders][:5] == [
        ("BUY", "entry"), ("SELL", "exit"), ("SELL", "entry"), ("BUY", "exit"), ("BUY", "entry")]
    assert "past the 0.5% dip" in orders[3]["reason"]
    assert all(o["filled_qty"] > 0 for o in orders[:5])


def test_a_restart_carries_a_short_through_a_restore_order(tmp_path):
    """A paper restart holding a short: the margin sandbox can't be given a position, so the strategy sells
    it again there at no fee, unjournaled, and the account then reconciles with the journal."""
    from sleeve_fund.research.replay import replay
    from sleeve_fund.store import Store
    from test_replay import START

    store = Store.in_memory()
    create = store.create_sleeve

    def create_with_a_short(**kw):
        s = create(**kw)
        store.record_fill(kw["name"], side="SELL", qty=0.05, price=61_000.0, fee=1.5, order_id="carried",
                          trade_id="carried", ts=datetime.fromtimestamp(START / 1e9 - 60, tz=timezone.utc))
        return s

    store.create_sleeve = create_with_a_short
    path = tmp_path / "restart.jsonl.gz"
    book = replay_book([_fill("SELL", 0.05, 61_000.0, 1.5)], 10_000.0)
    # The paper node opens a perp's margin account with the cash it had when the position opened.
    _record(path, _meta(book["cash"] + book["qty"] * book["entry_px"], {"rise": 0.01, "dip": 0.005, **PERP}),
            [(10, 0.0)])
    orders, fills = replay(path, with_fills=True, store=store)
    events = store.events("ping-pong-test", limit=500)
    kinds = [e["kind"] for e in events]
    assert "restore_position" in kinds
    assert "reconcile_mismatch" not in kinds, [e["message"] for e in events if e["kind"] == "reconcile_mismatch"]
    assert any(e["kind"] == "reconcile" for e in events)
    # The restore is not a trade: the journal keeps only the carried short until the cycle acts on it.
    assert [f["order_id"] for f in fills][:1] == ["carried"]
    assert all(o["intent"] != "restore" for o in orders)
    # The cycle picked the short up from its entry: price flat at 60,000 is 1.6% below 61,000, past the 0.5% dip.
    assert orders and (orders[0]["side"], orders[0]["intent"]) == ("BUY", "exit")
