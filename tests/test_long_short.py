"""Long and short on a perpetual (plan L1-L3, 4 Oct 2026): the margin account, short and flipping trades,
funding, the liquidation price and guard, trade pairing, and a paper restart that carries a short."""

import re
import shutil
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from sleeve_fund import markets
from sleeve_fund.research.metrics import fills_to_rows, trades
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.store import replay_book
from test_backtest import _path
from test_dashboard import client  # noqa: F401, F811 - the dashboard fixture

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
    # (A chop to start, so RSI begins near 50.)
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


def _record(path, meta, legs, px=60_000.0, size=1.0):
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
        # A leg of 0 minutes is a gap: one trade at the new price, nothing between.
        step = (1 + move) ** (1 / (minutes * 60)) if minutes else 1 + move
        for _ in range(minutes * 60 or 1):
            px *= step
            t = START + s * 1_000_000_000
            rec.quote(QuoteTick(inst.id, Price(px - 0.5, 1), Price(px + 0.5, 1), Quantity(size, 8), Quantity(size, 8),
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


# --- dashboard -------------------------------------------------------------------


def test_the_dashboard_shows_a_short(client):  # noqa: F811
    from test_dashboard import AUTH

    c, store = client
    store.create_sleeve(name="pp-ls", strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params={"rise": 0.01, "dip": 0.005, **PERP, "stop_loss": 0.02})
    store.record_fill("pp-ls", side="BUY", qty=0.05, price=60_000.0, fee=1.5, order_id="o1", trade_id="t1")
    store.record_fill("pp-ls", side="SELL", qty=0.05, price=60_600.0, fee=1.5, order_id="o2", trade_id="t2")
    store.record_fill("pp-ls", side="SELL", qty=0.05, price=60_600.0, fee=1.5, order_id="o3", trade_id="t3")
    store.record_equity("pp-ls", equity=10_050.0, cash=13_057.0, qty=-0.05, price=60_140.0, benchmark=10_000)
    page = c.get("/sleeves/pp-ls", auth=AUTH)
    assert page.status_code == 200
    html = page.text
    assert "Short BTC/USD" in html and "Why it was sold short" in html
    # A perp's 33% cap on balanced is its margin, so at 2x leverage a notional of 66% (PM, 5 Oct 2026).
    assert "of 66%" in html and "over the cap" not in html
    # A losing short grows: at 92,000 against 2,000 of equity it is 230%, past the 66% it was sized to.
    store.record_equity("pp-ls", equity=2_000.0, cash=6_600.0, qty=-0.05, price=92_000.0, benchmark=10_000)
    assert "over the cap" in c.get("/sleeves/pp-ls", auth=AUTH).text
    # The 2% stop sits above the entry; the open short gains as the price falls.
    assert "61,812" in html
    trades_page = c.get("/trades", auth=AUTH).text
    assert "short" in trades_page
    assert c.get("/", auth=AUTH).status_code == 200


def test_default_explain_names_a_short():
    from sleeve_fund.strategies.base import LongFlatStrategy

    assert LongFlatStrategy.explain(None, None, -1)[0] == "Signal to be short"
    assert LongFlatStrategy.explain(None, None, 1)[0] == "Signal to be long"
    assert LongFlatStrategy.explain(None, None, False)[0] == "Signal to be flat"


def test_a_perp_puts_up_the_position_cap_as_margin_at_the_leverage_cap(prices, instrument):
    """PM, 5 Oct 2026: on a perpetual the profile's position cap (33% on balanced) is the margin, and the
    notional is that margin times the leverage cap (2x), so 66% of equity; spot is unchanged at 33%."""
    closes = [100.0, 100.2, 100.1, 100.3]
    perp = run_backtest("ping_pong", _path(prices, closes), instrument, PERP, half_spread=0, risk_profile="balanced")
    spot = run_backtest("ping_pong", _path(prices, closes), instrument, {}, half_spread=0, risk_profile="balanced")
    notional = lambda r: float(r.fills.iloc[0]["filled_qty"]) * float(r.fills.iloc[0]["avg_px"])  # noqa: E731
    assert 0.64 * 10_000 < notional(perp) <= 0.66 * 10_000 + 1
    assert notional(spot) <= 0.33 * 10_000 + 1
    assert perp.decisions[perp.fills.index[0]]["signal"]["sized_by"] == "balanced risk profile cap"


def test_a_perp_position_is_read_exactly_not_from_its_float():
    """Review round 11, B11-2: Nautilus keeps signed_qty as a float sum of the fills, so a position filled in
    two slices of 0.01978348 and 0.08312022 reads 0.10290369999999999. Rounded down to the lot, the exit left
    1e-8 behind and the next entry on the other side rested no stop. The venue's quantity is exact."""
    from types import SimpleNamespace

    from nautilus_trader.model import Quantity

    from sleeve_fund.strategies.base import LongFlatStrategy

    def pos(slices, sign):
        exact = sum(Decimal(x) for x in slices)
        return SimpleNamespace(quantity=Quantity.from_str(str(exact)), signed_qty=sign * sum(float(x) for x in slices))

    slices = ["0.01978348", "0.08312022"]
    assert Decimal(repr(sum(float(x) for x in slices))) < Decimal("0.1029037")  # the float read is short
    for sign in (1, -1):
        fake = SimpleNamespace(_margin=True, _cfg=SimpleNamespace(instrument_id=None),
                               cache=SimpleNamespace(positions_open=lambda instrument_id, ps=[pos(slices, sign)]: ps))
        assert LongFlatStrategy._signed_qty(fake) == sign * Decimal("0.1029037")


def test_every_perp_exit_leaves_the_book_flat_and_every_entry_rests_its_stop(prices, instrument):
    """Review round 11, B11-2: the position was read from Nautilus' float signed_qty (0.08838354999999999
    for 0.08838355) and the exit rounded it down, leaving one lot. The next entry on the other side then
    counted as a reduction: no stop, no target. Every exit must close the whole position, and every entry
    from flat must rest its stop."""
    import numpy as np

    rng = np.random.default_rng(7)
    closes = list(60_000 * np.exp(np.cumsum(rng.normal(0, 0.004, 400))))
    feed = _path(prices, closes)
    feed["volume"] = 1.0  # the venue shows about 0.1 a bar, so each order fills in slices that float sums blur
    params = {"rsi_period": 5, "stop_loss": 0.01, "take_profit": 0.02, **PERP}
    res = run_backtest("rsi_bands", feed, instrument, params, half_spread=0, risk_profile="aggressive")
    intent = {o: d["intent"] for o, d in res.decisions.items()}
    net, opened, closed = Decimal(0), 0, {}
    for f in res.journal.fills_:  # in fill order, slice by slice
        before = net
        net += Decimal(repr(f["qty"])) * (1 if f["side"] == "BUY" else -1)
        if intent[f["order_id"]] == "entry":
            opened += before == 0
        else:
            closed[f["order_id"]] = net
    assert opened >= 6 and closed
    # An exit order, once wholly filled, leaves nothing behind.
    assert all(net_after == 0 for oid, net_after in closed.items()
               if oid != res.journal.fills_[-1]["order_id"]), closed
    # Every entry from flat rested a stop: as many stop orders sent as entries opened from flat (the run may
    # end with one still resting), and no round trip was closed by the next entry.
    stops = sum(1 for d in res.decisions.values() if d["intent"] == "stop_loss")
    assert stops >= opened - 1, (stops, opened)


def _gapped(prices, closes):
    """Bars that each open at their own close: a change between bars is a gap no order can trade inside."""
    feed = _path(prices, closes)
    feed["open"] = feed["high"] = feed["low"] = feed["close"]
    return feed


def test_a_short_gapped_through_its_liquidation_price_is_liquidated_in_backtest(prices, instrument, full_margin):
    """Review round 11, M11-3: the liquidation close never traded (its intent wasn't journaled), so the
    short stayed open past liquidation. Shorted at 101.5, a gap to 160 is through any capped leverage."""
    closes = [100.0, 100.5, 100.8, 101.5, 101.5, 160.0, 160.0, 160.0]
    res = run_backtest("ping_pong", _gapped(prices, closes), instrument, PERP, half_spread=0, risk_profile="aggressive")
    assert not res.handler_errors, res.handler_errors
    fills = res.fills.sort_values("ts_last", kind="stable")
    intents = [res.decisions[o]["intent"] for o in fills.index]
    assert "liquidation" in intents, intents
    liq = fills.index[intents.index("liquidation")]
    assert fills.loc[liq, "side"] == "BUY"
    assert "Liquidated" in res.decisions[liq]["reason"]
    assert res.exposure.iloc[-1] == pytest.approx(0, abs=1e-9)  # closed, and halted: nothing reopened
    assert "liquidation" in {e["kind"] for e in res.risk_events}


def test_a_short_gapped_through_its_liquidation_price_is_liquidated_in_paper(tmp_path, full_margin):
    """The same in paper: a gap past the liquidation price leaves the book under water, which used to read
    as "can't value the book yet" and returned before any guard. It is liquidated and the strategy halts."""
    from sleeve_fund.research.replay import replay

    path = tmp_path / "gap.jsonl.gz"
    _record(path, _meta(10_000, {"rise": 0.01, "dip": 0.005, **PERP}), [(5, 0.0), (20, 0.015), (2, 0.0), (0, 0.7), (3, 0.0)])
    orders, fills = replay(path, with_fills=True)
    kinds = [(o["side"], o["intent"]) for o in orders]
    assert ("SELL", "entry") in kinds and ("BUY", "liquidation") in kinds, kinds
    liq = next(o for o in orders if o["intent"] == "liquidation")
    assert liq["filled_qty"] > 0 and liq["status"] == "filled"
    # Nothing after it: the strategy is halted with the book closed.
    assert orders[-1] is liq


def _ls_run(prices, instrument, seed, params, profile="balanced", n=600, volume=1.0):
    import numpy as np

    rng = np.random.default_rng(seed)
    closes = list(60_000 * np.exp(np.cumsum(rng.normal(0, 0.008, n))))
    feed = _path(prices, closes)
    feed["volume"] = volume  # the venue shows a share of it a bar, so orders fill in slices
    return run_backtest("ping_pong", feed, instrument, {**PERP, **params}, half_spread=0, risk_profile=profile)


def test_every_short_and_every_long_rests_its_stop_over_a_long_run(prices, instrument):
    """Review round 11, B11-2 (5-year case): one exit that left a lot behind put the strategy's own entry
    book out of phase with the venue for good, so every later short was booked as a reduction of a phantom
    long and not one short rested a stop in five years. The book is now read from the venue on every fill:
    each entry from flat, long or short, is followed by its stop on the other side."""
    res = _ls_run(prices, instrument, 2, {"rise": 0.01, "dip": 0.005, "stop_loss": 0.015, "take_profit": 0.015})
    j = res.journal
    orders = sorted(j.orders_.values(), key=lambda o: o["id"])
    filled = {o["order_id"] for o in orders if o["filled_qty"] > 0}
    net, entries, pending = Decimal(0), {1: 0, -1: 0}, None
    stops = {1: 0, -1: 0}
    by_order = {}
    for f in j.fills_:
        by_order.setdefault(f["order_id"], []).append(f)
    for o in orders:
        if o["intent"] == "stop_loss":
            side = 1 if o["side"] == "SELL" else -1  # a sell stop protects a long
            stops[side] += 1
            if pending == side:
                pending = None
        if o["order_id"] not in filled:
            continue
        before = net
        for f in by_order[o["order_id"]]:
            net += Decimal(repr(f["qty"])) * (1 if f["side"] == "BUY" else -1)
        if o["intent"] == "entry" and before == 0 and net != 0:
            assert pending is None, f"the {'long' if pending == 1 else 'short'} before this entry rested no stop"
            pending = 1 if net > 0 else -1
            entries[pending] += 1
    assert entries[1] >= 3 and entries[-1] >= 3, entries
    assert stops[-1] >= entries[-1] - 1 and stops[1] >= entries[1] - 1, (entries, stops)


def test_nothing_opens_after_a_drawdown_halt(prices, instrument, full_margin):
    """Review round 11, B11-3: after a halt, an exit resting for a position the strategy wrongly thought
    it held filled and opened a new short. Exits on a perp are reduce-only, and nothing new rests once the
    strategy is halted: no fill after the halt grows the position."""
    import numpy as np

    # Short at 101.5, a rally through the drawdown limit (the wide stop never fires), then a fall through
    # where the target rested.
    closes = [100.0, 100.5, 100.8, 101.5, *np.linspace(101.5, 125, 20), *np.linspace(125, 70, 30)]
    res = run_backtest("ping_pong", _path(prices, closes), instrument, {**PERP, "stop_loss": 0.3, "take_profit": 0.2},
                       half_spread=0, risk_profile="conservative")
    halts = [e for e in res.risk_events if e["kind"] == "risk_halt"]
    assert halts, [e["kind"] for e in res.risk_events]
    halted_at = halts[0]["ts"]
    net = Decimal(0)
    for f in res.journal.fills_:
        before = net
        net += Decimal(repr(f["qty"])) * (1 if f["side"] == "BUY" else -1)
        if f["ts"] > halted_at:
            assert abs(net) <= abs(before), (f, before, net)
    assert abs(net) < Decimal("1e-8")


def test_paper_shorts_rest_their_stop_after_exits_filled_in_slices(tmp_path):
    """Review round 11, B11-2 in paper: with the top of the book thinner than the order, entries and exits
    fill in slices whose float sum is short of the position; an exit then left a lot behind, the next short
    was booked as a reduction and the paper strategy never stopped it out. Long, then short, then a 5% rally
    through the short's 0.8% stop: the stop buys it back."""
    from sleeve_fund.research.replay import replay

    path = tmp_path / "slices.jsonl.gz"
    _record(path, _meta(10_000, {"rise": 0.01, "dip": 0.005, "stop_loss": 0.008, **PERP}),
            [(5, 0.0), (20, 0.015), (20, 0.05)], size=0.07828204)
    orders, fills = replay(path, with_fills=True)
    kinds = [(o["side"], o["intent"]) for o in orders if o["filled_qty"] > 0]
    assert kinds[:3] == [("BUY", "entry"), ("SELL", "exit"), ("SELL", "entry")], kinds
    assert ("BUY", "stop_loss") in kinds[3:], kinds
    net = sum(Decimal(repr(f["qty"])) * (1 if f["side"] == "BUY" else -1) for f in fills)
    assert abs(net) < Decimal("1e-8")  # the stop closed the whole short


def test_a_long_short_strategy_clones_backtests_and_starts_as_long_short(client, monkeypatch):  # noqa: F811
    """Review round 11, B11-1: "Clone with changes" carried the market and shorts as model parameters the
    forms had no field for, so "Backtest these settings" quietly ran a long-only spot backtest at the
    venue's spot fees and "Start" would have made a long-only spot copy. Both forms now have the fields,
    the backtest says which market and side it ran, and a setting a form can't honour is refused."""
    from urllib.parse import parse_qs, urlencode, urlparse

    from sleeve_fund.dashboard import preview
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.venues import KRAKEN
    from test_dashboard import AUTH, SAME

    c, store = client
    store.create_sleeve(name="pp-ls", strategy="ping_pong", instrument="ETH/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=10_000,
                        params={"rise": 0.01, "dip": 0.005, **PERP, "demo_mirror": True})
    page = c.get("/sleeves/pp-ls", auth=AUTH).text
    href = next(p for p in page.split('"') if p.startswith("/sleeves/new?")).replace("&amp;", "&")
    q = {k: v[0] for k, v in parse_qs(urlparse(href).query).items()}
    assert (q["market"], q["allow_short"], q["demo_mirror"]) == ("perp", "1", "1")
    assert not any(k.endswith(("__market", "__allow_short", "__demo_mirror")) for k in q)
    form = c.get(href, auth=AUTH).text
    assert '<option value="perp" selected>' in form and 'name="allow_short" value="1" checked' in form
    assert 'name="demo_mirror" value="1" checked' in form

    # Backtest these settings: the form's fields, as the page's link sends them (no name, reason or mirror).
    preview._history.clear()
    monkeypatch.setattr(KRAKEN, "daily_history", lambda pair: synthetic_ohlcv(days=400, seed=3, vol=0.012))
    bt = {k: v for k, v in q.items() if k not in ("name", "from", "source", "demo_mirror")}
    bt["risk_profile"] = "conservative"  # 1x: calm enough that the daily-loss pause doesn't end it early
    result = c.get("/backtest?" + urlencode({**bt, "run": "1"}), auth=AUTH).text
    assert "Couldn't run it" not in result, result[result.find("Couldn't"):][:300]
    assert "Low-fee perpetual (simulated), long and short" in result
    assert "0.02% maker, 0.05% taker" in result
    run = store.backtests(limit=1)[0]
    saved = store.sleeve(run["sleeve"])
    assert saved.params.get("market") == "perp" and saved.params.get("allow_short") is True
    fills = store.fills(saved.name, limit=100_000)
    net, shorted = 0.0, False
    for f in sorted(fills, key=lambda f: f["id"]):
        shorted = shorted or (net <= 1e-12 and f["side"] == "SELL")
        net += f["qty"] if f["side"] == "BUY" else -f["qty"]
    assert shorted, "the backtest never went short"
    # M12-U5: the side rides on the entry cell, which no width hides (Size, which says it too, goes at 1440 px).
    table = result.split('id="bt-tr-h"')[1].split("</table>")[0]
    # Not hidden on a phone's card either (data-m="hide"), where Size is gone too.
    sides = re.findall(r'<tr><td data-label="Entry">[^<]+<div class="sub">(long|short)</div></td>', table)
    assert "short" in sides and len(sides) == table.count('<tr class="detail"'), sides

    # Start: the new strategy keeps the market, shorts and mirror.
    data = {k: v for k, v in q.items() if k not in ("from", "source")}
    assert c.post("/sleeves/new", data={**data, "reason": "long/short copy"}, auth=AUTH, headers=SAME,
                  follow_redirects=False).status_code == 303
    made = store.sleeve(q["name"]).params
    assert (made["market"], made["allow_short"], made["demo_mirror"]) == ("perp", True, True)

    # An old clone link, with the market as a model parameter, is refused rather than run as spot.
    old = {**bt, "p_ping_pong__market": "perp", "run": "1"}
    old.pop("market")
    page = c.get("/backtest?" + urlencode(old), auth=AUTH).text
    assert "Couldn't run it" in page and "market" in page
    # And shorts on spot are refused, not dropped.
    page = c.get("/backtest?" + urlencode({**bt, "market": "spot", "run": "1"}), auth=AUTH).text
    assert "only on a perpetual" in page


def test_header_trade_stats_and_g2_count_shorts_as_the_trades_tab_does():
    """Review round 11, M11-4: the header's closed trades, win rate and profit factor (and the G2 checklist's
    trade count) paired a perpetual's fills as spot, ignoring shorts: "1 trade" against 3 on the Trades tab."""
    from sleeve_fund.dashboard import trading
    from sleeve_fund.dashboard.metrics import sleeve_summary
    from sleeve_fund.store import Store

    store = Store.in_memory()
    store.create_sleeve(name="pp-ls", strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params={**PERP})
    legs = [("BUY", 0.05, 60_000), ("SELL", 0.05, 60_600),  # a long
            ("SELL", 0.05, 60_600), ("BUY", 0.05, 60_000),  # a short
            ("BUY", 0.05, 60_000), ("SELL", 0.10, 59_000),  # a long, then one sell through flat into a short
            ("BUY", 0.05, 58_000)]
    for n, (side, qty, px) in enumerate(legs):
        store.record_fill("pp-ls", side=side, qty=qty, price=px, fee=1.0, order_id=f"o{n}", trade_id=f"t{n}")
    s = store.sleeve("pp-ls")
    stats = sleeve_summary(store, s)["trades"]
    tab = trading.trips(store.fills("pp-ls"), [], {}, shorts=True)
    assert stats["trades"] == len(tab) == 4
    assert stats["wins"] == sum(1 for t in tab if t["pnl"] > 0) == 3


def test_the_risk_page_stresses_a_short_book_both_ways(client):  # noqa: F811
    """Review round 11, M11-6: two shorts gain in a fall and lose in a rally; gross counts them, net offsets."""
    from test_dashboard import AUTH

    c, store = client
    for name, qty, px in (("short-a", -0.05, 60_000.0), ("short-b", -1.0, 3_000.0), ("long-c", 0.02, 60_000.0)):
        store.create_sleeve(name=name, strategy="ping_pong", instrument="ETH/USD" if px == 3_000 else "BTC/USD",
                            bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=10_000,
                            params={"rise": 0.01, "dip": 0.005, **PERP, "stop_loss": 0.02})
        store.record_equity(name, equity=10_000.0, cash=10_000.0 - qty * px, qty=qty, price=px, benchmark=10_000)
    # Positions: -3,000, -3,000 and +1,200. Gross 7,200 (24% of 30,000); net -4,800 (-16%).
    page = c.get("/risk", auth=AUTH).text
    assert "7,200.00 of 30,000.00" in page and "net −16% short" in page.replace("-16%", "−16%")
    # Down 20%: the shorts make 1,200, the long loses 240, so the book makes 960; up 20% it loses 960.
    assert "+960.00" in page and "−960.00" in page
    assert "+960" in page.split("Market down 20%")[1].split("</div></div>")[0]


def test_a_perp_shows_leverage_liquidation_and_funding(client):  # noqa: F811
    """Review round 11, M11-5: the Position tab and header show what a perp adds, and every trade's P&L
    is after fees and funding, in the Trades tab, the header and the export alike."""
    from datetime import timedelta

    from test_dashboard import AUTH

    from sleeve_fund.store import utcnow

    c, store = client
    store.create_sleeve(name="pp-fund", strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params={"rise": 0.01, "dip": 0.005, **PERP, "stop_loss": 0.02})
    t0 = utcnow() - timedelta(days=1)
    fill = lambda side, px, h, oid: store.record_fill("pp-fund", side=side, qty=0.1, price=px, fee=1.0,  # noqa: E731
                                                      order_id=oid, trade_id=oid, ts=t0 + timedelta(hours=h))
    fill("SELL", 60_000.0, 0, "o1")  # short 0.1 at 60,000
    store.record_funding("pp-fund", qty=-0.1, price=60_000.0, rate=0.0001, amount=0.6, ts=t0 + timedelta(hours=8))
    fill("BUY", 59_000.0, 10, "o2")  # closes: +100 - 2 fees + 0.6 funding = 98.60
    fill("SELL", 59_000.0, 11, "o3")  # a new short, still open
    store.record_funding("pp-fund", qty=-0.1, price=59_000.0, rate=0.0001, amount=0.59, ts=t0 + timedelta(hours=16))
    # cash = 10,000 + 6,000 - 5,900 + 5,900 - 3 fees + 1.19 funding; equity at 59,500 = cash - 5,950
    cash = 10_000 + 6_000 - 5_900 + 5_900 - 3 + 1.19
    store.record_equity("pp-fund", equity=cash - 5_950, cash=cash, qty=-0.1, price=59_500.0, benchmark=10_000)
    html = c.get("/sleeves/pp-fund", auth=AUTH).text
    liq = markets.liquidation_price(cash, -0.1, markets.LOW_FEE_PERP.maintenance_margin)
    assert "Margin" in html and "Short 0.59×" in html and f"{liq:,.0f}"[:5] in html and "above" in html
    assert "+0.59" in html and "+1.19" in html  # last payment and since start
    assert "funding +0.60" in html and "+98.60" in html  # the closed trip, after fees and funding
    assert "· 0.59× · liq" in html  # the header
    csv = c.get("/exports/trades.csv?sleeve=pp-fund", auth=AUTH)
    assert csv.status_code == 200 and "funding" in csv.text.splitlines()[0] and "98.6" in csv.text


# --- review round 11, M11-7: exits in the position's own terms -----------------


def test_a_short_swing_stop_rests_at_the_highest_high(prices, instrument):
    closes = [100.0, 100.5, 100.8, 101.5, 102.0, 103.0, 104.6, 104.6]
    feed = _path(prices, closes)
    res = run_backtest("ping_pong", feed, instrument, {**PERP, "stop_swing_bars": 3}, half_spread=0)
    shorts = [d for d in res.decisions.values() if d["intent"] == "entry" and d["signal"].get("side") == "short"]
    stops = [d for d in res.decisions.values() if d["intent"] == "stop_loss" and "resting buy" in d["reason"]]
    assert shorts and stops
    entry_px, level = stops[0]["signal"]["entry_px"], stops[0]["signal"]["trigger"]
    assert level > entry_px and "above the" in stops[0]["reason"] and "highest high" in stops[0]["reason"]
    assert "lowest low" not in stops[0]["reason"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_exit_plan_line_prices_a_short_as_the_strategy_does():
    import json
    import re
    import subprocess
    from pathlib import Path

    from sleeve_fund.strategies.base import gain_at_target, loss_at_stop, r_target

    js = (Path(__file__).parent.parent / "sleeve_fund" / "dashboard" / "static" / "console.js").read_text()
    math = re.search(r"^  const exitMath = \{.*?^  \};$", js, re.S | re.M).group(0)
    cases = [(stop, tp, leg, side) for stop in (0.01, 0.03) for tp in (0.02, 0.1) for leg in (0.0006, 0.0085)
             for side in (1, -1)]
    script = math + f"\nconsole.log(JSON.stringify({json.dumps(cases)}.map(([s, t, l, d]) => "\
        "[exitMath.loss(s, l, d), exitMath.gain(t, l, d), exitMath.rTarget(2, s, l, d)])));"
    got = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout)
    for (stop, tp, leg, side), (loss, gain, rt) in zip(cases, got):
        assert loss == pytest.approx(loss_at_stop(stop, leg, side), abs=1e-15)
        assert gain == pytest.approx(gain_at_target(tp, leg, side), abs=1e-15)
        assert rt == pytest.approx(r_target(2, stop, leg, side), abs=1e-15)


def test_an_open_shorts_exits_form_and_decision_log_say_above(client):  # noqa: F811
    from test_dashboard import AUTH, SAME

    c, store = client
    store.create_sleeve(name="pp-x", strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params={"rise": 0.01, "dip": 0.005, **PERP, "stop_loss": 0.02})
    flat = c.get("/sleeves/pp-x", auth=AUTH).text
    assert "A % below (a short: above) the entry" in flat  # flat and long/short: both ways
    store.record_fill("pp-x", side="SELL", qty=0.05, price=60_000.0, fee=1.5, order_id="o1", trade_id="t1")
    store.record_equity("pp-x", equity=10_000.0, cash=12_998.5, qty=-0.05, price=60_000.0, benchmark=10_000)
    html = c.get("/sleeves/pp-x", auth=AUTH).text
    assert 'data-side="-1"' in html and "A % above the entry" in html and "Target, % below entry" in html
    assert "61,200" in html  # the 2% stop rests above the entry
    form = {"risk_profile": "balanced", "stop_loss_pct": "1.5", "reason": "tighter"}
    r = c.post("/sleeves/pp-x/settings", data=form, auth=AUTH, headers=SAME, follow_redirects=False)
    assert "saved=settings" in r.headers["location"], r.headers["location"]
    log = store.decisions("pp-x", limit=5)[0]["reason"]
    assert "Stop-loss 2% above the entry (short) to 1.5% above the entry (short)" in log


def test_a_gap_past_bankruptcy_loses_the_margin_and_no_more(prices, instrument):
    """Sanity 5 Oct: a gap far past the liquidation price booked a loss beyond the strategy's equity. Under
    isolated margin the venue's insurance fund takes the rest: equity ends at zero, not below, and the trade's
    P&L (after fees, funding and the insurance fund) is the margin lost, as equity says."""
    closes = [100.0, 100.5, 100.8, 101.5, 101.5, 400.0, 400.0, 400.0]
    res = run_backtest("ping_pong", _gapped(prices, closes), instrument, PERP, half_spread=0, risk_profile="aggressive")
    assert res.insurance and res.insurance[0]["amount"] > 0
    assert res.equity.min() >= 0 and res.equity.iloc[-1] < 0.05
    trips = trades(fills_to_rows(res.fills), True, res.funding, res.insurance)
    assert trips[-1]["insurance"] == pytest.approx(res.insurance[0]["amount"])
    assert sum(t["pnl"] for t in trips) == pytest.approx(res.equity.iloc[-1] - res.starting_capital, abs=0.05)
    assert "insurance_fund" in {e["kind"] for e in res.journal.events_}
    assert res.journal.journal_book("backtest", res.starting_capital)["cash"] == pytest.approx(0, abs=0.05)


def test_a_risk_exit_in_a_backtest_fills_where_it_was_judged_not_at_the_close(prices, instrument, full_margin):
    """Coordinator, 4 Oct: a wick through the daily-loss level that recovers inside the bar paused the
    strategy, but its buy-back filled at the bar's close, kinder than paper by the wick. A reduce-only stop
    now rests at the level the guard acts at, so the exit fills there, as paper's does on the breaching trade."""
    closes = [100.0, 100.0, 100.0, 100.0]
    feed = _path(prices, closes)
    feed.iloc[2, feed.columns.get_loc("low")] = 90.0  # a wick 10% down, back to 100 by the close
    res = run_backtest("ping_pong", feed, instrument, {**PERP, "rise": 0.4, "dip": 0.4}, half_spread=0,
                       risk_profile="balanced")
    orders = sorted(res.journal.orders_.values(), key=lambda o: o["id"])
    assert [o["intent"] for o in orders][:2] == ["entry", "risk_pause"], orders
    exit_ = orders[1]
    assert "Risk stop" in exit_["reason"] and exit_["order_type"] == "STOP"
    # Balanced: 2x, a 5% daily loss, so the level is about 2.5% under the entry, far above the 90 low.
    assert 97.0 < exit_["avg_px"] < 98.0, exit_
    buy, sell = res.journal.fills_[:2]
    loss = buy["qty"] * (buy["price"] - sell["price"]) + buy["fee"] + sell["fee"]
    # The 5% daily loss, plus the exit's fee and a day's funding: not the 20% the wick reached.
    assert res.starting_capital * 0.05 <= loss <= res.starting_capital * 0.053


# --- review round 11, M11-9: tests for the mutations that survived the suite ------------------------


def test_a_shorts_1r_and_r_target_carry_its_side():
    """A short buys back above the entry at its stop, so the exit leg costs more than a long's, and its
    target sits below the entry, so the target's exit leg costs less. "2R" pays two of that 1R after costs."""
    from sleeve_fund.strategies.base import gain_at_target, loss_at_stop, r_target

    stop, leg = 0.02, 0.001
    assert loss_at_stop(stop, leg, 1) == pytest.approx(stop + leg + 0.98 * leg)
    assert loss_at_stop(stop, leg, -1) == pytest.approx(stop + leg + 1.02 * leg)
    for side in (1, -1):
        tp = r_target(2, stop, leg, side)
        assert gain_at_target(tp, leg, side) == pytest.approx(2 * loss_at_stop(stop, leg, side))
    assert r_target(2, stop, leg, -1) != pytest.approx(r_target(2, stop, leg, 1), abs=1e-9)


def test_an_entry_whose_stop_sits_past_half_way_to_liquidation_is_refused(prices, instrument, full_margin):
    """Aggressive sizes a perp at 3x, liquidated about 33% away: a 25% stop is past half of that, so the
    entry is refused and says why; a 10% stop is inside it and enters."""
    closes = [100.0, 100.5, 100.8, 101.5, 101.5, 101.0, 100.0]
    feed = _path(prices, closes)
    res = run_backtest("ping_pong", feed, instrument, {**PERP, "stop_loss": 0.25}, half_spread=0,
                       risk_profile="aggressive")
    assert res.fills.empty
    assert "entry_refused_liquidation" in {e["kind"] for e in res.journal.events_}
    res = run_backtest("ping_pong", feed, instrument, {**PERP, "stop_loss": 0.10}, half_spread=0,
                       risk_profile="aggressive")
    assert not res.fills.empty
    assert "entry_refused_liquidation" not in {e["kind"] for e in res.journal.events_}


def test_a_short_take_profit_rests_below_the_entry(prices, instrument):
    # Short at 101.5, then a fall: the 2% target buys back at 101.5 x 0.98, not above the entry.
    closes = [100.0, 100.5, 100.8, 101.5, 101.0, 100.0, 99.0, 98.0, 98.0]
    res = run_backtest("ping_pong", _path(prices, closes), instrument,
                       {**PERP, "take_profit": 0.02, "dip": 0.05}, half_spread=0)
    fills = res.fills.sort_values("ts_last")
    got = [(res.decisions[o]["intent"], fills.loc[o, "side"]) for o in fills.index]
    assert got == [("entry", "BUY"), ("exit", "SELL"), ("entry", "SELL"), ("take_profit", "BUY")], got
    assert float(fills.loc[fills.index[3], "avg_px"]) == pytest.approx(101.5 * 0.98, rel=1e-4)


def test_a_strategy_wiped_out_by_a_gap_is_marked_at_zero_and_halted_through_a_restart(tmp_path, full_margin):
    """Review round 12, B12-1: a paper short gapped through its bankruptcy price ended flat at zero equity, which
    read as a book that couldn't be valued yet: no mark, no risk check, no halt, even after a restart, so the
    dashboard kept its last mark before the gap (running, in profit, still short) and the alerts showed raw
    account text. Wiped out is a state: marked at zero, halted with the reason, through a resume and a restart."""
    from sleeve_fund.research.replay import replay
    from sleeve_fund.store import Store

    store, name, params = Store.in_memory(), "ping-pong-test", {"rise": 0.01, "dip": 0.005, **PERP}
    gap = tmp_path / "gap.jsonl.gz"
    _record(gap, _meta(10_000, params), [(5, 0.0), (20, 0.015), (0, 0.6), (5, 0.0)])  # short, then +60%
    orders = replay(gap, store=store)
    assert orders[-1]["intent"] == "liquidation" and orders[-1]["side"] == "BUY"
    covered = store.insurance_total(name)
    assert covered > 0
    s = store.sleeve(name)
    assert s.status == "halted" and s.status_reason.startswith("wiped out: a gap took the price past"), s.status_reason
    assert "insurance fund covers the shortfall, about" in s.status_reason  # halted while open: an estimate (mF-1)
    last = store.equity_series(name)[-1]
    assert (last["equity"], last["qty"]) == (0.0, 0.0)  # the book counts it at zero, not its last mark
    book = store.journal_book(name, 10_000)
    assert book["qty"] == 0 and book["cash"] == pytest.approx(0, abs=0.01)
    assert "mark_unavailable" not in {e["kind"] for e in store.events(name, limit=500)}

    # The PM resumes it and the process restarts: still nothing to trade, so it halts again, at zero.
    store.command(name, "resume", "try again")
    store.create_sleeve = lambda **kw: store.sleeve(kw["name"])
    seen = len(store.equity_series(name))
    restart = tmp_path / "restart.jsonl.gz"
    _record(restart, _meta(book["cash"], params), [(5, 0.0)], px=97_440.0)
    assert len(replay(restart, store=store)) == len(orders)  # no new order
    s = store.sleeve(name)
    assert s.status == "halted", (s.status, s.status_reason)
    assert f"the venue's insurance fund covered the {covered:,.2f} shortfall" in s.status_reason, s.status_reason
    marks = store.equity_series(name)[seen:]
    assert marks and all((m["equity"], m["qty"]) == (0.0, 0.0) for m in marks)
    events = store.events(name, limit=500)
    assert "mark_unavailable" not in {e["kind"] for e in events}
    assert [e["kind"] for e in events].count("risk_halt") == 2


def test_a_restart_holding_a_perp_settles_the_funding_it_was_down_for(tmp_path):
    """Funding owed while the process was down is settled on the first tick: from the last fill or funding
    payment, not from the restart. A short opened at 15:00 and restarted at midnight owes 16:00 and 00:00."""
    from sleeve_fund.research.replay import replay
    from sleeve_fund.store import Store
    from test_replay import START

    store = Store.in_memory()
    create = store.create_sleeve
    opened = datetime.fromtimestamp(START / 1e9 - 9 * 3600, tz=timezone.utc)

    def create_with_a_short(**kw):
        s = create(**kw)
        store.record_fill(kw["name"], side="SELL", qty=0.05, price=61_000.0, fee=1.5, order_id="carried",
                          trade_id="carried", ts=opened)
        return s

    store.create_sleeve = create_with_a_short
    path = tmp_path / "funding.jsonl.gz"
    book = replay_book([_fill("SELL", 0.05, 61_000.0, 1.5)], 10_000.0)
    _record(path, _meta(book["cash"] + book["qty"] * book["entry_px"], {"rise": 0.01, "dip": 0.005, **PERP}),
            [(10, 0.0)])
    replay(path, with_fills=True, store=store)
    paid = sorted(f["ts"].replace(tzinfo=timezone.utc) for f in store.funding("ping-pong-test", limit=10))
    assert paid == [datetime(2025, 10, 2, 16, tzinfo=timezone.utc), datetime(2025, 10, 3, tzinfo=timezone.utc)]


def test_paper_reconciles_the_position_to_two_lots(tmp_path, monkeypatch):
    """The reconcile tolerance on the position is two lots of the base currency (review rounds 9 and 10,
    B9-1, B10-1): wider would let a real gap pass, narrower halts on rounding."""
    from sleeve_fund.paper.runtime import SleeveRuntime
    from sleeve_fund.research.replay import replay

    seen = []
    real = SleeveRuntime.reconcile

    def spy(self, **kw):
        seen.append(kw["qty_tolerance"])
        return real(self, **kw)

    monkeypatch.setattr(SleeveRuntime, "reconcile", spy)
    path = tmp_path / "tol.jsonl.gz"
    _record(path, _meta(10_000, {"rise": 0.01, "dip": 0.005, **PERP}), [(5, 0.0)])
    replay(path)
    assert seen and all(t == pytest.approx(2e-8) for t in seen)  # BTC trades in lots of 0.00000001


def test_without_a_resting_risk_stop_the_guard_still_judges_the_wick(prices, instrument, monkeypatch):
    """The risk stop can't rest while an exit or a flip is in flight. The guard is the backstop then: it
    judges each bar at its worst price for the position (the low for a long), so a wick through the
    daily-loss level that recovers by the close still pauses the strategy (long/short verdict, L3)."""
    from sleeve_fund.strategies.base import LongFlatStrategy

    monkeypatch.setattr(LongFlatStrategy, "_rest_risk_stop", lambda self: None)
    closes = [100.0, 100.0, 100.0, 100.0]
    feed = _path(prices, closes)
    feed.iloc[2, feed.columns.get_loc("low")] = 90.0  # a wick 10% down, back to 100 by the close
    res = run_backtest("ping_pong", feed, instrument, {**PERP, "rise": 0.4, "dip": 0.4}, half_spread=0,
                       risk_profile="balanced")
    assert "risk_pause" in {e["kind"] for e in res.risk_events}, res.risk_events


@pytest.mark.parametrize("volume", [1e6, 5.0])
def test_a_flip_opens_the_new_side_only_once_the_old_one_is_closed(prices, instrument, volume):
    """Turning from long to short (or back) closes the position first and opens the new side only once
    it is flat: no entry fill ever adds to the other side's position or crosses zero (review round 11). On
    a thin book (volume 5) the orders fill in slices across bars. It keeps trading, too: the risk stop's
    re-pricing each bar once read as an order in flight, and the strategy never decided again after its
    first entry."""
    import numpy as np

    feed = _path(prices, list(60_000 * (1 + 0.02 * np.sin(np.arange(120) / 4))))  # a cycle every 25 bars
    feed["volume"] = volume
    res = run_backtest("ping_pong", feed, instrument, {**PERP, "rise": 0.01, "dip": 0.005}, half_spread=0,
                       risk_profile="balanced")
    intent = {o: d["intent"] for o, d in res.decisions.items()}
    net, entries, sides = Decimal(0), set(), set()
    for f in res.journal.fills_:
        before = net
        step = Decimal(repr(f["qty"])) * (1 if f["side"] == "BUY" else -1)
        net += step
        if intent[f["order_id"]] == "entry":
            entries.add(f["order_id"])
            sides.add(f["side"])
            # From flat, or adding to its own side: never against an open position of the other side.
            assert before == 0 or (before > 0) == (step > 0), (f, before)
    assert len(entries) >= 4 and sides == {"BUY", "SELL"}, (len(entries), res.risk_events)


def test_a_perp_backtests_benchmark_is_an_unlevered_hold_at_the_perps_fee(monkeypatch):
    """Round 12, M12-U1: on a perp the benchmark held at the position cap past 1x and never liquidated, so it
    ran from -200% to +400%. It is at most a 1x hold, paying the perp's taker fee, not spot's. Aggressive's
    cap passes 1x: 50% of equity as margin at 3x is a notional of 150%."""
    from sleeve_fund.dashboard import preview
    from sleeve_fund.data import synthetic_ohlcv
    from test_dashboard import KRAKEN

    bars = synthetic_ohlcv(days=200, seed=2, start_price=150)
    preview._history.clear()
    monkeypatch.setattr(KRAKEN, "daily_history", lambda pair: bars)
    d = preview.run("buy_and_hold", "SOL/USD", {**PERP}, starting=5000, risk_profile="aggressive")
    taker = float(markets.LOW_FEE_PERP.fees.taker)
    assert d["cap"] == pytest.approx(1.5) and d["bench_cap"] == 1.0
    assert d["fee_schedule"]["taker"] == pytest.approx(taker)
    assert d["benchmark"][0] == pytest.approx(5000 * (1 - taker), abs=0.01)
    held = 5000 * (1 - taker) * bars["close"].iloc[-1] / bars["close"].iloc[0]
    assert d["benchmark"][-1] == pytest.approx(held, rel=1e-6)


@pytest.mark.sanity
@pytest.mark.parametrize(("params", "label"), [({**PERP}, "66% invested"), ({}, "33% invested")], ids=["perp", "spot"])
def test_a_saved_runs_screen_shows_the_benchmark_its_result_shows(client, monkeypatch, params, label):  # noqa: F811
    """Round 12, M12-U1 (re-check): the result held a perp's benchmark at 1x while the paper runtime, which
    writes the saved run's journal and every paper strategy's, held it at the profile's 33% spot cap, so the
    result and the strategy screen of the same run disagreed. One definition now (the position cap, never
    above 1x), labelled with its exposure on both."""
    from sleeve_fund.dashboard import preview
    from sleeve_fund.data import synthetic_ohlcv
    from test_dashboard import AUTH, KRAKEN

    c, store = client
    bars = synthetic_ohlcv(days=200, seed=2, start_price=150)
    preview._history.clear()
    monkeypatch.setattr(KRAKEN, "daily_history", lambda pair: bars)
    keep = {}
    d = preview.run("buy_and_hold", "SOL/USD", params, starting=5000, risk_profile="balanced", keep=keep)
    name = store.save_backtest(keep["journal"], run_id="u1", key="k", title="Benchmark run", query="", result=d)
    rows = store.equity_series(name)
    assert rows[-1]["benchmark"] == pytest.approx(d["benchmark"][-1], abs=0.01)
    screen = c.get(f"/sleeves/{name}", auth=AUTH).text
    bench_ret = d["benchmark"][-1] / 5000 - 1
    assert f"buy and hold, {label}: {bench_ret * 100:+.1f}%" in screen, screen.split("Since start")[1].split("</button>")[1][:300]


@pytest.mark.sanity
def test_the_kill_switch_dialog_counts_shorts_by_their_size(client):  # noqa: F811
    """Round 12, M12-U4: the kill switch's confirm dialog summed only long positions, so a book of shorts read
    as about a seventh of the notional the switch would trade. Every open position counts, by its size."""
    from test_dashboard import AUTH

    c, store = client
    for name, qty, px in (("short-a", -0.3, 60_000.0), ("long-b", 0.05, 60_000.0)):
        store.create_sleeve(name=name, strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                            starting_balance=10_000, params={"rise": 0.01, "dip": 0.005, **PERP})
        store.set_desired_state(name, "running")
        store.record_equity(name, equity=10_000.0, cash=10_000.0 - qty * px, qty=qty, price=px, benchmark=10_000)
    dialog = c.get("/risk", auth=AUTH).text.split('id="dlg-kill"')[1].split("</dialog>")[0]
    # 18,000 short and 3,000 long: 21,000 to trade, not the 3,000 of longs alone (nor the -15,000 net).
    assert "worth about 21,000" in dialog, dialog
    assert "close to cash at market (longs sell, shorts buy back)" in dialog
    assert "Close everything and pause" in dialog  # not "Sell everything" with a short to buy back


@pytest.mark.sanity
def test_the_book_and_strategy_drawdowns_count_a_loss_from_the_starting_capital(client):  # noqa: F811
    """Round 12, M12-F1: the running peak started at the first daily close, never at the starting capital, so a
    strategy wiped out on its first day left the book reading -20% since start beside a 0.0% drawdown, and a
    second wipe-out read 25% worst (from that close), not 40%."""
    from test_dashboard import AUTH

    c, store = client
    day1, day2 = datetime(2026, 10, 4, 12, tzinfo=timezone.utc), datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
    for n in range(5):
        store.create_sleeve(name=f"s{n}", strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                            starting_balance=10_000, params={"rise": 0.01, "dip": 0.005, **PERP})
        first = 0.0 if n == 0 else 10_000.0
        store.record_equity(f"s{n}", equity=first, cash=first, qty=0.0, price=60_000.0, benchmark=10_000, ts=day1)
        last = 0.0 if n <= 1 else 10_000.0  # a second strategy is wiped out on day two
        store.record_equity(f"s{n}", equity=last, cash=last, qty=0.0, price=60_000.0, benchmark=10_000, ts=day2)
    book = c.get("/api/book/equity", auth=AUTH).json()
    assert book["start"] == 50_000 and book["equity"] == [40_000, 30_000]
    assert book["drawdown"] == [pytest.approx(0.2), pytest.approx(0.4)]  # from 50,000, not the 40,000 close
    tile = c.get("/", auth=AUTH).text.split("Drawdown")[1].split("</div></div>")[0]
    assert "40.0%" in tile and "worst 40.0%" in tile, tile
    # One strategy whose first mark is already a loss: its chart and figures count it from its starting balance.
    store.create_sleeve(name="late", strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params={"rise": 0.01, "dip": 0.005, **PERP})
    store.record_equity("late", equity=9_000.0, cash=9_000.0, qty=0.0, price=60_000.0, benchmark=10_000, ts=day1)
    one = c.get("/api/sleeves/late/equity", auth=AUTH).json()
    assert one["drawdown"] == [pytest.approx(0.1)] and one["worst"] == pytest.approx(0.1)
    assert store.max_drawdown("late") == 0.0 and store.max_drawdown("late", 10_000) == pytest.approx(0.1)


def test_a_wiped_out_strategy_says_it_cannot_trade_and_is_no_stress_breach(client):  # noqa: F811
    """Round 12 fix re-check, mF-2 and mF-3: Resume on a wiped-out strategy promised it would trade again on its
    next signal, and /risk listed it as a strategy that would halt at every market move."""
    from test_dashboard import AUTH

    c, store = client
    store.create_sleeve(name="gone", strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params={"rise": 0.01, "dip": 0.005, **PERP})
    store.record_equity("gone", equity=0.0, cash=0.0, qty=0.0, price=60_000.0, benchmark=10_000)
    store.set_status("gone", "halted", "wiped out: a gap took the price past the bankruptcy price, so equity is zero")
    page = c.get("/sleeves/gone", auth=AUTH).text
    assert "cannot open a trade: resuming only halts it again" in page and "trades again on its next signal" not in page
    stress = c.get("/risk", auth=AUTH).text.split("Strategies that would halt")[1].split("</table>")[0]
    assert "gone" not in stress, stress


def test_the_largest_asset_counts_shorts_by_gross_exposure(client):  # noqa: F811
    """Round 12, M12-U2: the tile took the largest signed share, so a big short (a negative share) read as
    nothing and a small long elsewhere was shown as the book's concentration."""
    from sleeve_fund.dashboard.riskops import largest_asset
    from test_dashboard import AUTH

    # A short-only book: its one short is the concentration, by its size.
    only = largest_asset([{"name": "BTC short", "value": -20_825.0}, {"name": "Cash", "value": 60_000.0}], 39_175.0)
    assert only["name"] == "BTC" and only["share"] == pytest.approx(20_825 / 39_175)
    assert only["net_share"] == pytest.approx(-20_825 / 39_175)
    c, store = client
    for name, qty, px in (("short-a", -0.05, 60_000.0), ("short-b", -1.0, 3_000.0), ("long-c", 0.02, 60_000.0)):
        store.create_sleeve(name=name, strategy="ping_pong", instrument="ETH/USD" if px == 3_000 else "BTC/USD",
                            bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=10_000,
                            params={"rise": 0.01, "dip": 0.005, **PERP})
        store.record_equity(name, equity=10_000.0, cash=10_000.0 - qty * px, qty=qty, price=px, benchmark=10_000)
    # BTC: a 3,000 short and a 1,200 long, 4,200 gross (14% of 30,000), -1,800 net (-6%); ETH: a 3,000 short.
    tile = c.get("/risk", auth=AUTH).text.split("Largest asset")[1].split("</div></div>")[0]
    assert "BTC 14%" in tile and "net" in tile and "6%" in tile, tile
