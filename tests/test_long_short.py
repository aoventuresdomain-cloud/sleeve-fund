"""Long and short on a perpetual (plan L1-L3, 4 Oct 2026): the margin account, short and flipping trades,
funding, the liquidation price and guard, trade pairing, and a paper restart that carries a short."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from sleeve_fund import markets
from sleeve_fund.research.metrics import trades
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
    # A perp is capped by balanced's 2x leverage, not its 33% spot position cap (PM, 4 Oct 2026).
    assert "of 200%" in html and "over the cap" not in html
    # A losing short grows: at 92,000 against 2,000 of equity it is 230%, past the 2x it was sized to.
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


def test_a_perp_is_sized_by_the_leverage_cap(prices, instrument):
    """PM, 4 Oct 2026: a perpetual's position is sized by the profile's leverage cap (2x on balanced),
    not its spot position cap (33%); spot is unchanged."""
    closes = [100.0, 100.2, 100.1, 100.3]
    perp = run_backtest("ping_pong", _path(prices, closes), instrument, PERP, half_spread=0, risk_profile="balanced")
    spot = run_backtest("ping_pong", _path(prices, closes), instrument, {}, half_spread=0, risk_profile="balanced")
    notional = lambda r: float(r.fills.iloc[0]["filled_qty"]) * float(r.fills.iloc[0]["avg_px"])  # noqa: E731
    assert 1.9 * 10_000 < notional(perp) <= 2 * 10_000
    assert notional(spot) <= 0.33 * 10_000 + 1
    assert perp.decisions[perp.fills.index[0]]["signal"]["sized_by"] in ("2x leverage cap", "balanced risk profile cap")


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


def test_a_short_gapped_through_its_liquidation_price_is_liquidated_in_backtest(prices, instrument):
    """Review round 11, M11-3: the liquidation close never traded (its intent wasn't journaled), so the
    short stayed open past liquidation. Shorted at 101.5, a gap to 160 is through any capped leverage."""
    closes = [100.0, 100.5, 100.8, 101.5, 101.5, 160.0, 160.0, 160.0]
    res = run_backtest("ping_pong", _path(prices, closes), instrument, PERP, half_spread=0, risk_profile="aggressive")
    assert not res.handler_errors, res.handler_errors
    fills = res.fills.sort_values("ts_last", kind="stable")
    intents = [res.decisions[o]["intent"] for o in fills.index]
    assert "liquidation" in intents, intents
    liq = fills.index[intents.index("liquidation")]
    assert fills.loc[liq, "side"] == "BUY"
    assert "Liquidated" in res.decisions[liq]["reason"]
    assert res.exposure.iloc[-1] == pytest.approx(0, abs=1e-9)  # closed, and halted: nothing reopened
    assert "liquidation" in {e["kind"] for e in res.risk_events}


def test_a_short_gapped_through_its_liquidation_price_is_liquidated_in_paper(tmp_path):
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


def test_nothing_opens_after_a_drawdown_halt(prices, instrument):
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
