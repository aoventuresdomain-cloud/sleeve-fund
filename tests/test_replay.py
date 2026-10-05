"""A recorded paper session, replayed through the backtest engine, sends the same orders (plan M3)."""

import json
import math
from pathlib import Path

import pytest

from sleeve_fund.paper.recorder import Recorder
from sleeve_fund.research.replay import comparable, replay

# Live paper sessions and the orders paper sent. The maker sessions' orders were re-recorded when paper began
# filling post-only orders in slices, as backtests do (review round 9, M9-3): the venue used to fill them whole.
RECORDINGS = Path(__file__).parent / "data" / "replay"
START = 1_759_449_600_000_000_000  # 2025-10-03 00:00 UTC, in nanoseconds


def _synthetic(path, minutes=90, maker_wait=None, bar_spec="1-MINUTE-LAST-INTERNAL", silent=(0, 0)):
    from nautilus_trader.model import AggressorSide, Price, Quantity, QuoteTick, TradeId, TradeTick

    from sleeve_fund.venues import venue

    inst = venue("KRAKEN").instrument("BTC", "USD", price_precision=1)
    rec = Recorder(path)
    params = {"fast": 2, "slow": 4, **({"maker_wait_minutes": maker_wait} if maker_wait else {})}
    rec.meta = {"balances": ["10000.00 USD"],
                "sleeve": {"name": "replay-test", "strategy": "trend_filter", "instrument": "BTC/USD",
                           "bar_spec": bar_spec, "starting_balance": 10_000, "risk_profile": "aggressive",
                           "params": params, "max_notional": 1000, "maker_fee": "0.004", "taker_fee": "0.008",
                           "tick_seconds": 30}}
    rec.start(inst)
    for s in range(minutes * 60):
        if silent[0] * 60 <= s < silent[1] * 60:
            continue  # the feed says nothing
        px = 60_000 + 300 * math.sin(s / 600) + 20 * math.sin(s / 7)
        t = START + s * 1_000_000_000
        rec.quote(QuoteTick(inst.id, Price(px - 0.5, 1), Price(px + 0.5, 1), Quantity(1, 8), Quantity(1, 8),
                            t, t + 1000))
        rec.trade(TradeTick(inst.id, Price(px, 1), Quantity(0.01, 8),
                            AggressorSide.BUY if s % 2 else AggressorSide.SELL, TradeId(str(s)), t + 2000, t + 3000))
    rec.close()
    return path


def test_a_replay_trades_with_the_paper_runtime_and_is_deterministic(tmp_path):
    path = _synthetic(tmp_path / "s.jsonl.gz")
    a, b = replay(path), replay(path)
    assert len(a) >= 4 and comparable(a) == comparable(b)
    assert {o["intent"] for o in a} == {"entry", "exit"}
    buys = [o for o in a if o["side"] == "BUY"]
    # Market orders fill on the ask, half a dollar above the trade price: the spread is paid.
    assert all(o["avg_px"] is not None for o in a)
    assert all(o["filled_qty"] * o["avg_px"] <= 1000 * 1.001 for o in buys)  # max_notional holds


@pytest.mark.usefixtures("maker_on")
def test_a_maker_order_joins_the_best_bid(tmp_path):
    path = _synthetic(tmp_path / "m.jsonl.gz", minutes=120, maker_wait=2, bar_spec="5-MINUTE-LAST-INTERNAL")
    orders = replay(path)
    makers = [o for o in orders if o["order_type"] == "POST-ONLY LIMIT"]
    assert makers
    for o in makers:
        limit = o["signal"]["limit_px"]
        price = o["signal"]["price"]
        assert (limit < price) if o["side"] == "BUY" else (limit > price)  # resting on its own side of the book


@pytest.mark.usefixtures("maker_on")
@pytest.mark.parametrize("recording", sorted(RECORDINGS.glob("*.jsonl.gz")), ids=lambda p: p.name)
def test_recorded_paper_session_replays_to_the_same_orders(recording):
    from datetime import datetime

    sent = json.loads(recording.with_name(recording.name.replace(".jsonl.gz", ".orders.json")).read_text())
    for o in sent:
        o["ts"] = datetime.fromisoformat(o["ts"])
    assert sent, "the recording should include at least one order to be worth replaying"
    assert comparable(replay(recording)) == comparable(sent)


def test_a_silent_feed_is_flagged_then_restarted(tmp_path):
    """Review rounds 1 to 5: no stale-price watchdog. The tick timer keeps the heartbeat going by
    itself, so a feed gone quiet looked healthy while marks and the risk guard used a frozen price."""
    from sleeve_fund.store import Store

    path = _synthetic(tmp_path / "gap.jsonl.gz", minutes=40, silent=(10, 30))  # twenty silent minutes
    store = Store.in_memory()
    replay(path, store=store)
    events = list(reversed(store.events("replay-test", limit=500)))
    kinds = [e["kind"] for e in events if e["kind"] in ("stale_price", "feed_dead", "price_feed_back")]
    assert kinds == ["stale_price", "feed_dead", "price_feed_back"]  # each said once
    warn, dead = (next(e for e in events if e["kind"] == k) for k in ("stale_price", "feed_dead"))
    assert dead["level"] == "error" and "restarts it" in dead["message"]
    assert (dead["ts"] - warn["ts"]).total_seconds() == pytest.approx(600, abs=60)
    back = next(e for e in events if e["kind"] == "price_feed_back")
    marks = [m["ts"] for m in store.equity_series("replay-test", limit=100_000)]
    assert not [t for t in marks if dead["ts"] < t < back["ts"]]  # no marks, no heartbeat: the supervisor restarts it


def test_paper_keeps_when_the_venue_last_sent_a_trade_or_quote(tmp_path):
    """PM, 5 Oct 2026: the price feed's age on the strategy page. The strategy notes the time of the venue's
    latest trade or quote, every few seconds at most, and a silent feed leaves it where the feed stopped."""
    from datetime import datetime, timedelta, timezone

    from sleeve_fund.store import Store

    path = _synthetic(tmp_path / "feed.jsonl.gz", minutes=12, silent=(10, 12))  # silent for the last two minutes
    store = Store.in_memory()
    replay(path, store=store)
    seen = store.last_feed("replay-test")
    stopped = datetime.fromtimestamp(START / 1e9, tz=timezone.utc) + timedelta(minutes=10)
    assert seen is not None and stopped - timedelta(seconds=4) <= seen <= stopped


@pytest.mark.usefixtures("maker_on")
def test_fills_at_8_lot_decimals_are_journaled_as_the_account_holds_them(tmp_path):
    """A venue lists XRP at 8 lot decimals while the account keeps 6. Post-only entries fill in slices sized
    by 8-decimal prints; the account rounds each slice to 6, so the journal must too, or the two drift apart
    until reconcile halts the strategy (review round 10, B10-1)."""
    from decimal import Decimal

    import numpy as np
    from nautilus_trader.model import (AggressorSide, Currency, CurrencyPair, InstrumentId, Price, Quantity,
                                       QuoteTick, Symbol, TradeId, TradeTick)

    import sleeve_fund.paper.runtime as rtmod
    from sleeve_fund.store import Store
    from sleeve_fund.venues import venue

    v = venue("KRAKEN").instrument("XRP", "USD", price_precision=5)
    inst = CurrencyPair(instrument_id=InstrumentId.from_str("XRP/USD.KRAKEN"), raw_symbol=Symbol("XRP/USD"),
                        base_currency=Currency.from_str("XRP"), quote_currency=Currency.from_str("USD"),
                        price_precision=5, size_precision=8, price_increment=Price(1e-5, 5),
                        size_increment=Quantity(1e-8, 8), lot_size=None, min_quantity=None, min_notional=None,
                        margin_init=Decimal(0), margin_maint=Decimal(0), maker_fee=v.maker_fee, taker_fee=v.taker_fee,
                        ts_event=0, ts_init=0)
    rng, rng2 = np.random.default_rng(7), np.random.default_rng(11)
    s = np.arange(4 * 3600)
    p = 0.5 * (1 + 0.02 * np.sin(s / 900) + 0.006 * np.sin(s / 300)) * (1 + 0.0004 * rng.standard_normal(len(s)).cumsum() / 30)
    path = tmp_path / "xrp8.jsonl.gz"
    rec = Recorder(path)
    rec.meta = {"balances": ["10000.00 USD"],
                "sleeve": {"name": "x", "strategy": "trend_filter", "instrument": "XRP/USD",
                           "bar_spec": "15-MINUTE-LAST-INTERNAL", "starting_balance": 10_000, "risk_profile": "aggressive",
                           "params": {"fast": 3, "slow": 8, "stop_loss": 0.02, "maker_wait_minutes": 10},
                           "max_notional": None, "maker_fee": "0.004", "taker_fee": "0.008", "tick_seconds": 30}}
    rec.start(inst)
    for j, px in enumerate(np.round(p, 5)):
        t = START + j * 1_000_000_000
        size = round(150.00617284 * (0.2 + 1.6 * rng2.random()), 8)
        rec.trade(TradeTick(inst.id, Price(px, 5), Quantity(size, 8), AggressorSide.BUY if j % 2 else AggressorSide.SELL,
                            TradeId(str(j)), t, t + 1000))
        rec.quote(QuoteTick(inst.id, Price(px - 3e-5, 5), Price(px + 3e-5, 5), Quantity(10 ** 6, 8),
                            Quantity(10 ** 6, 8), t + 2000, t + 3000))
    rec.close()
    store = Store(f"sqlite:///{tmp_path}/j.db")
    old = rtmod.RECONCILE_EVERY
    rtmod.RECONCILE_EVERY = old / (24 * 12)  # every 5 minutes
    try:
        replay(path, store=store)
    finally:
        rtmod.RECONCILE_EVERY = old
    fills = store.fills("x", limit=10_000)
    assert len(fills) > len({f["order_id"] for f in fills})  # some orders filled in slices
    assert all(Decimal(repr(f["qty"])) == Decimal(repr(f["qty"])).quantize(Decimal("1e-6")) for f in fills)
    kinds = [e["kind"] for e in store.events("x", limit=10_000)]
    assert "reconcile" in kinds and "reconcile_mismatch" not in kinds
