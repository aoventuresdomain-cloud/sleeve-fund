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


def test_a_maker_order_joins_the_best_bid(tmp_path):
    path = _synthetic(tmp_path / "m.jsonl.gz", minutes=120, maker_wait=2, bar_spec="5-MINUTE-LAST-INTERNAL")
    orders = replay(path)
    makers = [o for o in orders if o["order_type"] == "POST-ONLY LIMIT"]
    assert makers
    for o in makers:
        limit = o["signal"]["limit_px"]
        price = o["signal"]["price"]
        assert (limit < price) if o["side"] == "BUY" else (limit > price)  # resting on its own side of the book


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
