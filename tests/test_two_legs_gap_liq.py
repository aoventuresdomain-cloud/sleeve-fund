"""F283-P1 (Advisor 14 Oct 15:00 UK, a MUST before any leveraged pair runs on paper): GAP-LIQ on every exit path. An
exit fill at or beyond a leg's liquidation price is a LIQUIDATION, whatever closed it (a stop, a halt's flatten, the
pair's unwind, a PM close): loss exactly X at bankruptcy (GAP-LIQ-CAP), the insurance diagnostic, the incident, X/Y and
a reset after liquidation. On every mark, in this order: (1) each leg's own liquidation check, at the engine's own
liquidation price; (2) the pair's consequences (the other leg's unwind); (3) the summed halts, which still raise but
never replace the liquidation.

The case: a long pair (spot and a perp at 2x, liquidation near 30,300) gaps from 60,000 to 29,000 in one bar. Today the
summed halt on the first leg's mark closes the perp leg before its own check runs, as a plain risk close."""

import os
import re
from decimal import Decimal as D
from types import SimpleNamespace

import pytest

os.environ.setdefault("BACKTEST_ISOLATE", "0")

from sleeve_fund.research.two_leg_run import LegRun, run_two_legs  # noqa: E402
from sleeve_fund.strategies import REGISTRY  # noqa: E402
from tests.test_two_legs_engine import INST, PERP, SPOT, Window, WindowConfig, _orders, _rising  # noqa: E402

LONG_PERP = {**PERP, "side": 1, "leverage": 2}
SPEC = SimpleNamespace(default_params={})  # the paper cell's test-only model reads its defaults from its module
X_Y = re.compile(r"Position margin lost \(liquidated\): [^\d]*(\d[\d,]*(?:\.\d+)?)")


@pytest.fixture(autouse=True)
def _window(monkeypatch):
    monkeypatch.setitem(REGISTRY, "window", (Window, WindowConfig))


def _gap(factor, profile="balanced"):
    bars = _rising(daily=0.0, jump=(6, factor))
    params = {"windows": [[int(bars.index[1].value), int(bars.index[-1].value)]]}
    return run_two_legs("window", LegRun(SPOT, INST, bars), LegRun(LONG_PERP, INST), 10_000, params=params,
                        risk_profile=profile)


def _kinds(r, leg):
    return [e["kind"] for e in getattr(r, leg).journal.events_]


def test_iii_a_gap_past_the_perp_legs_liquidation_price_is_booked_as_a_liquidation_on_the_backtest():
    r = _gap(29 / 60)
    follow, lead = _orders(r, "follow"), _orders(r, "lead")
    closes = [o for o in follow if o["intent"] != "entry"]
    assert closes and closes[0]["intent"] == "liquidation", [(o["intent"], o.get("avg_px")) for o in follow]
    assert "liquidation" in _kinds(r, "follow") and "incident" in _kinds(r, "follow")
    # The pair's consequence: the first leg is closed with it, in the same trade.
    unwound = [o for o in lead if o["intent"] != "entry"]
    assert unwound and unwound[0]["parent_trade_id"] == closes[0]["parent_trade_id"]
    # The summed halt is still raised (23% down under balanced's 20%), beside the liquidation, not instead.
    assert "risk_halt" in _kinds(r, "lead") or "risk_halt" in _kinds(r, "follow")
    # Loss exactly X (GAP-LIQ-CAP): the halt's X is what the perp leg's cash lost.
    halt = next(e["message"] for e in r.follow.journal.events_ if e["kind"] == "risk_halt"
                and X_Y.search(e["message"]))
    x = float(X_Y.search(halt).group(1).replace(",", ""))
    start, end = float(r.follow.equity.iloc[0]), float(r.follow.equity.iloc[-1])
    assert start - end == pytest.approx(x, abs=0.02), (start, end, x)


def test_ii_a_gap_that_stops_just_inside_the_liquidation_price_is_a_normal_close():
    """52% of the entry (31,200) is inside the perp leg's liquidation price: the summed halt or the cut before
    liquidation closes both legs as a normal close, and nothing is booked as a liquidation."""
    r = _gap(0.52)
    follow = _orders(r, "follow")
    closes = [o["intent"] for o in follow if o["intent"] != "entry"]
    assert closes and "liquidation" not in closes, [(o["intent"], o.get("avg_px")) for o in follow]
    assert "liquidation" not in _kinds(r, "follow") and "incident" not in _kinds(r, "follow")


def test_i_on_paper_the_gap_books_the_perp_legs_liquidation_unwinds_the_spot_leg_and_still_halts(tmp_path, monkeypatch):
    """The real two-leg node on a local hub: a long pair (spot, and a perp at 2x) entered at 60,000, then the price
    prints 29,000. The perp leg's liquidation is booked as one (loss X, the insurance line, the incident, a reset after
    liquidation needed); the spot leg is unwound with it; the summed drawdown halt is raised too."""
    import threading
    import time
    from datetime import datetime, timezone

    from nautilus_trader.model import OrderSide

    from sleeve_fund.hub.server import Fanout
    from sleeve_fund.paper import node as node_mod
    from sleeve_fund.paper.config import from_store
    from sleeve_fund.paper.queued import QueuedStore
    from sleeve_fund.paper.runtime import SleeveRuntime, liquidation_head
    from sleeve_fund.store import Store
    from sleeve_fund.strategies.base import LongFlatConfig, LongFlatStrategy
    from sleeve_fund.venues import venue
    from tests.test_p21b_wiring_qa import portfolio_pass

    sent, started = [], []

    class Driven(LongFlatStrategy):
        """Never signals on its own; sends one entry through the pair on a quote once the node has settled."""

        def want_long(self, bar):
            return None

        def explain(self, bar, target):
            return "driven", {}

        def on_quote(self, quote):
            super().on_quote(quote)
            started.append(time.monotonic()) if not started else None
            if not sent and self._price() > 0 and time.monotonic() - started[0] > 4:
                sent.append(1)
                self._send_entry(OrderSide.BUY, D("0.06"), "entry", {"close": self._price()})

    monkeypatch.setitem(REGISTRY, "driven", (Driven, LongFlatConfig))
    monkeypatch.setattr(LongFlatStrategy, "_entry_atr_pct", lambda self: 0.05)
    inst = venue("kraken").instrument("BTC", "USDT")
    insts = {str(inst.id): inst}
    hub = Fanout("KRAKEN", known=lambda: set(insts), log=lambda *a: None,
                 instruments=lambda ids: [type(insts[i]).to_dict(insts[i]) for i in ids if i in insts])
    port = hub.start("127.0.0.1", 0)
    store = Store(f"sqlite:///{tmp_path}/t.db", portfolio_gate=True)
    store.create_two_legs(name="carry", strategy="driven", bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=10_000,
                          first={"instrument": "BTC/USDT", "params": {}, "venue": "KRAKEN", "side": 1},
                          second={"instrument": "BTC/USDT", "params": {"market": "perp"}, "venue": "KRAKEN",
                                  "side": 1, "leverage": 2})
    # A book big enough that its open-risk limit (5%) takes the whole pair, sized so the gap is a summed drawdown past
    # balanced's 20% as well as the perp leg's liquidation.
    portfolio_pass(store, datetime.now(timezone.utc), D(100_000))
    journal = QueuedStore(store)
    lead, follow = (from_store(store.sleeve(n)) for n in ("carry", "carry~2"))
    node, pair = node_mod.build_two_leg_node(lead, follow, log_level="ERROR", asset_fetch=dict,
                                             runtimes=(SleeveRuntime(journal, "carry"),
                                                       SleeveRuntime(journal, "carry~2")),
                                             hub=("127.0.0.1", port), history=lambda *a, **k: [])

    def feed():
        time.sleep(2)
        for k in range(60):
            px = "60000.00" if k < 40 else "29000.00"  # held for ~4 s after the entry, then the gap
            ts = time.time_ns()
            hub.publish({"t": "quote", "id": "BTC/USDT.KRAKEN", "bid": px, "ask": px.replace(".00", ".10"),
                         "bid_qty": "5.00000000", "ask_qty": "5.00000000", "ts": ts, "recv": ts})
            hub.publish({"t": "trade", "id": "BTC/USDT.KRAKEN", "px": px, "qty": "0.50000000", "side": "SELLER",
                         "tid": str(k), "ts": ts, "recv": ts})
            time.sleep(0.2)

    threading.Thread(target=feed, daemon=True).start()
    handle = node.handle()
    threading.Timer(16, handle.stop).start()
    try:
        node.run()
    finally:
        journal.settle(timeout=10)
        hub.stop()
    try:
        perp = sorted(store.orders("carry~2"), key=lambda o: o["id"])  # newest first as read
        assert [o["intent"] for o in perp][:1] == ["entry"] and perp[0]["status"] == "filled", perp
        closes = [o for o in perp if o["intent"] != "entry"]
        assert closes and closes[0]["intent"] == "liquidation", [(o["intent"], o["avg_px"]) for o in perp]
        kinds = [e["kind"] for e in store.events("carry~2", limit=500)]
        assert "liquidation" in kinds and "incident" in kinds
        assert liquidation_head(store, "carry~2") is not None  # a reset after liquidation is needed
        assert "insurance_fund" in kinds  # 29,000 is past the bankruptcy price (~30,000): the diagnostic line
        spot = [o["intent"] for o in store.orders("carry") if o["intent"] != "entry"]
        assert spot, "the spot leg was not unwound"
        halts = [e["message"] for n in ("carry", "carry~2") for e in store.events(n, limit=500)
                 if e["kind"] == "risk_halt"]
        assert any("drawdown" in m for m in halts), halts  # the summed halt raised beside the liquidation
    finally:
        node.dispose()


# --- the at-fill rule on any close (GAP-LIQ, _gap_liquidation), and every close carrying its liquidation price -------

def _closing(intent, liq=30_226.13):
    """A long perp position's close as the engine decided it: its decision row, with the liquidation price it was
    judged against (the engine's _sell_all puts it there), and stand-ins for the liquidation's own journaling."""
    booked = []
    s = SimpleNamespace(decisions={"c1": {"intent": intent, "reason": "a close", "signal": {"liquidation_px": liq}}},
                        _entry_side=1, _entry_px=60_000.0, _rebook=None, _exit_lock=None, _flip=None,
                        runtime=SimpleNamespace(), _liquidation_events=lambda *a: booked.append(("events", a)),
                        _on_liquidation=lambda *a: booked.append(("on", a)))
    return s, booked


@pytest.mark.parametrize("intent", ["risk_halt", "risk_pause", "pm_flatten", "leg_unwind", "exit", "stop_loss"])
def test_any_close_filled_past_the_liquidation_price_is_booked_as_a_liquidation(intent):
    from sleeve_fund.strategies.base import LongFlatStrategy

    s, booked = _closing(intent)
    assert LongFlatStrategy._gap_liquidation(s, "c1", "c1", 29_000.0)
    d = s.decisions["c1"]
    assert d["intent"] == "liquidation" and d["signal"]["market_px"] == 29_000.0 and booked
    assert d["signal"].get("relabelled_from", "stop_loss") == intent and s._rebook == "c1"


@pytest.mark.parametrize("intent, px", [("risk_halt", 31_000.0), ("entry", 29_000.0), ("liquidation", 29_000.0)])
def test_a_close_inside_the_liquidation_price_or_an_entry_is_left_as_it_is(intent, px):
    from sleeve_fund.strategies.base import LongFlatStrategy

    s, booked = _closing(intent)
    assert not LongFlatStrategy._gap_liquidation(s, "c1", "c1", px)
    assert s.decisions["c1"]["intent"] == intent and not booked


def test_every_close_of_a_margin_position_carries_its_liquidation_price():
    from sleeve_fund.strategies.base import LongFlatStrategy

    sent = []
    s = SimpleNamespace(_pending_exit=None, pair=None, _margin=True, _drop_kept=lambda earn: None,
                        _working=lambda: [], _lot=lambda: D("0.001"), _min_qty=lambda: D("0.001"),
                        _position_qty=lambda free: D("0.05"), _pos_side=lambda: 1,
                        _liq_signal=lambda: {"liquidation_px": 30_226.13},
                        _submit=lambda side, qty, intent, reason, values: sent.append((intent, values)))
    LongFlatStrategy._sell_all(s, "risk_halt", "Risk halt", {"equity": 1.0})
    assert sent == [("risk_halt", {"equity": 1.0, "liquidation_px": 30_226.13})]
