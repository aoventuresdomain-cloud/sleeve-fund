"""P2-8 step 5: a two-leg Strategy's paper node (HoE Option A, 14 Oct): one process, one hub connection, the first
leg on its venue's simulated account and the second on the venue's clone, an account of its own, coupled by a Pair.

HoE conditions pinned here: a single-leg node is built as before (1); the clone's feed and staleness are its
source's (2, with test_two_legs_hub_alias); each leg's fees and terms are the real venue's (3). Advisor MUSTs: the
clone resolves to the real venue outside the matching engine (1), fees follow each leg's market (2), and a clone
never reaches a demo or live account (4)."""

from __future__ import annotations

import gc
from types import SimpleNamespace

import pytest

from sleeve_fund.paper import hub_client
from sleeve_fund.paper import node as node_mod
from sleeve_fund.paper.config import SleeveConfig
from sleeve_fund.paper.hub_client import HubStatus, HubStatusView
from sleeve_fund.strategies.pair import LegFollower, Pair

HUB = ("127.0.0.1", 1)
LEG_LONG = {"side": 1, "ratio": "1", "leverage": "1"}
LEG_SHORT = {"side": -1, "ratio": "1", "leverage": "1"}
SPEC = SimpleNamespace(default_params={})  # the test-only model defined below reads its defaults from its module


def _legs(first_params=None, second_params=None, **kw):
    lead = SleeveConfig(name="carry", strategy="trend_filter", instrument="BTC/USD", venue="kraken",
                        bar_spec="15-MINUTE-LAST-INTERNAL", starting_balance=5000.0,
                        params={**(first_params or {}), "leg": LEG_LONG})
    follow = SleeveConfig(name="carry~2", strategy="trend_filter", instrument=kw.get("instrument", "BTC/USD"),
                          venue=kw.get("venue", "kraken"), bar_spec=kw.get("bar_spec", "15-MINUTE-LAST-INTERNAL"),
                          starting_balance=5000.0, params={**(second_params or {"market": "perp"}), "leg": LEG_SHORT})
    return lead, follow


@pytest.fixture
def built(monkeypatch):
    """Builds without running: what each account, the feed and each strategy were given."""
    seen = {"accounts": [], "data": [], "strategies": []}
    real_sandbox, real_cfg = node_mod._sandbox, hub_client.HubDataClientConfig

    def sandbox(*a, **k):
        seen["accounts"].append(a)
        return real_sandbox(*a, **k)

    class Cfg(real_cfg):
        def __init__(self, **k):
            super().__init__(**k)
            seen["data"].append(self)

    monkeypatch.setattr(node_mod, "_sandbox", sandbox)
    monkeypatch.setattr(hub_client, "HubDataClientConfig", Cfg)
    monkeypatch.setattr(node_mod.LiveNode, "add_strategy", lambda self, st: seen["strategies"].append(st),
                        raising=False)
    yield seen
    _drop(seen)


def _drop(seen):
    """Nautilus objects are freed on the thread that made them, not by a later test's server thread."""
    for k in seen:
        if hasattr(seen[k], "clear"):
            seen[k].clear()
    gc.collect()


def test_the_second_leg_trades_on_the_venue_clone_with_an_account_of_its_own(built):
    node, pair = node_mod.build_two_leg_node(*_legs(), log_level="ERROR", asset_fetch=dict, hub=HUB)
    lead, follow = built["strategies"]
    assert isinstance(pair, Pair) and pair.lead is lead and pair.follow is follow and isinstance(follow, LegFollower)
    assert str(lead._cfg.instrument_id) == "BTC/USD.KRAKEN"
    assert str(follow._cfg.instrument_id) == "BTC/USD.KRAKEN_P2"
    assert str(follow._cfg.bar_type) == str(lead._cfg.bar_type).replace(".KRAKEN-", ".KRAKEN_P2-")
    (a1, a2) = built["accounts"]
    assert (str(a1[0]), a1[2]) == ("KRAKEN", "KRAKEN-PAPER-CARRY")
    assert (str(a2[0]), a2[2]) == ("KRAKEN_P2", "KRAKEN_P2-PAPER-CARRY-2")
    assert (a1[3], a2[3]) == (False, True)  # spot on a cash account, the perp on margin: each its own type
    assert pair.sides[id(lead)] == 1 and pair.sides[id(follow)] == -1
    node.dispose()


def test_one_hub_connection_feeds_both_legs_the_second_by_its_clone(built):
    node_mod.build_two_leg_node(*_legs(), log_level="ERROR", asset_fetch=dict, hub=HUB)
    (cfg,) = built["data"]
    assert cfg.instrument_ids == ("BTC/USD.KRAKEN",)  # one feed: both legs trade BTC (a perp simulated on spot)
    assert cfg.aliases == (("BTC/USD.KRAKEN", "BTC/USD.KRAKEN_P2"),)
    lead, follow = built["strategies"]
    assert isinstance(lead.hub_status, HubStatus) and isinstance(follow.hub_status, HubStatusView)
    assert follow.hub_status.base is lead.hub_status  # the hub's word and lost minutes are the source's (cond 2)
    assert follow.hub_status.aliases == {"BTC/USD.KRAKEN_P2": "BTC/USD.KRAKEN"}


def test_two_instruments_are_both_subscribed_and_each_recovered_from_its_own_minutes(built, monkeypatch):
    asked = []
    monkeypatch.setattr(node_mod, "stored_minutes",
                        lambda venue, pair: lambda iid, a, b: asked.append((venue, pair, iid)) or [])
    node_mod.build_two_leg_node(*_legs(instrument="ETH/USD"), log_level="ERROR", asset_fetch=dict, hub=HUB)
    (cfg,) = built["data"]
    assert cfg.instrument_ids == ("BTC/USD.KRAKEN", "ETH/USD.KRAKEN")
    assert cfg.aliases == (("ETH/USD.KRAKEN", "ETH/USD.KRAKEN_P2"),)
    cfg.recover("ETH/USD.KRAKEN", 0, 1)
    cfg.recover("BTC/USD.KRAKEN", 0, 1)
    assert asked == [("KRAKEN", "ETH/USD", "ETH/USD.KRAKEN"), ("KRAKEN", "BTC/USD", "BTC/USD.KRAKEN")]


def test_each_legs_fees_are_its_own_markets_at_the_real_venue(built):
    lead_cfg, follow_cfg = _legs()
    node_mod.build_two_leg_node(lead_cfg, follow_cfg, log_level="ERROR", asset_fetch=dict, hub=HUB)
    lead, follow = built["strategies"]
    assert lead.fee_model.fees == lead_cfg.fees and follow.fee_model.fees == follow_cfg.fees
    assert lead_cfg.fees != follow_cfg.fees  # spot pays the venue's schedule, the perp its market's (MUST 2, cond 3)
    assert lead._cfg.perp is None and follow._cfg.perp is not None  # the clone's perp terms are the venue's


@pytest.mark.parametrize("change, why", [
    ({"venue": "binance", "second_params": {"market": "perp"}}, "trade on one venue"),
    ({"bar_spec": "1-HOUR-LAST-INTERNAL"}, "share one bar size"),
])
def test_legs_on_two_venues_or_bar_sizes_are_refused(built, change, why):
    second = change.pop("second_params", None)
    with pytest.raises(ValueError, match=why):
        node_mod.build_two_leg_node(*_legs(second_params=second, **change), log_level="ERROR", asset_fetch=dict,
                                    hub=HUB)


def test_a_two_leg_node_needs_the_hub(built, monkeypatch):
    monkeypatch.delenv("HUB_KRAKEN", raising=False)
    with pytest.raises(ValueError, match="market data hub"):
        node_mod.build_two_leg_node(*_legs(), log_level="ERROR", asset_fetch=dict)


def test_a_single_leg_node_is_built_as_before(built):
    """HoE condition 1: no alias, the hub client on the venue (not the routing client), one account, the strategy's
    own instrument and the hub's own status."""
    cfg = SleeveConfig(name="t", strategy="trend_filter", instrument="BTC/USD", venue="kraken",
                       bar_spec="15-MINUTE-LAST-INTERNAL", starting_balance=1000.0, params={})
    n = node_mod.build_node(cfg, log_level="ERROR", asset_fetch=dict, hub=HUB)
    (data,), (acct,), (s,) = built["data"], built["accounts"], built["strategies"]
    assert data.aliases == () and data.instrument_ids == ("BTC/USD.KRAKEN",)
    assert (str(acct[0]), acct[2]) == ("KRAKEN", "KRAKEN-PAPER-T")
    assert str(s._cfg.instrument_id) == "BTC/USD.KRAKEN" and type(s.hub_status) is HubStatus
    assert getattr(s, "pair", None) is None and not s.two_leg
    n.dispose()


def test_the_hub_client_routes_a_clones_subscriptions_only_when_it_has_clones():
    """With clones it is the node's routing client (no venue), so a clone's subscription reaches it; without, it is
    registered on its venue exactly as before."""
    seen = []

    class Probe(hub_client.HubDataClient):
        def __init__(self, **k):
            real = hub_client.MarketDataClient.__init__
            hub_client.MarketDataClient.__init__ = lambda self, **kw: seen.append(kw["venue"])
            try:
                super().__init__(**k)
            finally:
                hub_client.MarketDataClient.__init__ = real

    for aliases in ((), (("BTC/USD.KRAKEN", "BTC/USD.KRAKEN_P2"),)):
        cfg = hub_client.HubDataClientConfig(venue="KRAKEN", instrument_ids=("BTC/USD.KRAKEN",), aliases=aliases)
        Probe(name="HUB", config=cfg, cache=None, clock=None)
    assert [str(v) if v is not None else None for v in seen] == ["KRAKEN", None]


def test_a_venue_clone_never_reaches_a_demo_or_live_account():
    from sleeve_fund import mirror

    with pytest.raises(mirror.MirrorRefused, match="clone"):
        mirror.target_for(SimpleNamespace(venue="BINANCE_P2", params={"demo_mirror": True}))
    assert mirror.target_for(SimpleNamespace(venue="BINANCE", params={})) == "BYBIT"


def test_main_runs_both_legs_in_the_first_legs_process_and_refuses_the_second_on_its_own(tmp_path, monkeypatch):
    from sleeve_fund.paper import single_lock
    from sleeve_fund.store import Store

    url = f"sqlite:///{tmp_path}/n.db"
    monkeypatch.setenv("DATABASE_URL", url)
    store = Store(url)
    store.create_two_legs(name="carry", strategy="trend_filter", bar_spec="15-MINUTE-LAST-INTERNAL",
                          starting_balance=10_000,
                          first={"instrument": "BTC/USD", "params": {}, "venue": "KRAKEN", "side": 1},
                          second={"instrument": "BTC/USD", "params": {"market": "perp"}, "venue": "KRAKEN",
                                  "side": -1})
    monkeypatch.setattr(single_lock, "instance_lock", lambda name: object())
    monkeypatch.setattr(node_mod, "assert_portfolio_gate", lambda store: None)
    built = []

    class Ran:
        def run(self):
            pass

    monkeypatch.setattr(node_mod, "build_two_leg_node", lambda lead, follow, **k: built.append((lead, follow, k))
                        or (Ran(), None))
    monkeypatch.setattr(node_mod, "build_node", lambda *a, **k: pytest.fail("a two-leg Strategy built as one leg"))
    assert node_mod.main(["--db-sleeve", "carry"]) == 0
    ((lead, follow, k),) = built
    assert (lead.name, follow.name) == ("carry", "carry~2")
    r1, r2 = k["runtimes"]
    assert (r1.name, r2.name) == ("carry", "carry~2") and r1.store._exact is r2.store._exact  # one writer, both legs in order
    with pytest.raises(SystemExit, match="second leg of carry"):
        node_mod.main(["--db-sleeve", "carry~2"])


class _Leg:
    def __init__(self, restore=None, price=100.0):
        self._restore, self.price = restore, price

    def _price(self):
        return self.price



def test_a_retry_is_refused_once_the_journal_no_longer_holds_the_trade_legging(tmp_path):
    """Advisor leg-2 retry MUST 3 (HoQA C13, QT2's lost-coordinator cell): another writer moved the pair's journal row,
    so this coordinator lost the trade; its second leg is not retried (the pair unwinds instead)."""
    from datetime import datetime, timezone

    from sleeve_fund.store import Store

    store = Store(f"sqlite:///{tmp_path}/t.db")
    store.create_two_legs(name="carry", strategy="trend_filter", bar_spec="1-HOUR-LAST-INTERNAL",
                          starting_balance=10_000,
                          first={"instrument": "BTC/USD", "params": {}, "venue": "KRAKEN", "side": 1},
                          second={"instrument": "BTC/USD", "params": {"market": "perp"}, "venue": "KRAKEN",
                                  "side": -1})
    now = datetime.now(timezone.utc)
    store.set_pair_state("carry", expected=None, state="flat", ts=now)
    store.set_pair_state("carry", expected="flat", state="legging", trade_id="carry-t1", deadline=now, ts=now)
    p = Pair.__new__(Pair)
    p.lead, p.follow = SimpleNamespace(runtime=None), SimpleNamespace(runtime=None)
    p.lead.pair = p.follow.pair = p
    p._store, p._parent, p.trade_id = store, "carry", "carry-t1"
    assert Pair._retry_refusal(p, p.follow) is None  # still its trade, legging: the retry may go
    assert store.set_pair_state("carry", expected="legging", state="flat", ts=now, expected_trade_id="carry-t1")
    assert "lost the trade" in Pair._retry_refusal(p, p.follow)


def _paper_pair(lead_block=(False, None), follow_block=(False, None), lead_status="running",
                follow_status="running"):
    """A pair whose two paper legs' runtimes say what CHOKE says (entry_blocked) and the Strategy's own status
    (can_open is true only while running, as SleeveRuntime.can_open), no journal."""
    def rt(name, block, status):
        return SimpleNamespace(name=name, venue="KRAKEN", backtest=False, status=status,
                               can_open=lambda: status == "running", entry_blocked=lambda: block)

    p = Pair.__new__(Pair)
    p.lead = SimpleNamespace(runtime=rt("carry", lead_block, lead_status),
                             _cfg=SimpleNamespace(instrument_id="BTC/USD.KRAKEN"))
    p.follow = SimpleNamespace(runtime=rt("carry~2", follow_block, follow_status),
                               _cfg=SimpleNamespace(instrument_id="BTC/USD.KRAKEN"))
    p.lead.pair = p.follow.pair = p
    p._store = None
    return p


def test_a_retry_is_refused_on_stale_data_on_the_other_leg_though_it_can_open(monkeypatch):
    """CR F283-4 (Advisor leg-2 retry MUST 3): leg 1 filled, its feed goes stale, leg 2 is refused once: can_open is
    still true on both, but CHOKE on the other leg blocks, so the retry is not sent and the pair unwinds."""
    from sleeve_fund.strategies import pair as pair_mod

    monkeypatch.setattr(pair_mod.router, "check", lambda *a: None)
    assert Pair._retry_refusal(_paper_pair(), _paper_pair().follow) is None
    p = _paper_pair(lead_block=(True, "Stale data: no trade or quote for 90 s"))
    assert Pair._retry_refusal(p, p.follow) == "Stale data: no trade or quote for 90 s"
    p = _paper_pair(follow_block=(True, "The portfolio is halted"))
    assert Pair._retry_refusal(p, p.follow) == "The portfolio is halted"


@pytest.mark.parametrize("leg", ["lead", "follow"])
@pytest.mark.parametrize("status", ["paused", "halted"])
def test_a_retry_is_refused_on_a_strategy_pause_or_halt_on_either_paper_leg(monkeypatch, leg, status):
    """Advisor leg-2 retry MUST 3 (14 Oct 12:20 UK): the Strategy's own pause or halt on either leg, not only the
    portfolio's, refuses the retry, so the pair unwinds. CHOKE is clear here; can_open alone says no."""
    from sleeve_fund.strategies import pair as pair_mod

    monkeypatch.setattr(pair_mod.router, "check", lambda *a: None)
    p = _paper_pair(**{f"{leg}_status": status})
    assert Pair._retry_refusal(p, p.follow) == f"a leg is {status}"


def _journal_pair(tmp_path):
    from datetime import datetime, timezone

    from sleeve_fund.store import Store

    store = Store(f"sqlite:///{tmp_path}/t.db")
    store.create_two_legs(name="carry", strategy="trend_filter", bar_spec="1-HOUR-LAST-INTERNAL",
                          starting_balance=10_000,
                          first={"instrument": "BTC/USD", "params": {}, "venue": "KRAKEN", "side": 1},
                          second={"instrument": "BTC/USD", "params": {"market": "perp"}, "venue": "KRAKEN",
                                  "side": -1})
    now = datetime.now(timezone.utc)
    store.set_pair_state("carry", expected=None, state="flat", ts=now)
    store.set_pair_state("carry", expected="flat", state="legging", trade_id="carry-t1", deadline=now, ts=now)
    p = Pair.__new__(Pair)
    p.lead, p.follow = SimpleNamespace(runtime=None), SimpleNamespace(runtime=None)
    p.lead.pair = p.follow.pair = p
    p._store, p._parent, p.trade_id = store, "carry", "carry-t1"
    assert Pair._retry_refusal(p, p.follow) is None  # its trade, legging: the retry may go
    return store, p


def test_a_retry_is_refused_when_the_pairs_state_cant_be_read(tmp_path, monkeypatch):
    """Advisor leg-2 retry MUST 3 (14 Oct 12:20 UK): the persisted state unreadable at retry time (the journal
    errors) is a lost coordinator: no retry, the pair unwinds, and the note says why."""
    store, p = _journal_pair(tmp_path)

    def unreadable(name):
        raise RuntimeError("the journal is unreachable")

    monkeypatch.setattr(store, "pair_state", unreadable)
    why = Pair._retry_refusal(p, p.follow)
    assert why is not None and "can't read the pair's state" in why and "unreachable" in why


def test_a_retry_is_refused_when_the_pairs_state_row_is_gone(tmp_path, monkeypatch):
    """Advisor leg-2 retry MUST 3: no persisted state at retry time is a lost coordinator too."""
    store, p = _journal_pair(tmp_path)
    monkeypatch.setattr(store, "pair_state", lambda name: None)
    assert "lost the trade" in Pair._retry_refusal(p, p.follow)


def test_a_retry_is_refused_when_either_legs_coordinator_is_gone():
    """Advisor leg-2 retry MUST 3: a leg no longer coupled to this coordinator (the in-process side of a lost
    coordinator) refuses the retry on paper as in the backtest (test_two_legs_engine, cause no_coordinator)."""
    for leg in ("lead", "follow"):
        p = _paper_pair()
        getattr(p, leg).pair = None
        assert Pair._retry_refusal(p, p.follow) == "its coordinator is gone"


def test_a_retry_is_refused_when_the_legs_venue_pin_doesnt_match(monkeypatch):
    """CR F283-4: the venue-pin trigger of MUST 3, pinned: a pin mismatch on the retried leg unwinds the pair."""
    from sleeve_fund.strategies import pair as pair_mod

    seen = []

    def mismatch(name, instrument, venue):
        seen.append((name, instrument, venue))
        raise ValueError("carry~2 is pinned to another venue")

    monkeypatch.setattr(pair_mod.router, "check", mismatch)
    p = _paper_pair()
    assert Pair._retry_refusal(p, p.follow) == "carry~2 is pinned to another venue"
    assert seen == [("carry~2", "BTC/USD.KRAKEN", "KRAKEN")]


def test_a_restarted_pair_is_recovered_once_both_legs_are_back_and_priced_and_opens_nothing_before():
    p = Pair.__new__(Pair)
    p.lead, p.follow, p.recovered = _Leg(restore={"qty": 1}), _Leg(price=0.0), False
    ran = []
    p.recover = lambda: ran.append(1) or setattr(p, "recovered", True)
    p._state = "flat"
    assert "after the restart" in Pair.blocks_entry(p)
    p.ready()
    p.lead._restore = None
    p.ready()
    assert ran == []  # the second leg has no price yet: nothing to unwind at
    p.follow.price = 99.0
    p.ready()
    p.ready()
    assert ran == [1]


def test_a_two_leg_node_runs_on_the_hub_and_a_trade_the_journal_left_legging_with_both_legs_flat_is_closed(tmp_path):
    """The real node, fed by a local hub: both legs start on their own accounts and journals, the summed book is
    marked once, and a pair the journal says was legging when the process stopped, with nothing held, goes back to
    flat on the first tick both legs are priced (Pair.ready), with no order."""
    import threading
    import time
    from datetime import datetime, timedelta, timezone

    from nautilus_trader.model import CryptoPerpetual, Currency, InstrumentId, Price, Quantity, Symbol

    from sleeve_fund.hub.server import Fanout
    from sleeve_fund.paper.config import from_store
    from sleeve_fund.paper.queued import QueuedStore
    from sleeve_fund.paper.runtime import SleeveRuntime
    from sleeve_fund.store import Store

    usdt = Currency.from_str("USDT")

    def perp(sym, base):
        return CryptoPerpetual(InstrumentId.from_str(f"{sym}-PERP.BINANCE"), Symbol(sym), Currency.from_str(base),
                               usdt, usdt, False, 2, 3, Price.from_str("0.10"), Quantity.from_str("0.001"), 0, 0)

    insts = {str(i.id): i for i in (perp("BTCUSDT", "BTC"), perp("ETHUSDT", "ETH"))}
    hub = Fanout("BINANCE", known=lambda: set(insts), log=lambda *a: None,
                 instruments=lambda ids: [type(insts[i]).to_dict(insts[i]) for i in ids if i in insts])
    port = hub.start("127.0.0.1", 0)
    store = Store(f"sqlite:///{tmp_path}/t.db")
    store.create_two_legs(name="pairx", strategy="trend_filter", bar_spec="1-MINUTE-LAST-INTERNAL",
                          starting_balance=10_000,
                          first={"instrument": "BTC/USDT", "params": {"market": "perp"}, "venue": "BINANCE", "side": 1},
                          second={"instrument": "ETH/USDT", "params": {"market": "perp"}, "venue": "BINANCE",
                                  "side": -1})
    now = datetime.now(timezone.utc)
    assert store.set_pair_state("pairx", expected=None, state="flat", ts=now)
    assert store.set_pair_state("pairx", expected="flat", state="legging", trade_id="pairx-t1",
                                deadline=now - timedelta(seconds=5), ts=now)
    journal = QueuedStore(store)
    lead, follow = (from_store(store.sleeve(n)) for n in ("pairx", "pairx~2"))
    node, pair = node_mod.build_two_leg_node(lead, follow, log_level="ERROR",
                                             runtimes=(SleeveRuntime(journal, "pairx"), SleeveRuntime(journal, "pairx~2")),
                                             hub=("127.0.0.1", port), history=lambda *a, **k: [])
    assert not pair.recovered and "after the restart" in pair.blocks_entry()

    def feed():
        time.sleep(2)
        for k in range(30):
            ts = time.time_ns()
            for iid, px in (("BTCUSDT-PERP.BINANCE", 60000), ("ETHUSDT-PERP.BINANCE", 3000)):
                hub.publish({"t": "quote", "id": iid, "bid": f"{px:.2f}", "ask": f"{px + 0.1:.2f}", "bid_qty": "5.000",
                             "ask_qty": "5.000", "ts": ts, "recv": ts})
                hub.publish({"t": "trade", "id": iid, "px": f"{px:.2f}", "qty": "0.500", "side": "BUYER",
                             "tid": str(k), "ts": ts, "recv": ts})
            time.sleep(0.2)

    threading.Thread(target=feed, daemon=True).start()
    handle = node.handle()
    threading.Timer(9, handle.stop).start()
    try:
        node.run()
    finally:
        journal.settle(timeout=10)
        hub.stop()
    for name in ("pairx", "pairx~2"):
        kinds = [e["kind"] for e in store.events(name, limit=50)]
        assert "start" in kinds and "reconcile" in kinds  # each leg on its own account and journal
        assert store.orders(name) == []  # nothing held, nothing to unwind: no order
    assert pair.recovered and pair.state == "flat"
    row = store.pair_state("pairx")
    assert row["state"] == "flat" and row["marked_at"] is not None and row["hwm"] == 10_000  # both legs summed
    assert any(w == "recovered_flat" for _, w, _ in pair.log)
    node.dispose()


@pytest.mark.parametrize("gate", [False, True])
def test_a_spot_and_perp_pair_judges_its_guards_on_every_spot_trade_and_fill_without_failing(tmp_path, monkeypatch,
                                                                                            gate):
    """HoQA P2-8 (QT2's probe): on the real node, the spot leg of a spot + perp pair runs the guards on each trade print
    and each fill of either leg (Advisor M5). A spot leg has no liquidation price, so the guard reads none rather than
    failing on its missing perp terms (which fired handler_failed on every print and fill, leaving the summed halt,
    pause and ruling B's clamp to the 30-second tick)."""
    import threading
    import time
    from datetime import datetime, timezone
    from decimal import Decimal as D

    from nautilus_trader.model import OrderSide

    from sleeve_fund.hub.server import Fanout
    from sleeve_fund.paper.config import from_store
    from sleeve_fund.paper.queued import QueuedStore
    from sleeve_fund.paper.runtime import SleeveRuntime
    from sleeve_fund.store import Store
    from sleeve_fund.strategies import REGISTRY
    from sleeve_fund.strategies.base import LongFlatConfig, LongFlatStrategy
    from sleeve_fund.venues import venue
    from test_p21b_wiring_qa import bridge, portfolio_pass

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
                self._send_entry(OrderSide.BUY, D("0.02"), "entry", {"close": self._price()})

    judged = []
    real_book = LongFlatStrategy._guard_book

    def guard_book(self, *a, **k):
        judged.append(self.runtime.name)
        return real_book(self, *a, **k)

    liq_read = []
    real_liq = LongFlatStrategy._liq_now

    def liq_now(self, *a, **k):
        liq_read.append(self.runtime.name)
        return real_liq(self, *a, **k)

    marked = []
    real_mark = Pair.on_mark

    def on_mark(self, leg, price):  # CR F283-6: Advisor M5's call on each fill, pinned
        marked.append(leg.runtime.name)
        return real_mark(self, leg, price)

    monkeypatch.setitem(REGISTRY, "driven", (Driven, LongFlatConfig))
    monkeypatch.setattr(LongFlatStrategy, "_entry_atr_pct", lambda self: 0.05)
    monkeypatch.setattr(LongFlatStrategy, "_guard_book", guard_book)
    monkeypatch.setattr(LongFlatStrategy, "_liq_now", liq_now)
    monkeypatch.setattr(Pair, "on_mark", on_mark)
    inst = venue("kraken").instrument("BTC", "USDT")
    insts = {str(inst.id): inst}
    hub = Fanout("KRAKEN", known=lambda: set(insts), log=lambda *a: None,
                 instruments=lambda ids: [type(insts[i]).to_dict(insts[i]) for i in ids if i in insts])
    port = hub.start("127.0.0.1", 0)
    # TEC-2LEG-GATEOFF (HoE): outside the fund (gate off) the legs' own summed halt and pause judge each print; inside it
    # (gate on) they don't act, so the trade doesn't ask them (F-GUARD-OWN-EQ).
    store = Store(f"sqlite:///{tmp_path}/t.db", portfolio_gate=gate)
    store.create_two_legs(name="carry", strategy="driven", bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=10_000,
                          first={"instrument": "BTC/USDT", "params": {}, "venue": "KRAKEN", "side": 1},
                          second={"instrument": "BTC/USDT", "params": {"market": "perp"}, "venue": "KRAKEN",
                                  "side": -1})
    portfolio_pass(store, datetime.now(timezone.utc), D(20_000))
    if gate:
        bridge(store, 20_000)  # inside the fund (TEC-B34)
    journal = QueuedStore(store)
    lead, follow = (from_store(store.sleeve(n)) for n in ("carry", "carry~2"))
    node, pair = node_mod.build_two_leg_node(lead, follow, log_level="ERROR", asset_fetch=dict,
                                             runtimes=(SleeveRuntime(journal, "carry"),
                                                       SleeveRuntime(journal, "carry~2")),
                                             hub=("127.0.0.1", port), history=lambda *a, **k: [])
    prints_after_fill = 25

    def feed():
        time.sleep(2)
        for k in range(30 + prints_after_fill):
            ts = time.time_ns()
            hub.publish({"t": "quote", "id": "BTC/USDT.KRAKEN", "bid": "60000.00", "ask": "60000.10",
                         "bid_qty": "5.00000000", "ask_qty": "5.00000000", "ts": ts, "recv": ts})
            hub.publish({"t": "trade", "id": "BTC/USDT.KRAKEN", "px": "60000.00", "qty": "0.50000000",
                         "side": "BUYER", "tid": str(k), "ts": ts, "recv": ts})
            time.sleep(0.2)

    threading.Thread(target=feed, daemon=True).start()
    handle = node.handle()
    threading.Timer(15, handle.stop).start()
    try:
        node.run()
    finally:
        journal.settle(timeout=10)
        hub.stop()
    try:
        for name in ("carry", "carry~2"):
            assert [o["status"] for o in store.orders(name) if o["intent"] == "entry"] == ["filled"]
            failed = [e["message"] for e in store.events(name, limit=200) if e["kind"] == "handler_failed"]
            assert failed == [], failed
        assert pair.state == "hedged"
        # Held, the spot leg is judged on its own prints (well over the 30-second tick's once) and on the fills: the
        # account and liquidation distance always; the legs' own summed halt and pause only outside the fund, as inside
        # it they don't act (F-GUARD-OWN-EQ; the fund's brakes, T5, do).
        assert liq_read.count("carry") >= 10 and liq_read.count("carry~2") >= 2
        if gate:
            assert judged == [], judged
        else:
            assert judged.count("carry") >= 10 and judged.count("carry~2") >= 2
        # Advisor M5 (CR F283-6): each leg's own entry fill ran both legs' guards there and then (Pair.on_mark).
        assert {"carry", "carry~2"} <= set(marked)
    finally:
        node.dispose()


def test_an_outage_replays_risk_guard_judges_a_pair_on_both_legs_summed_as_the_live_tick_does():
    """CR F283-5: the outage replay's risk-guard branch judges the equity at the minute's worst price through
    _guard_book, as the live tick does. A hedged 5k + 5k carry whose spot leg fell from 50,000 to 40,000 is still worth
    10,000 summed, not the spot leg's 4,000 against the summed peak (a false 60% drawdown halt)."""
    from sleeve_fund.strategies.base import LongFlatStrategy

    seen = []

    def other_book(leg, price):
        seen.append(price)
        return 6_000.0, 10_000.0, -0.1  # the perp short, marked at the same worst price

    spot = SimpleNamespace(pair=SimpleNamespace(same_prices=True, other_book=other_book))
    spot._guard_book = lambda *a: LongFlatStrategy._guard_book(spot, *a)
    assert LongFlatStrategy._replayed_guard_equity(spot, 0.0, 0.1, 40_000.0) == 10_000.0
    assert seen == [40_000.0]
    single = SimpleNamespace(pair=None)
    single._guard_book = lambda *a: LongFlatStrategy._guard_book(single, *a)
    assert LongFlatStrategy._replayed_guard_equity(single, 0.0, 0.1, 40_000.0) == 4_000.0
