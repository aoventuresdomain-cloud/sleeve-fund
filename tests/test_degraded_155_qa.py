"""QA round on PR #155 (heads 73d3908, delta c966f13): every finding as a test, the open ones as strict xfails. Each
xfail must fail on the PR today; once fixed it XPASSes, which strict mode reports as an error, so the mark is then
removed. Fixed at c966f13 (marks removed): P1-D1, D2, D5, D6, D8 (D4, D7 at aa7c2c6). Fixed at 3be572a: P1-D3, D9 (D9 rewritten
to rule (c) by the Head of QA). Fixed at ba4f533/e7ee497/26fd993 (marks removed in the full re-run): R1 own feed, D10,
D16, D15 (full-margin case). Open on 26fd993: D11 (#146), D13 (Advisor 18:16), D15 on the shipped margin caps, D17.
Harness changes adopted from PE2's copy on 26fd993: _paper_from's own feed attaches stored_minutes as paper.node does
(R1; the node half is test_r1_an_own_feed_node_...), and _reg stubs the Binance contract lookup per test (the same
data lib155 sets at import). Synthetic data only, no venue called.

  cd <scripts dir> && BACKTEST_ISOLATE=0 PYTHONPATH=/tmp/claude-0/p155h:/tmp/claude-0/p155h/tests \
    /tmp/claude-0/v/bin/python -m pytest -q -p no:cacheprovider -c /tmp/claude-0/p155h/pyproject.toml \
    --rootdir=/tmp/claude-0/p155h test_degraded_155_qa.py -rxX
"""
import collections
import re
import json
import os
import shutil
import sys

from qa155_lib import *  # noqa: E402,F401,F403  (lib155 in the repo)
import pytest  # noqa: E402
from sleeve_fund import funding, risk  # noqa: E402
from sleeve_fund.store import Store  # noqa: E402
from sleeve_fund.strategies import REGISTRY  # noqa: E402
from sleeve_fund.strategies.base import LongFlatConfig, LongFlatStrategy  # noqa: E402

PERP = {"market": "perp", "allow_short": True}
SEEN: dict = {}
SPEC = type("Spec", (), {"default_params": {}})()  # paper.config.auto_warmup reads the strategy module's SPEC


class TConfig(LongFlatConfig):
    def __init__(self, *, at: int = 0, until: int = 2**62, side: int = 1, tag: str = "", **kw):
        super().__init__(**kw)
        self.at, self.until, self.side, self.tag = at, until, side, tag


class T(LongFlatStrategy):
    """Wants `side` from the bar closing at `at` (ns) until `until`, else flat; records every bar it is fed."""

    def update_indicators(self, bar):
        SEEN.setdefault(self._cfg.tag, []).append((pd.Timestamp(bar.ts_event, tz="UTC"), bar.volume.as_double()))

    def want_long(self, bar):
        return None

    def want_side(self, bar):
        return self._cfg.side if self._cfg.at <= bar.ts_event < self._cfg.until else 0


def stub_contract(monkeypatch):
    """tests/qa155_lib.py::stub_contract (PE2): Binance's BTCUSDT contract as lib155 stubs it at import (0.1 tick,
    0.001 lot, 5 USDT min notional), set per test so the file runs offline on its own."""
    monkeypatch.setattr(venues.VENUES["BINANCE"], "contract", lambda pair: {
        "price_precision": 1, "size_precision": 3, "min_quantity": 0.001, "min_notional": 5.0})


@pytest.fixture(autouse=True)
def _reg(monkeypatch, tmp_path):
    stub_contract(monkeypatch)  # harness, adopted from PE2's copy (26fd993): the lookup lib155 already stubs, per test
    monkeypatch.setitem(REGISTRY, "qa_t", (T, TConfig))
    monkeypatch.setattr(funding, "DEFAULT_ROOT", tmp_path / "funding")
    monkeypatch.setattr(funding, "fetch", lambda *a, **k: (_ for _ in ()).throw(OSError("QA: no venue calls")))
    SEEN.clear()
    profiles = dict(risk.PROFILES)  # harness (PE2, 6047b50): _liquidate_then edits the margin caps in place; put them back
    yield
    risk.PROFILES.clear()
    risk.PROFILES.update(profiles)


def ns(s):
    return pd.Timestamp(s, tz="UTC").value


def put_rates(times):
    p = funding._path("BINANCE", "BTC/USDT")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"rates": [[int(t.timestamp() * 1000), 0.0001 * (1 + i % 3)] for i, t in enumerate(times)]}))


def _fills(j):
    return [(pd.Timestamp(f["ts"]), f["side"], (j.orders_.get(f["order_id"]) or {}).get("intent"))
            for f in sorted(j.fills_, key=lambda f: f["ts"])]


def test_d1_no_decision_bar_is_made_from_nothing_on_the_execution_path(tmp_path):
    inst = binance_inst()
    m1 = synth_1m(days=1, seed=4, vol_day=0.01)
    hole_close = pd.Timestamp("2025-01-01 02:00", tz="UTC")
    hs = holed_store(tmp_path / "h", m1, list(pd.date_range(hole_close - pd.Timedelta(minutes=15), periods=15, freq="1min")))
    q = hs.read("BINANCE", "BTC/USDT", 15)
    ex = hs.read("BINANCE", "BTC/USDT", 1)[["open", "high", "low", "close", "volume"]]
    assert hole_close not in q.index  # the store builds no bar from nothing (rule 5a)
    backtest(inst, q, strategy="qa_t", params={**PERP, "at": 0, "side": 0, "tag": "d1"}, minutes=15, profile="aggressive",
             half_spread=0.0, exec_prices=ex, exec_minutes=1)
    assert hole_close not in [t for t, _ in SEEN["d1"]]


def _paper(inst, m1x, params, spec="15-MINUTE-LAST-INTERNAL", profile="aggressive"):
    from nautilus_trader.backtest import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.common import LoggerConfig, LogLevel
    from nautilus_trader.model import AccountType, BarType, Money, OmsType, TraderId
    from sleeve_fund import markets
    from sleeve_fund.instruments import ScheduleFeeModel, fill_model
    from sleeve_fund.paper.runtime import SleeveRuntime
    from sleeve_fund.store import utcnow
    st = Store.in_memory()
    st.create_sleeve(name="P", strategy="qa_t", instrument="BTC/USDT", bar_spec=spec, starting_balance=10_000,
                     risk_profile=profile, params=params, venue="BINANCE")
    fees = markets.fees_for(params, venues.venue("BINANCE").fees, "BINANCE")
    rt = SleeveRuntime(st, "P", tick_seconds=30)
    eng = BacktestEngine(BacktestEngineConfig(trader_id=TraderId.from_str("PAPER-001"),
                                              logging=LoggerConfig(stdout_level=LogLevel.ERROR)))
    fm = ScheduleFeeModel(fees)
    try:
        eng.add_venue(venue=inst.id.venue, oms_type=OmsType.NETTING, account_type=AccountType.MARGIN,
                      default_leverage=markets.VENUE_LEVERAGE, base_currency=None,
                      starting_balances=[Money(10_000, inst.quote_currency)], fee_model=fm, fill_model=fill_model())
        eng.add_instrument(inst)
        eng.add_data(ticks_from_bars(inst, m1x, 0.2))
        s = T(TConfig(instrument_id=inst.id, bar_type=BarType.from_str(f"{inst.id}-{spec}"),
                      assumed_taker_fee=float(fees.taker), assumed_half_spread=0.1 / 60_000, **params)).attach_runtime(rt)
        s.simulated_venue, s.fee_model = True, fm
        eng.add_strategy(s)
        eng.run()
    finally:
        rt.now = utcnow
        eng.dispose()
    return st


def test_d2_paper_and_backtest_take_the_same_decision_on_a_holed_slower_candle(tmp_path):
    inst = binance_inst()
    m1 = synth_1m(days=1, seed=8, vol_day=0.01)
    write_funding(m1.index[0], m1.index[-1], root=tmp_path / "funding")
    close = pd.Timestamp("2025-01-01 03:00", tz="UTC")
    holes = list(pd.date_range(close - pd.Timedelta(minutes=12), periods=2, freq="1min"))
    q = holed_store(tmp_path / "h", m1, holes).read("BINANCE", "BTC/USDT", 15)
    assert bool(q.loc[close, "degraded"])
    params = {**PERP, "at": close.value, "side": 1, "tag": "d2"}
    bt = _fills(backtest(inst, q, strategy="qa_t", params=params, minutes=15, profile="aggressive", half_spread=0.0).journal)
    m1x = m1[~(m1.index - pd.Timedelta(minutes=1)).isin(pd.DatetimeIndex(holes))]
    st = quiet(lambda: _paper(inst, m1x, params))
    paper = sorted(pd.Timestamp(f["ts"]) for f in st.fills("P", limit=100))
    assert bt[0][0] == close + pd.Timedelta(minutes=15)  # the backtest enters on the next whole bar
    assert paper[0].floor("min") == bt[0][0]


# P1-D3 MAJOR fixed at 3be572a (Advisor ruling (a), margin cap): strict mark removed by QA, delta run 6 Oct.
def test_d3_the_risk_pages_stress_loss_matches_what_the_engine_books_on_a_gap(tmp_path):
    from sleeve_fund.dashboard.riskops import most_it_can_lose
    inst = binance_inst()
    idx = pd.date_range("2025-01-01 01:00", periods=40, freq="1h", tz="UTC")
    write_funding(idx[0], idx[-1], rate_fn=lambda i, t: 0.0, root=tmp_path / "funding")
    c = np.full(40, 60_000.0)
    c[20:] = 30_000.0
    o = np.r_[c[0], c[:-1]]
    o[20] = 30_000.0
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c, "volume": 1e9}, index=idx)
    res = backtest(inst, df, strategy="qa_t", params={**PERP, "side": 1, "tag": "d3"}, minutes=60, profile="aggressive",
                   half_spread=0.0)
    e = sorted(res.journal.fills_, key=lambda f: f["ts"])[0]
    before = float(res.equity[res.equity.index < idx[20]].iloc[-1])
    booked = before - float(res.equity.iloc[-1])
    s = type("S", (), {"params": dict(PERP), "venue": "BINANCE", "name": "x"})()
    x = {"sleeve": s, "profile": risk.profile("aggressive"), "equity": before, "qty": e["qty"], "price": 60_000.0,
         "entry_px": e["price"], "cash": 10_000.0 - e["fee"] - e["qty"] * e["price"], "unrealised": 0.0,
         "position_value": e["qty"] * 60_000.0}
    shown = min(x["position_value"] * 0.5, most_it_can_lose(x))
    assert shown == pytest.approx(booked, abs=15.0)


# P1-D4 BLOCKER on d70ee9f (paper charged a phantom 12:00 settlement at the baseline rate when the venue lengthened its
# interval 4h -> 8h); fixed in aa7c2c6 (markets.settlement_wait). Kept as a regression test.
def test_d4_paper_charges_no_settlement_the_venue_never_made(tmp_path):
    inst = binance_inst()
    put_rates(pd.date_range("2025-10-03 00:00", "2025-10-03 08:00", freq="4h", tz="UTC"))  # then 8-hourly: next 16:00
    m1 = synth_1m(days=1, seed=3, vol_day=0.005, start="2025-10-03 07:40")[:300]  # to 12:40
    st = quiet(lambda: _paper(inst, m1, {**PERP, "side": 1, "tag": "d4"}, spec="1-MINUTE-LAST-INTERNAL", profile="balanced"))
    charged = [pd.Timestamp(x["ts"]).strftime("%H:%M") for x in st.funding("P", limit=100)]
    assert "08:00" in charged
    assert "12:00" not in charged


def _holed_bh(tmp_path):
    m1 = synth_1m(days=90, seed=31, vol_day=0.04)
    write_funding(m1.index[0], m1.index[-1], root=tmp_path / "funding")
    holes = random_holes(m1, 0.015, seed=6)
    return holed_store(tmp_path / "h", m1, holes).read("BINANCE", "BTC/USDT", 15)


def test_d5_the_degraded_bar_event_is_only_written_when_an_entry_was_actually_held_back(tmp_path):
    q = _holed_bh(tmp_path)
    runs = {}
    for v, px in (("gate", q), ("nogate", q.assign(degraded=False))):
        r = backtest(binance_inst(), px, strategy="buy_and_hold", params=dict(PERP), minutes=15, profile="balanced",
                     half_spread=0.0002)
        runs[v] = (_fills(r.journal), collections.Counter(e["kind"] for e in r.journal.events("backtest", limit=100000)))
    assert runs["gate"][0] == runs["nogate"][0]  # the gate deferred nothing
    assert runs["gate"][1].get("degraded_bar", 0) == 0


def test_d6_a_reset_of_a_dust_position_finishes():
    from sleeve_fund.supervisor import Supervisor
    st = Store.in_memory()
    st.create_sleeve(name="dust", strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                     starting_balance=10_000)
    st.record_fill("dust", side="BUY", qty=3e-8, price=86_000.0, fee=0.0, order_id="o1", trade_id="t1")
    st.request_reset("dust", "QA reset")
    sup = Supervisor(st, python="true")
    for _ in range(30):
        sup.reset_pending()
        for c in st.pending_commands("dust"):
            st.mark_applied(c["id"])
    assert st.pending_reset("dust") is None


# P1-D7 MINOR on main and d70ee9f (3h bars, no execution bars: a stop at 08:00:05 did not pay 08:00); fixed in aa7c2c6
# (a fill settles funding up to its own time first). Kept as a regression test; see P1-D9 for what that rule overdoes.
def test_d7_a_position_journaled_through_a_settlement_pays_it_without_execution_bars(tmp_path):
    put_rates(pd.date_range("2025-10-03 00:00", "2025-10-04 00:00", freq="8h", tz="UTC"))
    t0 = pd.Timestamp("2025-10-03 03:00", tz="UTC")
    sec = pd.date_range(t0, pd.Timestamp("2025-10-03 12:00", tz="UTC"), freq="5s")
    px = np.where(sec < pd.Timestamp("2025-10-03 08:00:05", tz="UTC"), 60_000.0 + np.arange(len(sec)) * 0.001, 59_000.0)
    dec = pd.Series(px, index=sec).resample("180min", closed="left", label="right", origin="start_day").ohlc()
    dec["volume"] = 1e9
    r = backtest(binance_inst(), dec, strategy="qa_t", minutes=180, profile="aggressive", half_spread=0.0,
                 params={**PERP, "at": ns("2025-10-03 06:00"), "until": ns("2025-10-03 12:00"), "side": 1,
                         "stop_loss": 0.005, "tag": "d7"})
    exit_ts = _fills(r.journal)[-1][0]
    assert exit_ts > pd.Timestamp("2025-10-03 08:00", tz="UTC")
    assert [pd.Timestamp(x["ts"]) for x in r.journal.funding_] == [pd.Timestamp("2025-10-03 08:00", tz="UTC")]


def test_d8_a_hole_inside_the_warm_up_window_is_named(tmp_path):
    from nautilus_trader.model import BarType
    from sleeve_fund.history import HistoryStore
    from sleeve_fund.paper.node import history_loader
    from sleeve_fund.strategies import TrendFilter, TrendFilterConfig
    now = pd.Timestamp.now(tz="UTC").floor("15min")
    opens = pd.date_range(end=now - pd.Timedelta(minutes=1), periods=2 * 1440, freq="1min")
    opens = opens[~opens.isin(pd.date_range(now - pd.Timedelta(hours=10), periods=180, freq="1min"))]
    hs = HistoryStore(tmp_path / "h")
    hs.append_bars("KRAKEN", "BTC/USD", [(int(t.value), 100.0, 100.0, 100.0, 100.0, 1.0) for t in opens], "live")
    events = []
    rt = type("R", (), {"name": "s1", "store": type("S", (), {"event": lambda self, *a, **k: events.append(a)})()})()
    inst = kraken_inst()
    cfg = TrendFilterConfig(instrument_id=inst.id, bar_type=BarType.from_str("BTC/USD.KRAKEN-15-MINUTE-LAST-INTERNAL"),
                            fast=5, slow=20, assumed_taker_fee=0.004, warmup_bars=100)
    s = TrendFilter(cfg).attach_history(history_loader("KRAKEN", "BTC/USD", hs))
    s.instrument, s.runtime = inst, rt
    s._warm_from_history()
    msg = " ".join(e[3] for e in events if e[2] == "warmup")
    assert "hole" in msg or "missing" in msg or "gap" in msg


@pytest.mark.parametrize("gap_at", ["2025-10-04 00:00:01", "2025-10-04 10:00:01"])
def test_d9_a_gap_stop_on_daily_bars_pays_only_the_settlements_before_the_gap(tmp_path, gap_at):
    times = (pd.date_range("2025-10-01 00:00", "2025-10-03 16:00", freq="8h", tz="UTC")
             .append(pd.date_range("2025-10-03 20:00", "2025-10-06 00:00", freq="4h", tz="UTC")))
    put_rates(times)
    sec = pd.date_range("2025-10-01 00:00:01", "2025-10-05 23:59:46", freq="15s", tz="UTC")
    px = np.round(60_000.0 + (np.arange(len(sec)) % 40) * 0.5, 1)
    gap = pd.Timestamp(gap_at, tz="UTC")
    px = np.where(sec >= gap, np.round(px * 0.95, 1), px)
    tape = pd.Series(px, index=sec)
    day = tape.resample("1D", closed="left", label="right").ohlc()
    day["volume"] = 1e9
    params = {**PERP, "at": ns("2025-10-02 00:00"), "side": 1, "stop_loss": 0.02, "tag": "d9"}
    r = backtest(binance_inst(), day, strategy="qa_t", params=params, minutes=1440, profile="balanced", half_spread=0.0)
    owed = [t for t in times if pd.Timestamp("2025-10-02 00:00", tz="UTC") < t < gap]  # what paper pays (s10_gapfund)
    paid = {pd.Timestamp(x["ts"]): x["amount"] for x in r.journal.funding_}
    if gap == gap.floor("D") + pd.Timedelta(seconds=1):  # the bar opens gapped: the fill is held to its open, exactly
        assert list(paid) == owed
        return
    # Advisor P1-D9 rule (c), 16:20: the stop is touched at an unknown time inside the bar, so it takes the worse
    # outcome: an in-bar settlement is charged only if it is a cost, never a credit. The 1-minute backtest stays the
    # reference and the bars-only result may differ from it only for the worse. (Rewritten by QA 17:58 for 3be572a.)
    one = tape.resample("1min", closed="left", label="right").ohlc()
    one["volume"] = 1e9
    ref = backtest(binance_inst(), day, strategy="qa_t", params=params, minutes=1440, profile="balanced",
                   half_spread=0.0, exec_prices=one, exec_minutes=1)
    ref_paid = {pd.Timestamp(x["ts"]): x["amount"] for x in ref.journal.funding_}
    assert list(ref_paid) == owed  # the reference pays what paper pays
    assert set(ref_paid) <= set(paid) and all(paid[t] == pytest.approx(ref_paid[t]) for t in ref_paid)
    extra = {t: a for t, a in paid.items() if t not in ref_paid}
    bar_close = gap.floor("D") + pd.Timedelta(days=1)
    assert extra, paid  # the in-bar settlements after the touch cost a long at these rates: rule (c) charges them
    assert all(gap < t <= bar_close and a < 0 for t, a in extra.items()), extra  # inside that bar only, costs only
    assert sorted(extra) == [t for t in times if gap < t <= bar_close]  # the worse outcome: every in-bar cost
    assert sum(paid.values()) <= sum(ref_paid.values())  # bars-only is never better than the 1-minute reference


def _paper_from(inst, m1x, params, hs, hub, spec_minutes=15, profile="aggressive", refills=()):
    """Paper started (deployed or restarted) at m1x's first minute, the history store `hs` holding the minutes the
    hub wrote before. hub=True: fed as paper.node builds a hub-fed node (hub_client.Decoder with HubStatus and
    recover=node.stored_minutes on `hs`, EXTERNAL bars, bar_execution off); hub=False: on its own trade feed.
    refills: (1-minute bar row from m1 by close time, receive time) the hub relays late, as its gap-fill does."""
    from nautilus_trader.backtest import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.common import LoggerConfig, LogLevel
    from nautilus_trader.model import AccountType, BarType, Money, OmsType, TraderId
    from sleeve_fund import markets
    from sleeve_fund.instruments import ScheduleFeeModel, fill_model
    from sleeve_fund.paper.hub_client import Decoder, HubStatus
    from sleeve_fund.paper.node import stored_minutes
    from sleeve_fund.paper.runtime import SleeveRuntime
    from sleeve_fund.store import utcnow
    spec = f"{spec_minutes}-MINUTE-LAST-{'EXTERNAL' if hub else 'INTERNAL'}"
    st = Store.in_memory()
    st.create_sleeve(name="P", strategy="qa_t", instrument="BTC/USDT", bar_spec=f"{spec_minutes}-MINUTE-LAST-INTERNAL",
                     starting_balance=10_000, risk_profile=profile, params=params, venue="BINANCE")
    fees = markets.fees_for(params, venues.venue("BINANCE").fees, "BINANCE")
    rt = SleeveRuntime(st, "P", tick_seconds=30)
    eng = BacktestEngine(BacktestEngineConfig(trader_id=TraderId.from_str("PAPER-001"),
                                              logging=LoggerConfig(stdout_level=LogLevel.ERROR)))
    fm = ScheduleFeeModel(fees)
    status = HubStatus() if hub else None
    told = []
    try:
        eng.add_venue(venue=inst.id.venue, oms_type=OmsType.NETTING, account_type=AccountType.MARGIN,
                      default_leverage=markets.VENUE_LEVERAGE, base_currency=None,
                      starting_balances=[Money(10_000, inst.quote_currency)], fee_model=fm, fill_model=fill_model(),
                      bar_execution=not hub)
        eng.add_instrument(inst)
        data = ticks_from_bars(inst, m1x, 0.2)
        if hub:
            dec = Decoder(spec, lambda lv, kind, msg: told.append((kind, msg)),
                          stored_minutes("BINANCE", "BTC/USDT", store=hs), status)
            msgs = [(int(t.value) + 2_000_000_000, t, r, False) for t, r in m1x.iterrows()]  # relayed 2 s late
            msgs += [(int(now.value), t, r, True) for (t, r), now in refills]
            for now, t, r, refilled in sorted(msgs, key=lambda m: m[0]):
                b = dec({"t": "bar", "id": str(inst.id), "o": f"{r.open:.1f}", "h": f"{r.high:.1f}",
                         "l": f"{r.low:.1f}", "c": f"{r.close:.1f}", "v": f"{max(r.volume, 0.001):.3f}",
                         "ts": int(t.value), **({"refilled": True} if refilled else {})}, now)
                if b is not None:
                    data.append(b)
        eng.add_data(data)
        s = T(TConfig(instrument_id=inst.id, bar_type=BarType.from_str(f"{inst.id}-{spec}"),
                      assumed_taker_fee=float(fees.taker), assumed_half_spread=0.1 / 60_000, **params)).attach_runtime(rt)
        s.simulated_venue, s.fee_model = True, fm
        if hub:
            s.hub_fed, s.hub_status = True, status
        else:  # as paper.node builds an own-feed node: its first bar from the store's minutes (R1; harness change, ba4f533)
            s.attach_minutes(stored_minutes("BINANCE", "BTC/USDT", store=hs))
        eng.add_strategy(s)
        eng.run()
    finally:
        rt.now = utcnow
        eng.dispose()
    return st, told


# R1 own feed fixed at ba4f533 (paper.node attaches the stored minutes): strict marks removed, full re-run.


@pytest.mark.parametrize("hub,stored_holes", [
    pytest.param(True, 0, id="hub-fed-all-stored"),
    pytest.param(True, 1, id="hub-fed-one-missing"),
    pytest.param(True, 2, id="hub-fed-two-missing"),
    pytest.param(True, "refilled", id="hub-fed-one-refilled-after-the-restart"),
    pytest.param(False, 0, id="own-feed-all-stored"),
    pytest.param(False, 1, id="own-feed-one-missing"),
    pytest.param(False, 2, id="own-feed-two-missing"),
])
def test_r1_a_restart_mid_bar_builds_its_first_bar_from_the_minutes_the_hub_stores(tmp_path, hub, stored_holes):
    """Ruling HoE + Advisor 6 Oct 17:18: after a restart the first bar is built from the minutes the hub already
    stores; it is degraded only for minutes truly missing from the store (after the hub's normal gap-fill); the
    venue's own candle is never used. Restarted at 02:52, seven minutes into the 15m bar closing 03:00; the model
    wants long from that bar. The decision (entry time) and the degraded mark equal a backtest on the same stored
    minutes: all stored (backtest enters 03:00); 02:48 missing for good (1 of 15, not degraded: 03:00); 02:47-02:48
    missing for good (2 of 15, degraded: 03:15); 02:47-02:48 not yet stored at the restart but refilled by the hub
    at 02:55:30 (the store then holds all 15: 03:00, where unrefilled they would make it degraded). own-feed-two-missing passes only because the bar is degraded
    anyway (8 of 15 minutes seen)."""
    inst = binance_inst()
    m1 = synth_1m(days=1, seed=8, vol_day=0.01)
    write_funding(m1.index[0], m1.index[-1], root=tmp_path / "funding")
    close = pd.Timestamp("2025-01-01 03:00", tz="UTC")
    restart = pd.Timestamp("2025-01-01 02:52", tz="UTC")  # the first minute open after the restart
    refilled = stored_holes == "refilled"
    n = 0 if refilled else stored_holes
    holes = list(pd.date_range("2025-01-01 02:48", periods=n, freq="-1min", tz="UTC"))
    hs = holed_store(tmp_path / "h", m1, holes)
    q = hs.read("BINANCE", "BTC/USDT", 15)  # what the backtest sees: the store after the hub's gap-fill
    assert int(q.loc[close, "missing"]) == n and bool(q.loc[close, "degraded"]) == (n > 1)
    params = {**PERP, "at": close.value, "side": 1, "tag": "r1"}
    bj = backtest(inst, q, strategy="qa_t", params=params, minutes=15, profile="aggressive", half_spread=0.0).journal
    bt = _fills(bj)
    assert bt[0][0] == close + pd.Timedelta(minutes=15 if n > 1 else 0)
    m1x = m1[m1.index - pd.Timedelta(minutes=1) >= restart]  # the live feed after the restart (the store has the rest)
    refills = ()
    if refilled:  # at the restart the store lacks 02:47-02:48 (degraded if left); the hub's gap-fill relays them
        gone = list(pd.date_range("2025-01-01 02:47", periods=2, freq="1min", tz="UTC"))
        hs = holed_store(tmp_path / "h-at-restart", m1, gone)
        refills = [((k, m1.loc[k]), pd.Timestamp("2025-01-01 02:55:30", tz="UTC"))
                   for k in (g + pd.Timedelta(minutes=1) for g in gone)]  # by close time
    st, told = quiet(lambda: _paper_from(inst, m1x, params, hs, hub, refills=refills))
    paper = sorted(pd.Timestamp(f["ts"]) for f in st.fills("P", limit=100))
    held_back = [e for e in st.events("P", limit=1000) if e["kind"] == "degraded_bar"]
    bt_held = [e for e in bj.events("backtest", limit=1000) if e["kind"] == "degraded_bar"]
    assert paper and paper[0].floor("min") == bt[0][0]  # the same decision as the backtest on the stored minutes
    assert len(held_back) == len(bt_held)  # degraded exactly where the store says so


# P1-D10 MINOR fixed at ba4f533 (decision_bars subtracts each execution bar's stored missing minutes): mark removed.
def test_d10_decision_bars_count_minutes_not_execution_bars(tmp_path):
    from sleeve_fund.research import runner
    m1 = synth_1m(days=1, seed=11, vol_day=0.01)
    close = pd.Timestamp("2025-01-01 03:00", tz="UTC")
    holes = [pd.Timestamp(f"2025-01-01 02:{m}", tz="UTC") for m in ("46", "51", "56")]
    hs = holed_store(tmp_path / "h", m1, holes)
    q = hs.read("BINANCE", "BTC/USDT", 15)
    ex5 = hs.read("BINANCE", "BTC/USDT", 5)
    assert int(q.loc[close, "missing"]) == 3 and bool(q.loc[close, "degraded"])
    d = runner.decision_bars(ex5, 15, 5)
    assert (int(d.loc[close, "missing"]), bool(d.loc[close, "degraded"])) == (3, True)


@pytest.mark.xfail(strict=True, reason="P1-D11 MAJOR, pre-existing since #140 (owner: #146's late-bar flip): a 15m bar "
                   "missing its last 2 minutes, with the next bar's first minute missing too, is complete at the hub client "
                   "only 120 s after its close, so it is skipped as late: paper (hub-fed) neither updates its indicators "
                   "nor exits on it and exits at 03:15, where the backtest on the same stored minutes exits at 03:00 on the "
                   "degraded bar (s16 pattern 5)")
def test_d11_hub_fed_paper_exits_on_a_holed_bar_whose_closing_minutes_are_missing(tmp_path):
    inst = binance_inst()
    m1 = synth_1m(days=1, seed=8, vol_day=0.01)
    write_funding(m1.index[0], m1.index[-1], root=tmp_path / "funding")
    close = pd.Timestamp("2025-01-01 03:00", tz="UTC")
    holes = list(pd.date_range("2025-01-01 02:58", periods=3, freq="1min", tz="UTC"))
    hs = holed_store(tmp_path / "h", m1, holes)
    q = hs.read("BINANCE", "BTC/USDT", 15)
    assert bool(q.loc[close, "degraded"])
    params = {**PERP, "at": ns("2025-01-01 02:00"), "until": close.value, "side": 1, "tag": "d11"}
    bt = [f for f in _fills(backtest(inst, q, strategy="qa_t", params=params, minutes=15, profile="aggressive",
                                     half_spread=0.0).journal) if f[1] == "SELL"]
    assert bt[0][0] == close  # the exit runs on the degraded bar
    m1x = m1[~(m1.index - pd.Timedelta(minutes=1)).isin(pd.DatetimeIndex(holes))]
    st, told = quiet(lambda: _paper_from(inst, m1x, params, hs, hub=True))
    sells = sorted(pd.Timestamp(f["ts"]) for f in st.fills("P", limit=100) if f["side"] == "SELL")
    assert sells and sells[0].floor("min") == close


# ---- Advisor 17:57 (#155 post-liquidation), pinned by QA at the Head of QA's request -------------------------------

@pytest.fixture
def _full_margin(monkeypatch):
    """tests/conftest.py::full_margin: every profile puts the whole equity up as margin, so liquidation is in reach."""
    import dataclasses
    for name, p in list(risk.PROFILES.items()):
        monkeypatch.setitem(risk.PROFILES, name, dataclasses.replace(p, max_position_pct=1.0))


def _record_session(path, meta, legs, px=60_000.0, size=1.0):
    """tests/test_long_short.py::_record (73d3908): a recorded paper session, quotes and trades every second, the
    price moving by each leg's share over its minutes; a leg of 0 minutes is a gap."""
    from nautilus_trader.model import AggressorSide, Price, Quantity, QuoteTick, TradeId, TradeTick
    from sleeve_fund.paper.recorder import Recorder
    from test_replay import START
    inst = venues.venue("KRAKEN").instrument("BTC", "USD", price_precision=1)
    rec = Recorder(path)
    rec.meta = meta
    rec.start(inst)
    s = 0
    for minutes, move in legs:
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


def _session_meta(balance, params):
    return {"balances": [f"{balance:.2f} USD"],
            "sleeve": {"name": "ping-pong-test", "strategy": "ping_pong", "instrument": "BTC/USD",
                       "bar_spec": "1-MINUTE-LAST-INTERNAL", "starting_balance": 10_000,
                       "risk_profile": "balanced", "params": params,
                       "maker_fee": "0.0002", "taker_fee": "0.0005", "tick_seconds": 30}}


# P1-D15 full-margin case fixed at 26fd993 (X with both fees): mark removed. Shipped caps: see the pin further down.
def test_p1_after_a_liquidation_the_strategy_stays_halted_through_a_resume_and_a_restart(tmp_path, _full_margin):
    """Advisor 6 Oct 17:57 (#155 post-liquidation), modelled on tests/test_long_short.py::test_a_strategy_wiped_out_by_
    a_gap_is_marked_at_zero_and_halted_through_a_restart at 73d3908 (the version with the resume/restart half). A paper
    short at 2x on full margin is gapped +60% through its liquidation price. It stays halted through a PM resume and a
    restart until an explicit reset after liquidation; no new order; every mark after the restart is the equity kept
    (isolated margin: only the position's margin is lost), not zero and not the last mark before the gap; risk_halt
    twice; and the halt text says "Position margin lost (liquidated): X, Y% of strategy equity", X the margin lost
    to the cent and Y = X over the strategy's equity before the gap, both as the journal has them."""
    import re
    from sleeve_fund.research.replay import replay
    name, params = "ping-pong-test", {"rise": 0.01, "dip": 0.005, **PERP}
    store = Store.in_memory()
    gap = tmp_path / "gap.jsonl.gz"
    _record_session(gap, _session_meta(10_000, params), [(5, 0.0), (20, 0.015), (0, 0.6), (5, 0.0)])  # short, +60%
    orders = replay(gap, store=store)
    assert orders[-1]["intent"] == "liquidation" and orders[-1]["side"] == "BUY"
    s = store.sleeve(name)
    assert s.status == "halted", (s.status, s.status_reason)
    fills = sorted(store.fills(name, limit=100), key=lambda f: f["ts"])
    # The short and its liquidation fill in slices (QA correction 18:35, PE2's dispute checked on e7ee497): X is the
    # whole position's margin plus its entry fee and the liquidation fee (Advisor 18:17 point 4, as in D3).
    by_order = {}
    for f in fills:
        by_order.setdefault(f["order_id"], []).append(f)
    liq_fills = by_order[orders[-1]["order_id"]]
    open_fills = by_order[orders[-2]["order_id"]]  # the short the gap liquidated
    assert orders[-2]["intent"] == "entry" and orders[-2]["side"] == "SELL"
    liq = max(liq_fills, key=lambda f: f["ts"])
    assert sum(f["qty"] for f in liq_fills) == pytest.approx(sum(f["qty"] for f in open_fills))
    margin_lost = round(sum(f["qty"] * f["price"] for f in open_fills) / risk.profile("balanced").max_leverage
                        + sum(f["fee"] for f in open_fills) + sum(f["fee"] for f in liq_fills), 2)
    marks = store.equity_series(name)
    eq_before = [m for m in marks if pd.Timestamp(m["ts"]) < pd.Timestamp(liq["ts"])][-1]["equity"]
    book = store.journal_book(name, 10_000)
    left = book["cash"]
    assert book["qty"] == 0 and left > 0  # isolated margin: what wasn't margined is kept

    def says_margin_lost(text):
        m = re.search(r"Position margin lost \(liquidated\): ([\d,]+\.\d\d), (\d+(?:\.\d+)?)% of strategy equity", text)
        assert m, text
        x, y = float(m.group(1).replace(",", "")), m.group(2)
        assert x == pytest.approx(margin_lost, abs=0.005), (x, margin_lost)
        dp = len(y.partition(".")[2])
        assert float(y) == pytest.approx(100 * x / eq_before, abs=0.5 * 10 ** -dp + 1e-9), (y, 100 * x / eq_before)

    # The PM resumes it and the process restarts: it stays halted, with nothing new traded.
    store.command(name, "resume", "try again")
    store.create_sleeve = lambda **kw: store.sleeve(kw["name"])
    seen = len(store.equity_series(name))
    restart = tmp_path / "restart.jsonl.gz"
    _record_session(restart, _session_meta(left, params), [(5, 0.0)], px=97_440.0)
    assert len(replay(restart, store=store)) == len(orders)  # no new order
    assert not [c for c in store.pending_commands(name) if c["command"] == "resume"]  # the resume was taken up
    s = store.sleeve(name)
    assert s.status == "halted", (s.status, s.status_reason)  # after the resume and the restart
    after = store.equity_series(name)[seen:]
    assert after and all(m["qty"] == 0.0 and m["equity"] == pytest.approx(left, abs=0.01) for m in after), after[:3]
    assert all(abs(m["equity"] - eq_before) > 1.0 for m in after)  # not the last mark before the gap
    events = store.events(name, limit=500)
    assert [e["kind"] for e in events].count("risk_halt") == 2
    assert "mark_unavailable" not in {e["kind"] for e in events}
    says_margin_lost(s.status_reason)


class EConfig(TConfig):
    def __init__(self, *, place_at: int = 0, trigger: float = 0.0, qty: float = 0.1, **kw):
        super().__init__(**kw)
        self.place_at, self.trigger, self.qty = place_at, trigger, qty


class E(T):
    """T, plus a resting stop ENTRY (STOP_MARKET, side `side`) placed at the close of the bar at `place_at`."""

    def on_bar(self, bar):
        super().on_bar(bar)
        c = self._cfg
        if bar.ts_event == c.place_at and str(bar.bar_type) == str(c.bar_type).split("@")[0]:
            from nautilus_trader.model import OrderSide, TimeInForce
            side = OrderSide.BUY if c.side > 0 else OrderSide.SELL
            order = self.order_factory.stop_market(instrument_id=c.instrument_id, order_side=side,
                                                   quantity=self.instrument.make_qty(c.qty),
                                                   trigger_price=self.instrument.make_price(c.trigger),
                                                   time_in_force=TimeInForce.GTC)
            self.decisions[str(order.client_order_id)] = {"intent": "entry", "reason": "QA resting stop entry",
                                                          "signal": {}}
            self.submit_order(order)


@pytest.mark.parametrize("side", [
    pytest.param(1, id="long-in-bar-costs"),  # P1-D16 fixed at ba4f533/e7ee497: mark removed
    pytest.param(-1, id="short-in-bar-credits"),
])
def test_p1_rule_c_applies_to_a_resting_entry_touched_inside_a_bar(tmp_path, monkeypatch, side):
    """Advisor 6 Oct 17:57 (b): the worse-of funding rule (c) applies to resting ENTRIES touched inside a bar too.
    Daily bars only, a resting stop entry placed at the 4 Oct 00:00 close and touched at about 10:04 on 4 Oct (the
    price ramps through it, so the fill price is the trigger in both runs and only funding can differ). Settlements
    8h, then 4h from 3 Oct 20:00, all rates positive: in-bar settlements after the touch (12:00, 16:00, 20:00 and
    5 Oct 00:00) cost a long and would credit a short. Rule (c): each in-bar one after the touch is charged if it
    costs the new position, never credited; the 1-minute execution-bar backtest is the reference; bars-only is
    never better than it."""
    monkeypatch.setitem(REGISTRY, "qa_e", (E, EConfig))
    times = (pd.date_range("2025-10-01 00:00", "2025-10-03 16:00", freq="8h", tz="UTC")
             .append(pd.date_range("2025-10-03 20:00", "2025-10-06 00:00", freq="4h", tz="UTC")))
    put_rates(times)
    sec = pd.date_range("2025-10-01 00:00:01", "2025-10-05 23:59:46", freq="15s", tz="UTC")
    px = 60_000.0 + (np.arange(len(sec)) % 40) * 0.5  # 60,000 to 60,019.5
    ramp_at = pd.Timestamp("2025-10-04 10:00:01", tz="UTC")
    j = np.clip(((sec - ramp_at) / pd.Timedelta(seconds=15)).astype(int), 0, 40)
    up, down = 60_020.0 + j, 60_000.0 - (j + 1)
    px = np.where(sec >= ramp_at, up if side > 0 else down, px)
    trigger = 60_039.0 if side > 0 else 59_980.0  # touched at the last trade of the 10:04 minute, never gapped
    tape = pd.Series(np.round(px, 1), index=sec)
    day = tape.resample("1D", closed="left", label="right").ohlc()
    day["volume"] = 1e9
    one = tape.resample("1min", closed="left", label="right").ohlc()
    one["volume"] = 1e9
    params = {**PERP, "at": ns("2025-10-05 00:00"), "side": side, "tag": "e", "place_at": ns("2025-10-04 00:00"),
              "trigger": trigger}
    bo = backtest(binance_inst(), day, strategy="qa_e", params=params, minutes=1440, profile="balanced", half_spread=0.0)
    rf = backtest(binance_inst(), day, strategy="qa_e", params=params, minutes=1440, profile="balanced",
                  half_spread=0.0, exec_prices=one, exec_minutes=1)
    fb, fr = (sorted(r.journal.fills_, key=lambda f: f["ts"]) for r in (bo, rf))
    assert len(fb) == len(fr) == 1 and fb[0]["price"] == pytest.approx(trigger) and fr[0]["price"] == pytest.approx(trigger)  # the same entry, at the trigger
    paid = {pd.Timestamp(x["ts"]): x["amount"] for x in bo.journal.funding_}
    ref = {pd.Timestamp(x["ts"]): x["amount"] for x in rf.journal.funding_}
    bar_open, bar_close = pd.Timestamp("2025-10-04 00:00", tz="UTC"), pd.Timestamp("2025-10-05 00:00", tz="UTC")
    touch = pd.Timestamp("2025-10-04 10:05", tz="UTC")
    in_bar_after = [t for t in times if touch <= t <= bar_close]
    assert list(ref) == [t for t in times if t > touch]  # the reference pays every settlement it held through
    assert all((ref[t] < 0) == (side > 0) for t in ref)  # positive rates: a long pays, a short receives
    # After the bar, both runs book the same settlements alike.
    assert {t: a for t, a in paid.items() if t > bar_close} == pytest.approx({t: a for t, a in ref.items() if t > bar_close})
    inside = {t: a for t, a in paid.items() if bar_open < t <= bar_close}
    assert all(a < 0 for a in inside.values()), inside  # never a credit inside the bar
    if side > 0:  # costs: every in-bar settlement after the touch is charged, as the reference charges it
        assert all(t in inside and inside[t] == pytest.approx(ref[t]) for t in in_bar_after), (inside, in_bar_after)
    else:  # credits: none is booked
        assert not [t for t in in_bar_after if t in paid], paid
    assert sum(paid.values()) <= sum(ref.values()) + 1e-9  # bars-only never better on funding
    assert float(bo.equity.iloc[-1]) <= float(rf.equity.iloc[-1]) + 0.01  # nor on equity


@pytest.mark.xfail(strict=True, reason="P1-D13 MAJOR, pre-existing (outside rule (c), which covers funding): on daily bars "
                   "only, a 2% stop touched inside the day by a 5% gap at 10:00:01 fills at its trigger (58,819.1), "
                   "while the market traded through to 57,000 and the 1-minute reference fills there: bars-only ends "
                   "about 193 (1.9% of capital) better than the reference. Advisor ruled 18:16 (pessimistic bars-only stop fills, G1 never judges bars-only resting exits); not built in #155")
def test_d13_a_bars_only_stop_gapped_through_inside_the_bar_is_never_better_than_the_1_minute_run(tmp_path):
    times = (pd.date_range("2025-10-01 00:00", "2025-10-03 16:00", freq="8h", tz="UTC")
             .append(pd.date_range("2025-10-03 20:00", "2025-10-06 00:00", freq="4h", tz="UTC")))
    put_rates(times)
    sec = pd.date_range("2025-10-01 00:00:01", "2025-10-05 23:59:46", freq="15s", tz="UTC")
    px = np.round(60_000.0 + (np.arange(len(sec)) % 40) * 0.5, 1)
    gap = pd.Timestamp("2025-10-04 10:00:01", tz="UTC")
    px = np.where(sec >= gap, np.round(px * 0.95, 1), px)
    tape = pd.Series(px, index=sec)
    day = tape.resample("1D", closed="left", label="right").ohlc()
    day["volume"] = 1e9
    one = tape.resample("1min", closed="left", label="right").ohlc()
    one["volume"] = 1e9
    params = {**PERP, "at": ns("2025-10-02 00:00"), "side": 1, "stop_loss": 0.02, "tag": "d13"}
    bo = backtest(binance_inst(), day, strategy="qa_t", params=params, minutes=1440, profile="balanced", half_spread=0.0)
    rf = backtest(binance_inst(), day, strategy="qa_t", params=params, minutes=1440, profile="balanced",
                  half_spread=0.0, exec_prices=one, exec_minutes=1)
    assert float(bo.equity.iloc[-1]) <= float(rf.equity.iloc[-1]) + 0.01


def test_r1_an_own_feed_node_attaches_the_stored_minutes_before_its_first_bar(tmp_path, monkeypatch):
    """R1, the node half (Head of QA, re-run on ba4f533): _paper_from's own-feed branch attaches
    stored_minutes("BINANCE", "BTC/USDT", store=hs) itself, so the R1 cases above prove the strategy side only. Here
    paper.node.build_node builds an own-feed node (no HUB_BINANCE) as production does, and the strategy it adds already
    carries a minutes loader when it is added (before the node can run, so before any bar); the loader reads the
    history store (the same one the hub writes and research reads) for that venue and pair, the same rows as the
    harness's loader, and it is the same function the hub-fed node uses as its recover (node.py builds both with
    stored_minutes(profile.name, sleeve.instrument))."""
    from sleeve_fund import history
    from sleeve_fund.paper import node as node_mod
    from sleeve_fund.paper.config import SleeveConfig
    for k in list(os.environ):
        if k.upper().startswith(("HUB_", "KRAKEN_", "BINANCE_")):
            monkeypatch.delenv(k)
    m1 = synth_1m(days=1, seed=8, vol_day=0.01)
    hs = holed_store(tmp_path / "h", m1, [pd.Timestamp("2025-01-01 02:48", tz="UTC")])
    monkeypatch.setattr(history, "DEFAULT_ROOT", tmp_path / "h")
    added = []
    monkeypatch.setattr(node_mod.LiveNode, "add_strategy",
                        lambda self, st: added.append((st, st.minutes_loader, st.hub_fed)), raising=False)
    cfg = SleeveConfig(name="t", strategy="trend_filter", instrument="BTC/USDT", venue="binance",
                       bar_spec="15-MINUTE-LAST-INTERNAL", starting_balance=1000.0, params={"market": "perp"})
    n = node_mod.build_node(cfg, log_level="ERROR", asset_fetch=dict)
    try:
        ((s, loader, hub_fed),) = added
        assert loader is not None and not hub_fed  # attached at build, before the strategy is added and any bar
        after, before = ns("2025-01-01 02:45"), ns("2025-01-01 02:52") + 1
        rows = loader(str(s._cfg.instrument_id) if hasattr(s, "_cfg") else "x", after, before)
        ref = node_mod.stored_minutes("BINANCE", "BTC/USDT", store=hs)("x", after, before)
        assert rows == ref and len(rows) == 6  # 02:45-02:51 opens, closing 02:46-02:52, less the 02:48 hole
        assert [pd.Timestamp(r[0], tz="UTC") for r in rows][0] == pd.Timestamp("2025-01-01 02:46", tz="UTC")
    finally:
        n.dispose()
        added.clear()


def _liquidate_then(tmp_path, *, side, profile, pct, balance=10_000.0, size=1.0, after=("resume",)):
    """s21_liq.py as a helper: ping_pong on a perp, recorded and replayed as paper, liquidated by a 60% gap against
    it (pct: the profiles' margin cap, None keeps the shipped ones). Then each of `after` in turn ("resume": a PM
    resume; "stopstart": the PM's Stop then Start through the dashboard's /sleeves/{name}/command and the supervisor's
    step), each followed by a restarted process on a session that trades both ways (a dip, a rise, a dip), so a
    strategy free to trade does. Each later session starts two hours after the one before."""
    import dataclasses
    import test_replay
    from sleeve_fund.research.replay import replay
    if pct is not None:
        for nm, p in list(risk.PROFILES.items()):
            risk.PROFILES[nm] = dataclasses.replace(p, max_position_pct=pct)
    name, params = "ping-pong-test", {"rise": 0.01, "dip": 0.005, **PERP}

    def meta(bal):
        m = _session_meta(bal, params)
        m["sleeve"].update(risk_profile=profile, starting_balance=balance)
        return m
    store = Store(f"sqlite:///{tmp_path}/t.db")
    up = side == "short"
    gap = tmp_path / "gap.jsonl.gz"
    _record_session(gap, meta(balance), [(5, 0.0), (20, 0.015 if up else -0.012), (0, 0.6 if up else -0.6), (5, 0.0)],
                    size=size)
    orders = replay(gap, store=store)
    first = {"orders": orders, "status": store.sleeve(name).status, "reason": store.sleeve(name).status_reason}
    fills = store.fills(name, limit=200)
    entry = [o for o in orders if o["intent"] == "entry"][-1]
    opened = [f for f in fills if f["order_id"] == entry["order_id"]]
    closed = [f for f in fills if f["order_id"] == orders[-1]["order_id"]]
    lev = risk.profile(profile).max_leverage
    # X (Advisor 18:17 point 4): the whole position's margin plus its entry fee and the liquidation fee
    first["margin"] = (sum(f["qty"] * f["price"] for f in opened) / lev + sum(f["fee"] for f in opened)
                       + sum(f["fee"] for f in closed))
    liq_ts = min(pd.Timestamp(f["ts"]) for f in closed)
    # Y (18:17): on the mark-to-market equity just before the liquidation: the last mark up to its fill still holding
    # the position (the gap's own mark, when it was marked before the fill), else the last one before the gap
    held = [m for m in store.equity_series(name) if pd.Timestamp(m["ts"]) <= liq_ts and m["qty"] != 0 and m["equity"] > 0]
    first["equity_before"] = held[-1]["equity"]
    left = store.journal_book(name, balance)["cash"]
    store.create_sleeve = lambda **kw: store.sleeve(kw["name"])
    start0, steps = test_replay.START, []
    px = 60_000.0 * (1.015 * 1.6 if up else 0.988 * 0.4)
    try:
        for i, step in enumerate(after):
            if step == "resume":
                store.command(name, "resume", "try again")
            else:
                from fastapi.testclient import TestClient
                from sleeve_fund import supervisor as sup
                from sleeve_fund.dashboard import app as app_mod
                os.environ.setdefault("DASHBOARD_PASSWORD", "qa-pw")
                os.environ.setdefault("TEARSHEET_DIR", str(tmp_path))

                class FakePopen:
                    pid, returncode = 4242, None
                    poll = lambda self: None  # noqa: E731
                    send_signal = wait = kill = lambda self, *a, **k: 0  # noqa: E731
                c = TestClient(app_mod.create_app(store))
                auth, same = ("pm", os.environ["DASHBOARD_PASSWORD"]), {"origin": "http://testserver"}
                real_popen, sup.subprocess.Popen = sup.subprocess.Popen, lambda *a, **k: FakePopen()
                try:
                    sv = sup.Supervisor(store)
                    sv.procs[name] = sup.Proc()
                    sv.procs[name].popen, sv.procs[name].started_at = FakePopen(), sup.utcnow()
                    assert c.post(f"/sleeves/{name}/command", data={"command": "stop", "reason": "QA stop"}, auth=auth,
                                  headers=same, follow_redirects=False).status_code == 303
                    sv.step()
                    assert store.sleeve(name).status == "stopped"  # the supervisor writes "stopped" over the halt
                    assert c.post(f"/sleeves/{name}/command", data={"command": "start", "reason": "QA start"}, auth=auth,
                                  headers=same, follow_redirects=False).status_code == 303
                    sv.procs[name].popen = None
                    sv.step()
                finally:
                    sup.subprocess.Popen = real_popen
            test_replay.START = start0 + (i + 1) * 2 * 3600 * 10**9
            path = tmp_path / f"after{i}.jsonl.gz"
            _record_session(path, meta(left), [(3, -0.02), (3, 0.03), (4, -0.02)], px=px)
            got = replay(path, store=store)
            s = store.sleeve(name)
            steps.append({"step": step, "new_orders": got[len(orders):], "status": s.status, "reason": s.status_reason})
            orders = got
    finally:
        test_replay.START = start0
    events = store.events(name, limit=5000)
    return first, steps, events, left


def _says_margin_lost(text, margin, equity_before):
    """The ruled halt text (Advisor 6 Oct 17:57 (a), 18:17 point 4): X the margin plus the entry and liquidation fees
    to the cent, Y = X over the mark-to-market equity just before the liquidation, as the journal has them, to the
    figures shown."""
    import re
    m = re.search(r"Position margin lost \(liquidated\): ([\d,]+\.\d\d), (\d+(?:\.\d+)?)% of strategy equity", text)
    assert m, text
    x, y = float(m.group(1).replace(",", "")), m.group(2)
    assert x == pytest.approx(margin, abs=0.005), (x, margin)
    dp = len(y.partition(".")[2])
    assert float(y) == pytest.approx(100 * x / equity_before, abs=0.5 * 10 ** -dp + 1e-9), (y, 100 * x / equity_before)
    assert float(y) > 0, text  # a loss is never shown as 0%


# P1-D15 shipped caps, P1-D17 and P1-D18 fixed on the next head after 26fd993 (PE2): marks removed.
@pytest.mark.parametrize("side,profile", [("short", "balanced"), ("long", "aggressive")])
def test_p1_after_a_liquidation_on_the_shipped_margin_caps_the_strategy_stays_halted(tmp_path, side, profile):
    """Advisor 6 Oct 17:57: halted through a resume and a restart until an explicit reset after liquidation, whatever
    margin the profile puts up; the halt reads the ruled text throughout."""
    first, steps, events, left = _liquidate_then(tmp_path, side=side, profile=profile, pct=None, after=("resume",))
    assert first["orders"][-1]["intent"] == "liquidation"
    _says_margin_lost(first["reason"], first["margin"], first["equity_before"])
    (st,) = steps
    assert not st["new_orders"], st
    assert st["status"] == "halted" and st["reason"].startswith("Position margin lost (liquidated): "), st


@pytest.mark.parametrize("side,profile", [("short", "balanced"), ("long", "aggressive")])
def test_hc_a_small_liquidation_stays_halted_through_the_pms_stop_and_start(tmp_path, side, profile):
    """Head of QA adversarial case (HC, owned by PE2's stop-safety PR; reported on #155): a liquidation that loses
    about 10% of strategy equity, below the drawdown limit, so the old high-water mark can't re-halt it. The PM's Stop
    and Start through the dashboard's command path (the supervisor writes "stopped" over "halted"), then a restart on
    a session that trades: still halted with the liquidation's text, no new order; then a PM resume: the same."""
    first, steps, events, left = _liquidate_then(tmp_path, side=side, profile=profile, pct=0.1,
                                                 after=("stopstart", "resume"))
    assert first["orders"][-1]["intent"] == "liquidation" and first["status"] == "halted"
    assert 5 < 100 * first["margin"] / first["equity_before"] < 20  # under the drawdown limit (20% / 35%)
    for st in steps:
        assert not st["new_orders"], st
        assert st["status"] == "halted" and st["reason"].startswith("Position margin lost (liquidated): "), st
        assert "drawdown" not in st["reason"], st  # not a fresh drawdown text


@pytest.mark.parametrize("side,profile,pct,balance,size", [
    pytest.param("short", "balanced", 0.1, 10_000.0, 1.0, id="short-2x-10pct"),
    pytest.param("long", "aggressive", 0.1, 10_000.0, 1.0, id="long-3x-10pct"),
    pytest.param("short", "balanced", 0.1, 250_000.0, 25.0, id="short-2x-10pct-thousands"),
    pytest.param("short", "balanced", 1.0, 10_000.0, 1.0, id="short-2x-full-margin-right-after"),
    pytest.param("long", "aggressive", 0.004, 1_250_000.0, 50.0, id="long-3x-under-1pct-thousands"),
])
def test_p1_the_liquidation_halt_gives_x_and_y_as_the_journal_has_them(tmp_path, side, profile, pct, balance, size):
    """Advisor 17:57 (a) and 18:17 point 4, adversarial: X with thousands separators and both fees, Y on the
    mark-to-market equity just before the liquidation, on small liquidations (about 10% of equity, long and short,
    2x and 3x) and one under 1% of equity in the millions."""
    first, steps, events, left = _liquidate_then(tmp_path, side=side, profile=profile, pct=pct, balance=balance,
                                                 size=size, after=())
    assert first["orders"][-1]["intent"] == "liquidation" and first["status"] == "halted"
    if first["margin"] >= 1000:
        assert re.search(r"\(liquidated\): \d{1,3}(,\d{3})+\.\d\d, ", first["reason"]), first["reason"]  # commas
    _says_margin_lost(first["reason"], first["margin"], first["equity_before"])


@pytest.mark.parametrize("side", [pytest.param(1, id="long-pays"), pytest.param(-1, id="short-receives")])
def test_p1_rule_c_a_resting_entry_filled_on_a_gap_at_the_open_pays_and_receives_the_in_bar_settlements(
        tmp_path, monkeypatch, side):
    """Rule (c) as PE2 reads it, Advisor 16:20 and 17:57 (b): an entry filled on a gap at the bar's OPEN has a known fill
    time, so it is held through every settlement inside the bar after the open and both pays and receives them (only
    an in-bar touch at an unknown time is costs-only). Daily bars only vs the 1-minute reference: a resting stop entry
    placed at the 4 Oct 00:00 close; the 4 Oct bar opens gapped 1% through it and stays there, so the price at every
    settlement is the fill price on both runs. Rates positive, 4h from 3 Oct 20:00: a long pays 04:00 to 5 Oct 00:00,
    a short receives them. The gap fill is journaled at the bar's open (P1-D12)."""
    monkeypatch.setitem(REGISTRY, "qa_e", (E, EConfig))
    times = (pd.date_range("2025-10-01 00:00", "2025-10-03 16:00", freq="8h", tz="UTC")
             .append(pd.date_range("2025-10-03 20:00", "2025-10-06 00:00", freq="4h", tz="UTC")))
    put_rates(times)
    sec = pd.date_range("2025-10-01 00:00:01", "2025-10-05 23:59:46", freq="15s", tz="UTC")
    px = 60_000.0 + (np.arange(len(sec)) % 40) * 0.5
    bar_open, bar_close = pd.Timestamp("2025-10-04 00:00", tz="UTC"), pd.Timestamp("2025-10-05 00:00", tz="UTC")
    gapped = 60_600.0 if side > 0 else 59_400.0
    px = np.where(sec > bar_open, gapped, px)
    trigger = 60_300.0 if side > 0 else 59_700.0
    tape = pd.Series(np.round(px, 1), index=sec)
    day = tape.resample("1D", closed="left", label="right").ohlc()
    day["volume"] = 1e9
    one = tape.resample("1min", closed="left", label="right").ohlc()
    one["volume"] = 1e9
    assert day.loc[bar_close, "open"] == gapped  # the bar opens through the trigger
    params = {**PERP, "at": ns("2025-10-05 00:00"), "side": side, "tag": "g", "place_at": ns("2025-10-04 00:00"),
              "trigger": trigger}
    bo = backtest(binance_inst(), day, strategy="qa_e", params=params, minutes=1440, profile="balanced", half_spread=0.0)
    rf = backtest(binance_inst(), day, strategy="qa_e", params=params, minutes=1440, profile="balanced",
                  half_spread=0.0, exec_prices=one, exec_minutes=1)
    (fb,), (fr,) = bo.journal.fills_, rf.journal.fills_
    assert fb["price"] == pytest.approx(gapped) and fr["price"] == pytest.approx(gapped)  # filled at the open
    assert pd.Timestamp(fb["ts"]) == bar_open, fb["ts"]  # P1-D12: journaled at the open, not the bar's close
    paid = {pd.Timestamp(x["ts"]): x["amount"] for x in bo.journal.funding_}
    ref = {pd.Timestamp(x["ts"]): x["amount"] for x in rf.journal.funding_}
    in_bar = [t for t in times if bar_open < t <= bar_close]
    assert len(in_bar) == 6 and all(t in ref for t in in_bar)
    assert all((ref[t] < 0) == (side > 0) for t in ref)  # a long pays, a short receives
    assert {t: paid.get(t) for t in in_bar} == pytest.approx({t: ref[t] for t in in_bar})  # credits too, for a short
    assert paid == pytest.approx(ref)  # nothing before the fill, the same after the bar
    assert float(bo.equity.iloc[-1]) == pytest.approx(float(rf.equity.iloc[-1]), abs=0.01)


# ---- #155 full round on 6047b50 (QA, 6 Oct evening) -------------------------------------------------------------------

def _x_as_the_fill_path_computes_it(store, name, order_id, held_qty, held_px, profile="balanced", last_equity=None,
                                    trade_id=None, mark_px=None):
    """LongFlatStrategy.on_order_filled's X for a liquidation, called on the real methods with a stand-in for the
    strategy (they read only self.runtime and self._liquidated). held_qty, held_px: the position before this fill.
    Tolerant of both heads' shapes:
      6047b50 (base.py ~2843): fees, taken, before = _liquidation_figures(order_id); X = _margin_lost(max(taken, held),
        ...), always replacing the mark's estimate;
      eb737bd (base.py ~2854): fees, taken, before, journaled = _liquidation_figures(trade_id); the same X, but while
        the fill `trade_id` is not journaled yet the mark's estimate (self._liquidated) is kept.
    mark_px: the mark path ran first on that price, underwater (on_bar, ~2611 / ~2622), so self._liquidated holds its
    estimate with the taker fee on what is still held, as each head computes it."""
    import inspect
    import types
    from sleeve_fund.paper.runtime import SleeveRuntime
    rt = SleeveRuntime(store, name)
    rt.taker_fee, rt._last_equity = 0.0005, last_equity
    me = types.SimpleNamespace(runtime=rt, _liquidated=None)
    figures, lost = LongFlatStrategy._liquidation_figures, LongFlatStrategy._margin_lost
    by_trade = list(inspect.signature(figures).parameters)[1] == "trade_id"  # eb737bd keys the fill by its trade id
    if by_trade and trade_id is None:
        raise TypeError("this head's _liquidation_figures takes the fill's trade_id: pass trade_id=")
    if mark_px is not None:
        r = figures(me, None, held_qty * mark_px)
        me._liquidated = lost(me, held_qty + (r[1] if by_trade else 0.0), held_px, r[0], r[2])
    r = figures(me, trade_id if by_trade else order_id)
    fees, taken, before = r[:3]
    margin = lost(me, max(taken, held_qty), held_px, fees, before)
    if len(r) > 3 and not r[3]:
        margin = me._liquidated or margin  # its fill isn't in the journal yet: keep the estimate
    return margin


def _x_of(text):
    m = re.search(r"\(liquidated\): ([\d,]+\.\d\d), ", text)
    assert m, text
    return float(m.group(1).replace(",", ""))


def _seed_short(store, name, t0):
    """A 2x short of 0.3 at 60,000 (two slices, 9.00 fees) on a 10,000 strategy, marked just before the gap."""
    from datetime import timedelta
    store.create_sleeve(name=name, strategy="ping_pong", instrument="BTC/USDT", venue="binance",
                        bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=10_000, risk_profile="balanced",
                        params={"rise": 0.01, "dip": 0.005, **PERP})
    for i, q in enumerate((0.2, 0.1)):
        store.record_fill(name, side="SELL", qty=q, price=60_000.0, fee=q * 60_000.0 * 0.0005, order_id="ENTRY",
                          trade_id=f"E{i}", ts=t0)
    store.record_equity(name, equity=9_991.0, cash=9_991.0, qty=-0.3, price=60_000.0, benchmark=10_000.0,
                        ts=t0 + timedelta(minutes=5))


def test_p1_x_covers_the_whole_position_when_two_liquidation_orders_close_it(tmp_path):
    """Advisor 18:17 point 4: X = margin + entry fee + liquidation fee over the WHOLE position, grouped by order_id
    (every liquidation order's fills). A risk stop journaled as a liquidation takes 0.2 of a 0.3 short at the gap, the
    guard's liquidation order takes the last 0.1; X is the whole 0.3's margin and every fee."""
    from datetime import timedelta
    t0 = pd.Timestamp("2025-10-03 00:00", tz="UTC").to_pydatetime()
    store, name = Store(f"sqlite:///{tmp_path}/x1.db"), "liq-x1"
    _seed_short(store, name, t0)
    gap = t0 + timedelta(minutes=10)
    store.record_order(name, order_id="RISKSTOP", side="BUY", qty=0.2, intent="liquidation", reason="QA", order_type="STOP", ts=gap)
    store.record_fill(name, side="BUY", qty=0.2, price=97_000.0, fee=0.2 * 97_000.0 * 0.0005, order_id="RISKSTOP",
                      trade_id="R0", ts=gap)
    store.record_order(name, order_id="GUARD", side="BUY", qty=0.1, intent="liquidation", reason="QA", ts=gap)
    store.record_fill(name, side="BUY", qty=0.1, price=97_000.0, fee=0.1 * 97_000.0 * 0.0005, order_id="GUARD",
                      trade_id="G0", ts=gap)
    want = 0.3 * 60_000.0 / 2 + 9.0 + 0.3 * 97_000.0 * 0.0005  # 9,000 + 9.00 + 14.55
    x = _x_of(_x_as_the_fill_path_computes_it(store, name, "GUARD", held_qty=0.1, held_px=60_000.0, trade_id="G0"))
    assert x == pytest.approx(want, abs=0.005), (x, want)


def test_p1_x_keeps_the_liquidation_fee_when_its_fill_row_is_late(tmp_path):
    """Advisor 18:17 point 4. The liquidation of the whole 0.3 short filled at the venue, but its journal row is late:
    X must not shrink (it keeps the liquidation fee, estimated at the taker rate, and says so) and, once the row lands,
    the halt gives the journal's figure."""
    from datetime import timedelta
    t0 = pd.Timestamp("2025-10-03 00:00", tz="UTC").to_pydatetime()
    store, name = Store(f"sqlite:///{tmp_path}/x2.db"), "liq-x2"
    _seed_short(store, name, t0)
    full = 0.3 * 60_000.0 / 2 + 9.0 + 0.3 * 97_000.0 * 0.0005
    text = _x_as_the_fill_path_computes_it(store, name, "LIQ", held_qty=0.3, held_px=60_000.0, trade_id="L0",
                                           mark_px=97_000.0)  # halted on the gap mark first; fill "L0" not journaled
    x = _x_of(text)
    assert x == pytest.approx(full, abs=0.005) or ("estimat" in text or "provisional" in text), (x, full, text)
    assert x >= full - 0.005, (x, full)  # never smaller than what the liquidation lost


def test_p1_x_after_a_partial_reduce_and_a_restart_is_the_remainder_with_its_share_of_the_entry_fee(tmp_path):
    """CR case 1 (6047b50), through the engine: a 2x short of 0.3325 on full margin, a third bought back (journaled),
    a restart that restores the rest from the journal, then a +60% gap liquidates it. X = the remainder's margin, plus
    the entry fee pro-rated to the remainder, plus the liquidation fee (all its slices)."""
    import dataclasses
    import test_replay
    from sleeve_fund.research.replay import replay
    for nm, p in list(risk.PROFILES.items()):
        risk.PROFILES[nm] = dataclasses.replace(p, max_position_pct=1.0)
    name, params = "ping-pong-test", {"rise": 0.01, "dip": 0.005, **PERP}
    store = Store(f"sqlite:///{tmp_path}/b.db")
    start0 = test_replay.START
    try:
        s1 = tmp_path / "s1.jsonl.gz"
        _record_session(s1, _session_meta(10_000, params), [(5, 0.0), (20, 0.015), (2, 0.0)])
        orders = replay(s1, store=store)
        short = store.journal_book(name, 10_000)["qty"]
        assert short < 0
        red, px = round(abs(short) / 3, 8), 60_000.0 * 1.015
        store.record_order(name, order_id="QA-REDUCE", side="BUY", qty=red, intent="exit", reason="QA partial reduce")
        store.record_fill(name, side="BUY", qty=red, price=px, fee=round(red * px * 0.0005, 8), order_id="QA-REDUCE",
                          trade_id="QA-REDUCE", ts=pd.Timestamp(start0 + 3600 * 10**9, tz="UTC").to_pydatetime())
        store.create_sleeve = lambda **kw: store.sleeve(kw["name"])
        book = store.journal_book(name, 10_000)
        test_replay.START = start0 + 2 * 3600 * 10**9
        s2 = tmp_path / "s2.jsonl.gz"
        _record_session(s2, _session_meta(book["cash"] + book["qty"] * book["entry_px"], params),
                        [(5, 0.0), (0, 0.6), (5, 0.0)], px=px)
        orders = replay(s2, store=store)
    finally:
        test_replay.START = start0
    liq = [o for o in orders if o["intent"] == "liquidation"]
    assert liq
    fills = store.fills(name, limit=200)
    entry = [o for o in orders if o["intent"] == "entry"][-1]
    ef = [f for f in fills if f["order_id"] == entry["order_id"]]
    lf = [f for f in fills if f["order_id"] in {o["order_id"] for o in liq}]
    q0 = sum(f["qty"] for f in ef)
    left = q0 - red
    assert sum(f["qty"] for f in lf) == pytest.approx(left)
    avg = sum(f["qty"] * f["price"] for f in ef) / q0
    want = left * avg / 2 + sum(f["fee"] for f in ef) * left / q0 + sum(f["fee"] for f in lf)
    s = store.sleeve(name)
    assert s.status == "halted"
    assert _x_of(s.status_reason) == pytest.approx(want, abs=0.01), (s.status_reason, want)


def test_p1_x_after_a_restart_holding_the_position_keeps_the_stored_entry_fee(tmp_path):
    """CR case 2 (6047b50), through the engine: the short is opened in one process, the process restarts (the
    position restored from the journal by an unjournaled, free order), then the gap liquidates it. X includes the
    original entry fee from the stored fills and equals the in-process figure (10,106.69)."""
    import dataclasses
    import test_replay
    from sleeve_fund.research.replay import replay
    for nm, p in list(risk.PROFILES.items()):
        risk.PROFILES[nm] = dataclasses.replace(p, max_position_pct=1.0)
    name, params = "ping-pong-test", {"rise": 0.01, "dip": 0.005, **PERP}
    store = Store(f"sqlite:///{tmp_path}/a.db")
    start0 = test_replay.START
    try:
        s1 = tmp_path / "s1.jsonl.gz"
        _record_session(s1, _session_meta(10_000, params), [(5, 0.0), (20, 0.015), (2, 0.0)])
        replay(s1, store=store)
        store.create_sleeve = lambda **kw: store.sleeve(kw["name"])
        book = store.journal_book(name, 10_000)
        test_replay.START = start0 + 2 * 3600 * 10**9
        s2 = tmp_path / "s2.jsonl.gz"
        _record_session(s2, _session_meta(book["cash"] + book["qty"] * book["entry_px"], params),
                        [(5, 0.0), (0, 0.6), (5, 0.0)], px=60_000.0 * 1.015)
        orders = replay(s2, store=store)
    finally:
        test_replay.START = start0
    liq = [o for o in orders if o["intent"] == "liquidation"]
    fills = store.fills(name, limit=200)
    entry = [o for o in orders if o["intent"] == "entry"][-1]
    ef = [f for f in fills if f["order_id"] == entry["order_id"]]
    lf = [f for f in fills if f["order_id"] in {o["order_id"] for o in liq}]
    want = sum(f["qty"] * f["price"] for f in ef) / 2 + sum(f["fee"] for f in ef) + sum(f["fee"] for f in lf)
    assert sum(f["fee"] for f in ef) > 0 and len(fills) == len(ef) + len(lf) + 4  # no restore row journaled
    assert _x_of(store.sleeve(name).status_reason) == pytest.approx(want, abs=0.005)


@pytest.mark.parametrize("exec1m", [False, True], ids=["bars-only", "1m"])
@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_p1_nothing_opens_after_a_liquidation_on_a_gap_bar_with_a_resting_entry(tmp_path, _full_margin, monkeypatch,
                                                                                exec1m, side):
    """Advisor 17:57: liquidated, it stays halted and places nothing. A 3x position on full margin opens at the 2 Oct
    close with a resting stop add (0.01) 1,000 beyond; the 3 Oct bar opens 50% against it (liquidated at the open);
    on 4 Oct the price runs 5% the other way, through the add's trigger. Nothing may fill after the liquidation."""
    monkeypatch.setitem(REGISTRY, "qa_e", (E, EConfig))
    put_rates(pd.date_range("2025-10-01 00:00", "2025-10-06 00:00", freq="8h", tz="UTC"))
    sec = pd.date_range("2025-10-01 00:00:01", "2025-10-05 23:59:46", freq="15s", tz="UTC")
    base = 60_000.0 + (np.arange(len(sec)) % 40) * 0.5
    px = np.where(sec >= pd.Timestamp("2025-10-03 00:00:01", tz="UTC"), base * (1 - 0.5 * side), base)
    px = np.where(sec >= pd.Timestamp("2025-10-04 00:00:01", tz="UTC"), base * (1 + 0.05 * side), px)
    tape = pd.Series(np.round(px, 1), index=sec)
    day = tape.resample("1D", closed="left", label="right").ohlc()
    day["volume"] = 1e9
    one = tape.resample("1min", closed="left", label="right").ohlc()
    one["volume"] = 1e9
    params = {**PERP, "at": ns("2025-10-02 00:00"), "side": side, "tag": "ge", "place_at": ns("2025-10-02 00:00"),
              "trigger": 61_000.0 if side > 0 else 59_000.0, "qty": 0.01}
    kw = dict(exec_prices=one, exec_minutes=1) if exec1m else {}
    r = backtest(binance_inst(), day, strategy="qa_e", params=params, minutes=1440, profile="aggressive",
                 half_spread=0.0, **kw)
    fills = sorted(r.journal.fills_, key=lambda f: f["ts"])
    liq = [f for f in fills if r.decisions.get(f["order_id"], {}).get("intent") == "liquidation"]
    assert liq and any(e["kind"] == "risk_halt" for e in r.risk_events)
    assert not [f for f in fills if f["ts"] > max(x["ts"] for x in liq)]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="P1-D20 MINOR, needs an Advisor call (6047b50): Y divides by the "
                   "last mark still holding the position up to the liquidation, which on a gap is the gap's own mark, after "
                   "the loss: a remainder of 0.2217 short liquidated reads '6,737.79, 340% of strategy equity' (equity "
                   "10,083.66 before the gap). 18:17 says 'MTM equity just before liquidation'; a share above 100% can't be "
                   "what the PM is meant to read")
def test_p1_y_is_never_above_100_percent_when_x_is_below_the_equity_before_the_gap(tmp_path):
    """The partial-reduce run above: X (6,737.79) is below the equity marked before the gap (10,083.66), so Y, its
    share of strategy equity, is at most 100%."""
    import dataclasses
    import test_replay
    from sleeve_fund.research.replay import replay
    for nm, p in list(risk.PROFILES.items()):
        risk.PROFILES[nm] = dataclasses.replace(p, max_position_pct=1.0)
    name, params = "ping-pong-test", {"rise": 0.01, "dip": 0.005, **PERP}
    store = Store(f"sqlite:///{tmp_path}/y.db")
    start0 = test_replay.START
    try:
        s1 = tmp_path / "s1.jsonl.gz"
        _record_session(s1, _session_meta(10_000, params), [(5, 0.0), (20, 0.015), (2, 0.0)])
        replay(s1, store=store)
        short = store.journal_book(name, 10_000)["qty"]
        red, px = round(abs(short) / 3, 8), 60_000.0 * 1.015
        store.record_order(name, order_id="QA-REDUCE", side="BUY", qty=red, intent="exit", reason="QA partial reduce")
        store.record_fill(name, side="BUY", qty=red, price=px, fee=round(red * px * 0.0005, 8), order_id="QA-REDUCE",
                          trade_id="QA-REDUCE", ts=pd.Timestamp(start0 + 3600 * 10**9, tz="UTC").to_pydatetime())
        store.create_sleeve = lambda **kw: store.sleeve(kw["name"])
        book = store.journal_book(name, 10_000)
        test_replay.START = start0 + 2 * 3600 * 10**9
        s2 = tmp_path / "s2.jsonl.gz"
        _record_session(s2, _session_meta(book["cash"] + book["qty"] * book["entry_px"], params),
                        [(5, 0.0), (0, 0.6), (5, 0.0)], px=px)
        replay(s2, store=store)
    finally:
        test_replay.START = start0
    text = store.sleeve(name).status_reason
    m = re.search(r"\(liquidated\): ([\d,]+\.\d\d), (\d+(?:\.\d+)?)% of strategy equity", text)
    assert m, text
    assert float(m.group(2)) <= 100.0, text
