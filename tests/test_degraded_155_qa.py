"""QA round on PR #155 (heads 73d3908, delta c966f13; quant-review/v2-p1/degraded-155.md): every finding as a test,
fixed per the Independent Quant Advisor's rulings (P1-D3 option (a), P1-D9 rule (c), R1 the first bar after a restart
from the stored minutes) and P1-D10. Open: P1-D11, a strict xfail owned by #146. Synthetic data only, no venue called.

"""
import collections
import json
import os
import shutil
import sys

from qa155_lib import *  # noqa: E402,F401,F403
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


@pytest.fixture(autouse=True)
def _reg(monkeypatch, tmp_path):
    stub_contract(monkeypatch)  # noqa: F405
    monkeypatch.setitem(REGISTRY, "qa_t", (T, TConfig))
    monkeypatch.setattr(funding, "DEFAULT_ROOT", tmp_path / "funding")
    monkeypatch.setattr(funding, "fetch", lambda *a, **k: (_ for _ in ()).throw(OSError("QA: no venue calls")))
    SEEN.clear()


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
        else:  # as paper.node builds an own-feed node: its first bar from the store's minutes (R1)
            s.attach_minutes(stored_minutes("BINANCE", "BTC/USDT", store=hs))
        eng.add_strategy(s)
        eng.run()
    finally:
        rt.now = utcnow
        eng.dispose()
    return st, told


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
