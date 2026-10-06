"""QA round on PR #155 (head 73d3908; quant-review/v2-p1/degraded-155.md): every finding as a test, the open ones as
fixed per the Independent Quant Advisor's rulings (P1-D3 option (a), P1-D9 rule (c)). Synthetic data only, no venue called.

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
    if gap.floor("D") + pd.Timedelta(seconds=1) == gap:  # the bar opens gapped: the fill is held to its open, exactly
        assert list(paid) == owed
        return
    # Rule (c), the Independent Quant Advisor: a stop touched at an unknown time inside the bar takes the worse
    # outcome. The 1-minute backtest stays the reference, and the bars-only result differs from it only for the worse.
    one = tape.resample("1min", closed="left", label="right").ohlc()
    one["volume"] = 1e9
    ref = backtest(binance_inst(), day, strategy="qa_t", params=params, minutes=1440, profile="balanced",
                   half_spread=0.0, exec_prices=one, exec_minutes=1)
    ref_paid = {pd.Timestamp(x["ts"]): x["amount"] for x in ref.journal.funding_}
    assert list(ref_paid) == owed  # the reference pays what paper pays
    assert set(ref_paid) <= set(paid) and all(paid[t] == pytest.approx(ref_paid[t]) for t in ref_paid)
    assert [t for t in paid if t not in ref_paid] and all(paid[t] < 0 for t in paid if t not in ref_paid)
