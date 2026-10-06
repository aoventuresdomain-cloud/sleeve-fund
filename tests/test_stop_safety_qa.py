"""Legacy perp stop safety (Platform Engineer 2's PR): strict xfails written by QA BEFORE the build, from the
agreed scope, tightened to the Advisor's rulings of 6 Oct 16:45. The engineer makes each one pass (and removes its
mark) before handing over.

Sources (every trading-behaviour expectation below cites one of these):
- [B101] eng-board.md "6 Oct ~15:40 P2-1 scope" entry, incl. Advisor 15:31, Advisor 15:47 correction and the restart
  edge "base.py:835-841 (stop not restorable): block entries, place an immediate safety stop at half the
  liquidation distance, alert + incident, never flatten".
- [B97] eng-board.md "6 Oct 15:48 PE2 stop-minimum": "Shipped stopless 2x perp configs left as is (refused at start
  with reason)".
- [B63] eng-board.md "16:46 PE2 stop-safety patch QA": "S2 incident = error event kind=incident + alerts inbox" (HoE).
- [LS] quant-review/v2-p1/legacy-stops.md, incl. "Advisor 6 Oct 15:30: ... rsi_pullback counts as stopless (1x
  cap, full margin in open risk) until its trail is a resting stop or it is ported. Only dip_buy is exempt".
- [P2] v2/phase2-spec.md P2-2 table: "Open risk (sum of loss to stop) | <= 5% of book".
- [R-S1..R-S10] quant-review/v2-p1/stop-safety.md "Advisor rulings (6 Oct 16:45) and regrade", items S1-S10.

ASSUMED INTERFACES (adapt the names, never the assertions):
- sleeve_fund.strategies.check_perp_stop(strategy, params, risk_profile: str) -> None, raising ValueError whose text
  contains sleeve_fund.strategies.STOPLESS_PERP_REFUSAL; the same shape as check_perp_sizing.
- sleeve_fund.risk.OPEN_RISK_LIMIT = 0.05
- sleeve_fund.risk.position_open_risk(qty, mark, stop=None, daily_atr_pct=None) -> float: one position's open risk in
  quote currency; qty signed (+ long, - short), stop a price level or None (stopless), daily_atr_pct = daily Wilder
  ATR / price. Raises ValueError when a stopless position's risk can't be measured (R-S1: the entry is refused).
- sleeve_fund.risk.open_risk_breach(open_risk, book_equity) -> str | None: None within the limit, else the reason.
- A test that needs paper's REAL daily-ATR source (which has no history in tests, so reads as unknown) carries the
  marker `real_daily_atr` (the patch's name for opting out of its 2% test fixture).
- The restart incident (HoE-confirmed definition) is an events row of kind "incident" at level "error" AND that event
  listed, unacknowledged, in the alerts inbox (Store.alerts) [B63]; the safety-stop alert is a warning/error
  event (Store.alerts) whose message mentions the "safety stop".
- The backtest label of R-S8 appears in some text the BacktestResult carries (searched across all its fields).
The supervisor, dashboard, replay and backtest tests use only existing entry points.

Two tests here are GUARDS, not xfails: they pass on main and must keep passing (R-S7: start is never gated by open
risk; R-S9: rsi_pullback's % stop fills intrabar). They are marked as such.
"""

import glob
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

ITEM = "Legacy perp stop safety (Platform 2)"


def xf(done: str):
    return pytest.mark.xfail(strict=True, reason=f"{ITEM} Done-when: {done}")


PERP = {"market": "perp", "allow_short": True}
LEVERAGE = {"conservative": 1, "balanced": 2, "aggressive": 3}
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def store(tmp_path):
    # As test_sleeve_runtime: Postgres when TEST_DATABASE_URL is set, else SQLite.
    from sleeve_fund.store import Store

    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        from sleeve_fund.store import make_engine, metadata

        engine = make_engine(url)
        metadata.drop_all(engine)
        return Store(engine=engine)
    return Store(f"sqlite:///{tmp_path}/t.db")


# --- 1x cap: a stopless legacy model on a perp is refused above 1x ------------------------------------------------

# [LS] Only dip_buy always places a stop; rsi_bands, ping_pong, rsi_cross, trend_filter (side-only) and buy_and_hold
# have none by default; rsi_pullback's close-checked trail counts as stopless (Advisor 15:30).
STOPLESS = [("rsi_bands", PERP), ("ping_pong", PERP), ("rsi_cross", PERP), ("trend_filter", {"market": "perp"}),
            ("buy_and_hold", {"market": "perp"}), ("rsi_pullback", {"market": "perp"})]


@pytest.mark.parametrize("profile", ["balanced", "aggressive"])  # 2x, 3x
@pytest.mark.parametrize("strategy, params", STOPLESS)
def test_a_stopless_legacy_model_above_1x_on_a_perp_is_refused(strategy, params, profile):
    from sleeve_fund.strategies import check_perp_stop; STOPLESS_PERP_REFUSAL = 'places no stop'

    with pytest.raises(ValueError, match=STOPLESS_PERP_REFUSAL):
        check_perp_stop(strategy, params, profile)


ALLOWED = ([(s, p, "conservative") for s, p in STOPLESS]  # 1x
           + [("dip_buy", PERP, "balanced"), ("dip_buy", PERP, "aggressive")]  # its default stop_atr 2 / 20 bars
           + [("rsi_bands", {**PERP, "stop_loss": 0.03}, "balanced"),  # a stop set: placed on every entry [LS]
              ("rsi_bands", {**PERP, "stop_atr": 2.0}, "aggressive"),
              ("ping_pong", {**PERP, "stop_swing_bars": 10}, "balanced"),
              # R-S9: rsi_pullback with a % stop counts as stopped, as that stop rests intrabar (guard below)
              ("rsi_pullback", {"market": "perp", "stop_loss": 0.02}, "balanced")]
           + [("rsi_bands", {}, "aggressive"), ("buy_and_hold", {}, "balanced")])  # spot: no leverage


@pytest.mark.parametrize("strategy, params, profile", ALLOWED)
def test_1x_dip_buy_a_set_stop_and_spot_are_allowed(strategy, params, profile):
    from sleeve_fund.strategies import check_perp_stop

    check_perp_stop(strategy, params, profile)


def test_guard_rsi_pullbacks_pct_stop_fills_intrabar_at_its_level(instrument):
    """GUARD (passes on main) for R-S9: rsi_pullback's % stop is the base class's resting stop, so it fills at its
    trigger inside the bar, not at the close: that is why it may count as stopped."""
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.research.runner import run_backtest

    p = synthetic_ohlcv(days=900, seed=3)
    res = run_backtest("rsi_pullback", p, instrument, {"stop_loss": 0.02, "rsi_entry": 60, "vol_mult": 0.5,
                                                         "ema_period": 20})
    stops = [o for o in res.fills.index if res.decisions[o]["intent"] == "stop_loss"]
    assert stops
    for o in stops:
        trigger = res.decisions[o]["signal"]["trigger"]
        ts = res.fills.loc[o, "ts_last"]
        bar = p[p.index >= ts].iloc[0]
        px = float(res.fills.loc[o, "avg_px"])
        # at its level (or the open on a gap), never deferred to a close below it
        assert px == pytest.approx(min(trigger, bar.open), rel=0.002) or px >= bar.close, (px, trigger, bar.close)


def _stored(store, name, strategy="rsi_bands", params=None, profile="balanced", state="running"):
    return store.create_sleeve(name=name, strategy=strategy, instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                               starting_balance=10_000, params=PERP if params is None else params,
                               risk_profile=profile, desired_state=state)


def _popen(monkeypatch):
    from sleeve_fund import supervisor
    from test_supervisor import FakePopen

    started = []
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda args, **k: started.append(args[-1]) or FakePopen())
    return started


def test_the_supervisor_refuses_a_stopless_2x_perp_and_says_why(store, monkeypatch):
    from sleeve_fund import supervisor

    _stored(store, "rb-2x")
    _stored(store, "rb-1x", profile="conservative")
    _stored(store, "dip-2x", strategy="dip_buy")
    started = _popen(monkeypatch)
    supervisor.Supervisor(store).step()
    assert sorted(started) == ["dip-2x", "rb-1x"]
    s = store.sleeve("rb-2x")
    assert s.desired_state == "stopped" and "stop" in (s.status_reason or "").lower(), s.status_reason
    assert any(e["kind"] == "start_refused" for e in store.events("rb-2x", limit=20))


def test_a_stopless_2x_perp_holding_a_position_is_started_not_left_unwatched(store, monkeypatch):
    from sleeve_fund import supervisor

    _stored(store, "rb-flat")
    _stored(store, "rb-held")
    store.record_fill("rb-held", side="BUY", qty=0.01, price=60_000.0, fee=0.3, order_id="o1", trade_id="t1")
    started = _popen(monkeypatch)
    supervisor.Supervisor(store).step()
    assert started == ["rb-held"], started
    assert store.sleeve("rb-held").desired_state == "running"
    assert store.sleeve("rb-flat").desired_state == "stopped"


def test_guard_the_open_risk_check_never_gates_a_start(store, monkeypatch):
    """GUARD (passes on main) for R-S7: the open-risk check gates entries and adds only, never a start. Another
    strategy's stopless position puts the book far over 5%; a 1x strategy still starts."""
    from sleeve_fund import supervisor

    _stored(store, "big", strategy="ping_pong", profile="conservative", state="stopped")
    store.record_fill("big", side="BUY", qty=0.16, price=60_000.0, fee=4.8, order_id="o1", trade_id="t1")
    store.record_equity("big", equity=10_000, cash=10_000 - 0.16 * 60_000, qty=0.16, price=60_000, benchmark=10_000)
    _stored(store, "new", strategy="ping_pong", profile="conservative")
    started = _popen(monkeypatch)
    supervisor.Supervisor(store).step()
    # Open risk never gates a start: "new" starts and runs. Since U35 (CHOKE) the stopped holder "big" may also be
    # started for its exits only (PE2 22:27, QA option (a)); if it is, it must be paused "exits only", never running.
    assert "new" in started and set(started) <= {"new", "big"}, started
    assert store.sleeve("new").desired_state == "running"
    if "big" in started:
        big = store.sleeve("big")
        assert big.status == "paused" and "exits only" in (big.status_reason or ""), (big.status, big.status_reason)


def test_the_shipped_stopless_2x_perp_configs_seed_but_are_refused_at_start(store, monkeypatch):
    from sleeve_fund import supervisor
    from sleeve_fund.supervisor import seed

    paths = sorted(glob.glob(str(REPO / "configs" / "sleeves" / "*.toml")))
    added = seed(store, paths)  # must not raise: the files are left as they are
    assert len(added) == len(paths)
    perp_2x = {s.name for s in store.sleeves()
               if s.params.get("market") == "perp" and LEVERAGE[s.risk_profile] > 1
               and not any(s.params.get(k) for k in ("stop_loss", "stop_atr", "stop_swing_bars"))}
    assert perp_2x == {"ping-pong-ls-binance", "ping-pong-ls-test", "rsi-bands-15m-ls-test", "rsi-bands-ls-binance",
                       "rsi-bands-ls-test"}  # the fixture itself: what is shipped today
    for s in store.sleeves():
        store.set_desired_state(s.name, "running")
    started = _popen(monkeypatch)
    supervisor.Supervisor(store).step()
    assert not perp_2x & set(started), sorted(perp_2x & set(started))
    assert set(started) == {s.name for s in store.sleeves()} - perp_2x  # the spot ones still start
    for name in perp_2x:
        s = store.sleeve(name)
        assert s.desired_state == "stopped" and s.status_reason, (name, s.status_reason)


def test_the_dashboard_start_refuses_a_stopless_2x_perp(client):  # noqa: F811
    from test_dashboard import AUTH, SAME

    c, store = client
    _stored(store, "rb-2x", state="stopped")
    r = c.post("/sleeves/rb-2x/command", data={"command": "start", "reason": "try it"}, auth=AUTH, headers=SAME,
               follow_redirects=False)
    assert "command_error" in r.headers["location"], r.headers["location"]
    assert store.sleeve("rb-2x").desired_state == "stopped"


from test_dashboard import client  # noqa: E402, F401 - the dashboard fixture


# --- interim open-risk check: 5% limit (Advisor 15:47 correction) -------------------------------------------------

@pytest.mark.parametrize("qty", [0.1, -0.1])
@pytest.mark.parametrize("atr_pct, share", [(0.02, 0.10), (0.05, 0.15), (0.10 / 3, 0.10), (0.0, 0.10)])
def test_a_stopless_position_counts_notional_times_the_larger_of_10pct_and_3_atrs(qty, atr_pct, share):
    from sleeve_fund.open_risk import position_risk; position_open_risk = lambda q, m, stop=None, daily_atr_pct=None: position_risk(q, m, stop, daily_atr_pct)

    # 0.1 at 60,000 = 6,000 of notional; an ATR of 0 leaves the 10% floor
    assert position_open_risk(qty, 60_000.0, stop=None, daily_atr_pct=atr_pct) == pytest.approx(6_000 * share)


@pytest.mark.parametrize("qty, mark, stop, risk", [
    (0.1, 60_000.0, 57_000.0, 300.0),  # long: 3,000 a unit to the stop
    (-0.1, 60_000.0, 63_000.0, 300.0),  # short: the mirror image
    (0.1, 62_000.0, 57_000.0, 500.0),  # the mark rose since a 60,000 entry: counted from the mark, not the entry
    (-0.1, 58_000.0, 63_000.0, 500.0),
])
def test_a_stopped_position_counts_from_the_current_mark_to_its_stop(qty, mark, stop, risk):
    from sleeve_fund.open_risk import position_risk; position_open_risk = lambda q, m, stop=None, daily_atr_pct=None: position_risk(q, m, stop, daily_atr_pct)

    assert position_open_risk(qty, mark, stop=stop, daily_atr_pct=0.02) == pytest.approx(risk)


@pytest.mark.parametrize("atr_pct", [float("nan"), None])
def test_an_unknown_or_nan_atr_refuses_to_measure_a_stopless_position(atr_pct):
    from sleeve_fund.open_risk import position_risk; position_open_risk = lambda q, m, stop=None, daily_atr_pct=None: position_risk(q, m, stop, daily_atr_pct)

    with pytest.raises(ValueError):
        position_open_risk(0.1, 60_000.0, stop=None, daily_atr_pct=atr_pct)


def test_the_open_risk_limit_is_5pct_of_book_inclusive():
    from sleeve_fund.open_risk import LIMIT as OPEN_RISK_LIMIT, check_entry; open_risk_breach = lambda r, b: check_entry(b, r, 0.0)

    assert OPEN_RISK_LIMIT == 0.05
    assert open_risk_breach(1_000.0, 20_000.0) is None  # exactly 5%
    assert open_risk_breach(999.99, 20_000.0) is None
    why = open_risk_breach(1_000.01, 20_000.0)
    assert why and "open risk" in why.lower()
    # Worked from a position: a stopless 0.1 at 60,000 with a 2% ATR counts 600; on a 12,000 book that is 5.0%.
    from sleeve_fund.open_risk import position_risk; position_open_risk = lambda q, m, stop=None, daily_atr_pct=None: position_risk(q, m, stop, daily_atr_pct)

    assert open_risk_breach(position_open_risk(0.1, 60_000.0, None, 0.02), 12_000.0) is None
    assert open_risk_breach(position_open_risk(0.1, 60_000.0, None, 0.02), 11_999.0) is not None


@pytest.mark.parametrize("qty, mark, stop", [(0.1, 56_000.0, 57_000.0), (-0.1, 64_000.0, 63_000.0)])
def test_a_position_gapped_through_its_stop_counts_at_the_stopless_measure(qty, mark, stop):
    from sleeve_fund.open_risk import position_risk; position_open_risk = lambda q, m, stop=None, daily_atr_pct=None: position_risk(q, m, stop, daily_atr_pct)

    got = position_open_risk(qty, mark, stop=stop, daily_atr_pct=0.02)
    assert got == pytest.approx(abs(qty) * mark * 0.10)  # 560 / 640: max(10%, 3 x 2%) of the notional at the mark


# --- paper gates (refuse and log, never trim), a single backtest only reports -----------------------------------

def _meta(balance, params, strategy, profile, name="edge"):
    return {"balances": [f"{balance:.2f} USD"],
            "sleeve": {"name": name, "strategy": strategy, "instrument": "BTC/USD",
                       "bar_spec": "1-MINUTE-LAST-INTERNAL", "starting_balance": 10_000, "risk_profile": profile,
                       "params": params, "maker_fee": "0.0002", "taker_fee": "0.0005", "tick_seconds": 30}}


PING = {"rise": 0.01, "dip": 0.005, **PERP}
PING_LEGS = [(5, 0.0), (20, 0.015), (20, -0.012)]


def test_paper_refuses_and_logs_an_entry_past_the_open_risk_limit_never_trims(tmp_path, full_margin):
    from sleeve_fund.research.replay import replay
    from sleeve_fund.store import Store
    from test_long_short import _record

    # ping_pong buys at once. At 1x (allowed stopless) with the whole equity as margin it would take ~10,000 of
    # notional: >= 10% x 10,000 = 1,000 of open risk on a 10,000 book, double the limit.
    store = Store.in_memory()
    path = tmp_path / "gate.jsonl.gz"
    _record(path, _meta(10_000, PING, "ping_pong", "conservative"), PING_LEGS)
    orders, fills = replay(path, with_fills=True, store=store)
    assert [o for o in orders if o["intent"] == "entry"] == []  # refused outright: no smaller order either
    assert any("open risk" in e["message"].lower() for e in store.events("edge", limit=500))


@pytest.mark.real_daily_atr  # paper's own ATR source, with no history in tests: unknown
def test_paper_refuses_a_stopless_entry_when_the_daily_atr_is_unknown(tmp_path):
    from sleeve_fund.research.replay import replay
    from sleeve_fund.store import Store
    from test_long_short import _record

    store = Store.in_memory()
    path = tmp_path / "noatr.jsonl.gz"
    _record(path, _meta(10_000, PING, "ping_pong", "conservative"), PING_LEGS)  # 2,000 notional: 2% if measurable
    orders, _ = replay(path, with_fills=True, store=store)
    assert [o for o in orders if o["intent"] == "entry"] == []
    assert any(e["level"] in ("warning", "error", "info") and "atr" in e["message"].lower()
               for e in store.events("edge", limit=500))


def test_paper_counts_a_gapped_through_position_at_the_stopless_measure_and_alerts(tmp_path):
    from sleeve_fund.research.replay import replay
    from sleeve_fund.store import Store
    from test_long_short import _record

    store = Store.in_memory()
    store.create_sleeve(name="other", strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params={**PERP, "stop_loss": 0.05}, risk_profile="conservative")
    store.record_order("other", order_id="o-entry", side="BUY", qty=0.15, intent="entry", reason="held",
                       signal={"close": 60_000.0, "stop_frac": 0.05})  # its stop: 57,000
    store.record_fill("other", side="BUY", qty=0.15, price=60_000.0, fee=4.5, order_id="o-entry", trade_id="t1")
    store.record_equity("other", equity=10_000, cash=10_000 - 0.15 * 60_000 + 0.15 * 4_000 * 0, qty=0.15,
                        price=56_000.0, benchmark=10_000)  # marked at 56,000: through its stop, exit not filled
    # Book 20,000: limit 1,000. "other" counts 0.15 x 56,000 x 10% = 840 (not 0); this entry ~2,000 x 10% = 200.
    path = tmp_path / "gap.jsonl.gz"
    _record(path, _meta(10_000, PING, "ping_pong", "conservative"), PING_LEGS)
    orders, _ = replay(path, with_fills=True, store=store)
    assert [o for o in orders if o["intent"] == "entry"] == [], "entry let through: the gapped position counted 0"
    alerts = [a for a in store.alerts() if a["kind"] != "entry_refused_open_risk"]
    assert any(a["sleeve"] == "other" or "other" in a["message"] for a in alerts), [a["message"] for a in alerts]


def test_a_single_backtest_reports_open_risk_and_is_not_gated(prices, instrument, full_margin):
    from sleeve_fund.research.runner import run_backtest

    res = run_backtest("ping_pong", prices.iloc[:30], instrument, PING, risk_profile="conservative")
    first = res.fills.sort_values("ts_last").iloc[0]
    assert float(first.filled_qty) * float(first.avg_px) > 5_000  # same entry as today: not gated (~9,900 on main)
    assert res.open_risk_max > 0.05  # reported: >= 10% of equity at that entry
    assert res.open_risk_binds >= 1  # and counted as an entry paper would refuse


LABEL = "would be refused on paper (stopless above 1x)"


def test_a_stopless_backtest_above_1x_runs_and_is_labelled(prices, instrument):
    from sleeve_fund.research.runner import run_backtest

    def text(res):
        return " ".join(str(v) for v in vars(res).values())

    above = run_backtest("ping_pong", prices.iloc[:60], instrument, PING, risk_profile="balanced")
    assert len(above.fills) and LABEL in text(above)
    assert LABEL not in text(run_backtest("ping_pong", prices.iloc[:60], instrument, PING,
                                          risk_profile="conservative"))
    assert LABEL not in text(run_backtest("ping_pong", prices.iloc[:60], instrument, {**PING, "stop_loss": 0.03},
                                          risk_profile="balanced"))


# --- restart edge: stop not restorable (base.py ~835-841) --------------------------------------------------------

ENTRY = 60_000.0
QTY = 0.005  # 300 of notional: small, so the daily-loss pause and drawdown halt stay out of the way
MM = 0.005  # the perp's maintenance margin (markets.LOW_FEE_PERP)


def _restart(tmp_path, profile, side, legs, stop_params=None, start_px=ENTRY):
    """A paper restart holding a position opened at ENTRY whose entry journaled no stop. With an ATR stop (the
    default here) it is base.py's 'The open position has no stop yet' case for the first atr_bars minutes; with
    stop_params={} the model is stopless (R-S6)."""
    from sleeve_fund.research.replay import replay
    from sleeve_fund.store import Store, replay_book
    from test_long_short import _fill, _record
    from test_replay import START

    store = Store.in_memory()
    create = store.create_sleeve
    opened = datetime.fromtimestamp(START / 1e9 - 3600, tz=timezone.utc)
    word = "BUY" if side > 0 else "SELL"

    def create_holding(**kw):
        s = create(**kw)
        store.record_order(kw["name"], order_id="carried", side=word, qty=QTY, intent="entry", reason="carried",
                           signal={"close": ENTRY}, ts=opened)  # no stop_frac: not restorable
        store.record_fill(kw["name"], side=word, qty=QTY, price=ENTRY, fee=0.15, order_id="carried",
                          trade_id="carried", ts=opened)
        return s

    store.create_sleeve = create_holding
    params = {**PERP, **({"stop_atr": 3.0} if stop_params is None else stop_params)}
    book = replay_book([_fill(word, QTY, ENTRY, 0.15)], 10_000.0)
    path = tmp_path / "restart.jsonl.gz"
    _record(path, _meta(book["cash"] + book["qty"] * book["entry_px"], params, "rsi_bands", profile), legs, px=start_px)
    orders, fills = replay(path, with_fills=True, store=store)
    return store, [o for o in orders if o["order_id"] != "carried"], [f for f in fills if f["order_id"] != "carried"]


def _safety_level(side, lev, mark=ENTRY):
    """R-S3/R-S4: half the REMAINING distance from the mark to the isolated liquidation price (a 1x long has none:
    half way to zero)."""
    liq = 0.0 if (side > 0 and lev == 1) else ENTRY * (1 - side / lev) / (1 - side * MM)
    level = mark + 0.5 * (liq - mark)
    assert min(mark, liq) < level < max(mark, liq)  # always between the mark and liquidation
    return level


def _closes(orders, fills, side):
    closes = [o for o in orders if o["side"] == ("SELL" if side > 0 else "BUY")]
    assert closes, "no stop acted: the position rode the move unprotected"
    assert "stop" in closes[0]["intent"] and "liquidation" not in closes[0]["intent"], closes[0]
    (exit_fill,) = [f for f in fills if f["order_id"] == closes[0]["order_id"]]
    return closes[0], exit_fill


@pytest.mark.parametrize("profile, side, move", [("balanced", 1, -0.30), ("aggressive", 1, -0.30),
                                                 ("balanced", -1, 0.30), ("aggressive", -1, 0.30),
                                                 ("conservative", -1, 0.60), ("conservative", 1, -0.60)])
def test_a_restart_without_a_restorable_stop_places_a_safety_stop_half_way_to_liquidation(tmp_path, profile, side,
                                                                                           move):
    # The move runs over 5 minutes, inside the 15 bars the ATR stop needs: only a safety stop can act.
    store, orders, fills = _restart(tmp_path, profile, side, [(3, 0.0), (5, move), (4, 0.0)])
    _, exit_fill = _closes(orders, fills, side)
    level = _safety_level(side, LEVERAGE[profile])
    assert exit_fill["price"] == pytest.approx(level, rel=0.01), (exit_fill["price"], level)
    assert not [o for o in orders if o["intent"] == "entry"]


def test_a_long_entered_at_60000_and_restarted_at_42000_gets_its_safety_stop_from_the_mark(tmp_path):
    store, orders, fills = _restart(tmp_path, "balanced", 1, [(3, 0.0), (5, -0.19), (4, 0.0)], start_px=42_000.0)
    _, exit_fill = _closes(orders, fills, 1)
    level = _safety_level(1, 2, mark=42_000.0)  # 42,000 + 0.5 x (30,150.75 - 42,000) = 36,075.4
    assert exit_fill["price"] < 40_000, "closed at the restart price: the stop was set from the entry"
    assert exit_fill["price"] == pytest.approx(level, rel=0.01), (exit_fill["price"], level)


def _assert_incident(store, sleeve):
    """[B63, HoE-confirmed] An incident is BOTH an error-severity event with kind "incident" AND an entry in the alerts
    inbox: the same event listed, unacknowledged (visible), by Store.alerts(), and counted as open."""
    events = [e for e in store.events(sleeve, limit=500) if e["kind"] == "incident" and e["level"] == "error"]
    assert events, [(e["kind"], e["level"]) for e in store.events(sleeve, limit=500)]
    inbox = [a for a in store.alerts(limit=500) if a["sleeve"] == sleeve and a["kind"] == "incident"]
    assert inbox and all(a["acked_at"] is None for a in inbox), inbox  # present and visible
    assert {a["id"] for a in inbox} & {e["id"] for e in events}
    assert store.open_alert_count() >= 1


@pytest.mark.parametrize("side", [1, -1])
def test_a_restart_without_a_restorable_stop_alerts_and_never_flattens(tmp_path, side):
    store, orders, fills = _restart(tmp_path, "balanced", side, [(12, 0.0)])  # flat: nothing reaches the stop
    assert fills == [], fills  # nothing traded: no flatten, no entry
    assert not [o for o in orders if o["intent"] == "entry"]
    assert store.journal_book("edge", 10_000)["qty"] == pytest.approx(side * QTY)
    alerts = [a for a in store.alerts() if a["sleeve"] == "edge"]
    assert any("safety stop" in a["message"].lower() for a in alerts), [a["message"] for a in alerts]
    _assert_incident(store, "edge")


def test_a_gap_through_the_safety_stop_closes_on_the_gap(tmp_path):
    store, orders, fills = _restart(tmp_path, "balanced", 1, [(3, 0.0), (0, -0.35), (4, 0.0)])
    _, exit_fill = _closes(orders, fills, 1)
    assert ENTRY * 0.64 <= exit_fill["price"] <= _safety_level(1, 2), exit_fill["price"]
    assert store.journal_book("edge", 10_000)["qty"] == pytest.approx(0.0)


@pytest.mark.parametrize("side, move", [(1, -0.30), (-1, 0.30)])
def test_a_stopless_model_above_1x_restarted_holding_gets_a_safety_stop_from_the_mark(tmp_path, side, move):
    store, orders, fills = _restart(tmp_path, "balanced", side, [(3, 0.0), (5, move), (4, 0.0)], stop_params={})
    _, exit_fill = _closes(orders, fills, side)
    assert exit_fill["price"] == pytest.approx(_safety_level(side, 2), rel=0.01)
    assert not [o for o in orders if o["intent"] == "entry"]


def _spot_restart(store, instrument, prices):
    """test_sleeve_runtime's restart, on spot (aggressive: a 50% position, so a 60% fall stays inside the drawdown
    halt): a buy-and-hold with an ATR stop whose entry journaled no stop, restarted with too few bars for the ATR."""
    from sqlalchemy import update

    from sleeve_fund.paper.runtime import SleeveRuntime
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.store import orders_t

    store.create_sleeve(name="s1", strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=10_000, risk_profile="aggressive")
    exits = {"stop_atr": 10.0}
    run_backtest("buy_and_hold", prices.iloc[:25], instrument, exits, runtime=SleeveRuntime(store, "s1",
                                                                                           tick_seconds=6 * 3600))
    (entry,) = [o for o in store.orders("s1") if o["intent"] == "entry"]
    sig = {k: v for k, v in entry["signal"].items() if k not in ("stop_frac", "stop_basis")}
    with store.engine.begin() as c:
        c.execute(update(orders_t).where(orders_t.c.order_id == entry["order_id"]).values(signal=sig))
    mark = float(prices.iloc[24]["close"])
    run_backtest("buy_and_hold", prices.iloc[25:], instrument, exits, runtime=SleeveRuntime(store, "s1",
                                                                                           tick_seconds=6 * 3600))
    return mark


def test_a_spot_restart_without_a_restorable_stop_gets_a_safety_stop_alert_and_incident(store, instrument):
    from sleeve_fund.data import synthetic_ohlcv

    _spot_restart(store, instrument, synthetic_ohlcv(days=40, seed=3))
    assert [f["side"] for f in store.fills("s1")] == ["BUY"]  # never flattened for it
    alerts = [a for a in store.alerts() if a["sleeve"] == "s1"]
    assert any("safety stop" in a["message"].lower() for a in alerts), [a["message"] for a in alerts]
    _assert_incident(store, "s1")


def test_a_spot_safety_stop_closes_a_fall_through_half_way_to_zero(store, instrument):
    from sleeve_fund.data import synthetic_ohlcv

    prices = synthetic_ohlcv(days=40, seed=3)
    cols = ["open", "high", "low", "close"]
    prices.loc[prices.index[28]:, cols] = prices.loc[prices.index[28]:, cols] * 0.4  # gaps down 60% on bar 28
    mark = _spot_restart(store, instrument, prices)
    sells = [f for f in store.fills("s1") if f["side"] == "SELL"]
    assert sells, "the fall was ridden: no safety stop"
    assert sells[-1]["price"] <= 0.5 * mark * 1.01  # at half way to zero, or the gap's open below it
    orders = {o["order_id"]: o for o in store.orders("s1")}
    assert "stop" in orders[sells[-1]["order_id"]]["intent"]
