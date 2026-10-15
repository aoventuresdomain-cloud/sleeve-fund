"""A9: the minimum-size ramp and its track-length gate (pre-live set). Switched OFF by default (SLEEVE_MIN_SIZE_RAMP=1
turns it on). ON, a Strategy inside the fund stays at the venue's minimum order until it has MIN_TRADES closed trades
AND its minimum track record length (R5, mintrl.may_scale) on its own daily net returns since its first fill in the
current book. Only an entry or add is capped; exits, stops and reductions never are. Anything unreadable fails closed.

Advisor A9 ruling (15 Oct 09:13 UK): the cap is the venue minimum (quantity and notional); the cost band (c) counts as
not passed until it has a reader; the track counts only fills in the fund's own mode, weighted by the un-ramped size,
no-trade days 0; re-evaluated once a UTC day, a drop back never cuts an open position.

Red first on 14ed240 (no ramp: sleeve_fund.ramp missing, entries inside the fund sized at full size with it on)."""

from __future__ import annotations

import inspect
import os
import re
from datetime import UTC, datetime, timedelta
from decimal import Decimal as D
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sleeve_fund import ramp
from sleeve_fund.store import Store
from sleeve_fund.strategies.base import LongFlatStrategy

T0 = datetime(2026, 9, 1, 0, 0, tzinfo=UTC)


def _fills(*legs):
    return [{"side": s, "qty": q, "ts": T0 + timedelta(minutes=i), "order_id": f"o{i}"} for i, (s, q) in enumerate(legs)]


def _track(days, mean=0.004, sd=0.01, seed=5):
    r = np.random.default_rng(seed).normal(mean, sd, days)
    return pd.Series(r, index=pd.date_range("2026-01-01", periods=days, freq="D", tz="UTC"))


# ---- the switch ------------------------------------------------------------------------------------------------------
def test_the_ramp_is_off_unless_switched_on(monkeypatch):
    monkeypatch.delenv(ramp.ENV, raising=False)
    assert not ramp.enabled()
    for v in ("0", "true", "yes", ""):
        monkeypatch.setenv(ramp.ENV, v)
        assert not ramp.enabled()
    monkeypatch.setenv(ramp.ENV, "1")
    assert ramp.enabled()


# ---- closed trades ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("legs, n", [
    ([], 0),
    ([("BUY", 1)], 0),  # still open
    ([("BUY", 1), ("SELL", 1)], 1),
    ([("BUY", 2), ("SELL", 1), ("SELL", 1)], 1),  # a partial reduction is not a trade closed
    ([("BUY", 1), ("SELL", 2), ("BUY", 1)], 2),  # a flip closes one and opens the next
    ([("SELL", 1), ("BUY", 1), ("BUY", 1), ("BUY", 1), ("SELL", 2)], 2),  # shorts count too; adds don't
])
def test_a_trade_is_counted_once_the_position_returns_to_flat_or_flips(legs, n):
    assert ramp.closed_trades(_fills(*legs)) == n


# ---- the minimum and the cap -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("min_qty, lot, want", [(D("0.001"), D("0.001"), D("0.001")),
                                                (D("0.0015"), D("0.001"), D("0.002")),
                                                (D(0), D("0.01"), D("0.01"))])
def test_the_minimum_size_is_the_venues_smallest_order_rounded_up_to_the_lot(min_qty, lot, want):
    assert ramp.minimum_qty(min_qty, lot) == want


def test_the_minimum_size_also_clears_the_venues_smallest_order_value_with_headroom():
    """Found in the e2e cell: one lot at 60,000 is 60 USDT, under a 100 USDT minimum, and the venue denies it. 100 x 1.01
    / 60,000 = 0.00168 rounds up to 0.002."""
    assert ramp.minimum_qty(D("0.001"), D("0.001"), D(100), D(60_000)) == D("0.002")
    assert ramp.minimum_qty(D("0.001"), D("0.001"), D(100), D(200_000)) == D("0.001")  # the quantity binds


@pytest.mark.parametrize("price, want", [
    (D(50_000), D("0.003")),  # 100 / 50,000 = 0.002 exactly: the 1% headroom (0.00202) takes it to the next lot
    (D(45_000), D("0.003")),  # 101 / 45,000 = 0.002244: rounded UP to the lot, never to the nearest (0.002)
    (D(25_250), D("0.004")),  # 101 / 25,250 = 0.004 exactly: headroom and lot agree, no extra lot
])
def test_the_minimum_value_path_takes_the_headroom_before_rounding_up_to_the_lot(price, want):
    """CR F-A9-1: an instrument whose minimum value, not its minimum quantity, sets the size; at these prices dropping
    the headroom or rounding to the nearest lot each changes the lot count."""
    assert ramp.minimum_qty(D("0.001"), D("0.001"), D(100), price) == want


@pytest.mark.parametrize("held, side, want", [
    (D(0), 1, D("0.002")),  # flat: the whole minimum
    (D("0.001"), 1, D(0)),  # an add under the minimum would be under the venue's smallest order too: none
    (D("0.002"), 1, D(0)),  # at the minimum: nothing more
    (D("0.005"), 1, D(0)),  # above it (sized before the ramp was on): never more, never a forced cut
    (D("-0.003"), 1, D("0.002")),  # a flip from short: the close is an exit, the open part takes the minimum
    (D("-0.001"), -1, D(0)),  # a short's add
    (D("0.001"), -1, D("0.002")),  # a flip from long to short
])
def test_the_cap_is_one_minimum_order_from_flat_and_no_add(held, side, want):
    assert ramp.cap(D("0.002"), held, side, D(60_000)) == want * 60_000


# ---- the gate --------------------------------------------------------------------------------------------------------
PASS, NO_COST = ramp.Gate(True, "inside the band"), ramp.Gate(False, "not measured")


@pytest.mark.parametrize("trades, passed", [(29, False), (30, True), (31, True)])
def test_the_boundary_30_closed_trades_passes_and_29_does_not(trades, passed):
    g = ramp.gate(trades, _track(400), PASS)
    assert g.passed is passed and f"{trades} of the 30 closed trades" in g.words


def test_30_trades_on_a_track_shorter_than_its_minimum_track_record_stays_at_minimum_size():
    weak = pd.Series(np.tile([0.0105, -0.0095], 60))  # Sharpe 0.05 a day: about 1,100 days needed, 120 had
    g = ramp.gate(30, weak, PASS)
    assert not g.passed and g.words.startswith("stays at minimum size") and "120 of the" in g.words


def test_30_trades_a_long_enough_track_and_costs_in_band_may_scale():
    g = ramp.gate(30, _track(400), PASS)
    assert g.passed and g.words.startswith("may scale")


def test_the_track_words_say_it_describes_the_no_add_version():
    """Advisor A9 Q4: the ramp takes no adds, so for a Strategy that adds its track is its no-add version, in words."""
    for g in (ramp.gate(30, _track(400), PASS), ramp.gate(3, _track(10), NO_COST)):
        assert "its no-add version: the ramp takes no adds" in g.words


def test_with_no_cost_reader_the_cost_band_counts_as_not_passed_so_nobody_leaves_the_ramp():
    """Advisor Q4(c): (a) and (b) alone never pass while the cost band has no reader."""
    cost = ramp.cost_band(None, "s1")
    assert not cost.passed
    g = ramp.gate(500, _track(400), cost)
    assert not g.passed and "cost band isn't measured yet" in g.words


@pytest.mark.parametrize("track", [_track(400, mean=-0.002), _track(10), pd.Series(dtype=float)])
def test_no_edge_or_too_short_a_track_never_scales_however_many_trades(track):
    assert not ramp.gate(500, track, PASS).passed


# ---- the Strategy's daily track: un-ramped size, no-trade days 0 -----------------------------------------------------
FUND = lambda ts: D(100_000)  # noqa: E731


def _fill(ts, side, qty, price, oid):
    return {"ts": ts, "side": side, "qty": qty, "price": price, "order_id": oid}


def _marks(points):
    return [{"ts": ts, "equity": e} for ts, e in points]


def test_a_days_return_is_its_marked_pnl_over_the_funds_equity_at_00_00_and_today_is_left_out():
    since = T0 + timedelta(hours=9)
    fills = [_fill(since, "BUY", 1, 100, "e1"), _fill(T0 + timedelta(days=1, hours=20), "SELL", 1, 90, "x1")]
    marks = _marks([(since, 0.0), (T0 + timedelta(hours=23), 30.0),  # day 1: marked up 30, still open
                    (T0 + timedelta(days=1, hours=20), -10.0),  # day 2: closed at -10, so -40 on the day
                    (T0 + timedelta(days=2, hours=1), -10.0)])  # today: not a whole day yet
    fund = {T0: D(100_000), T0 + timedelta(days=1): D(80_000)}
    got = ramp.daily_track(marks, ramp.weights(fills, {}), lambda ts: fund.get(ts), since,
                           T0 + timedelta(days=2, hours=5))
    assert list(got.index) == [pd.Timestamp(T0), pd.Timestamp(T0 + timedelta(days=1))]
    assert got.tolist() == pytest.approx([30 / 100_000, -40 / 80_000])


def test_a_day_with_no_trade_counts_as_zero_not_skipped():
    since = T0
    fills = [_fill(T0, "BUY", 1, 100, "e1"), _fill(T0 + timedelta(hours=2), "SELL", 1, 110, "x1")]
    marks = _marks([(T0, 0.0), (T0 + timedelta(hours=2), 10.0), (T0 + timedelta(days=3, hours=1), 10.0)])
    got = ramp.daily_track(marks, ramp.weights(fills, {}), FUND, since, T0 + timedelta(days=4))
    assert got.tolist() == pytest.approx([10 / 100_000, 0.0, 0.0, 0.0])


def test_the_same_trades_at_ramp_size_and_at_normal_size_give_the_same_track():
    """Advisor Q5 cell: a position entered at the minimum (0.002 at 60,000 = 120) whose un-ramped size was 6,000 is
    weighted x50, so its track is the full-size one's to within rounding."""
    def run(qty, unramped):
        k = D(str(qty)) / D("0.1")
        fills = [_fill(T0, "BUY", qty, 60_000, "e1"), _fill(T0 + timedelta(days=1, hours=3), "SELL", qty, 61_200, "x1"),
                 _fill(T0 + timedelta(days=2), "SELL", qty, 61_000, "e2"),
                 _fill(T0 + timedelta(days=2, hours=5), "BUY", qty, 61_300, "x2")]
        orders = {} if unramped is None else {"e1": {"signal": {ramp.UNRAMPED: unramped}},
                                              "e2": {"signal": {ramp.UNRAMPED: unramped * 61_000 / 60_000}}}
        full = [(T0, -3.0), (T0 + timedelta(hours=12), 50.0), (T0 + timedelta(days=1, hours=3), 117.0),
                (T0 + timedelta(days=2, hours=1), 100.0), (T0 + timedelta(days=2, hours=5), 84.0)]
        marks = _marks([(t, float(D(str(e)) * k)) for t, e in full])
        return ramp.daily_track(marks, ramp.weights(fills, orders), FUND, T0, T0 + timedelta(days=4))

    normal, ramped = run(0.1, None), run(0.002, 6_000.0)
    assert ramped.tolist() == pytest.approx(normal.tolist(), rel=1e-9) and normal.abs().sum() > 0


def test_a_day_with_no_fund_equity_fails_the_track_closed():
    marks = _marks([(T0, 0.0), (T0 + timedelta(hours=1), 10.0)])
    with pytest.raises(ValueError, match="no fund equity"):
        ramp.daily_track(marks, [], lambda ts: None, T0, T0 + timedelta(days=2))


# ---- closed trades and positions -------------------------------------------------------------------------------------
def test_positions_open_at_the_first_fill_from_flat_and_a_flip_closes_one_and_opens_the_next():
    t = [T0 + timedelta(hours=h) for h in range(4)]
    fills = [_fill(t[0], "BUY", 1, 1, "a"), _fill(t[1], "SELL", 2, 1, "b"), _fill(t[2], "BUY", 1, 1, "c"),
             _fill(t[3], "BUY", 1, 1, "d")]
    assert ramp.positions(fills) == [(t[0], t[1], "a"), (t[1], t[2], "b"), (t[3], None, "d")]


# ---- read from the journal: the current book and mode only, fails closed ---------------------------------------------
def _store(tmp_path) -> Store:
    url = os.environ.get("TEST_DATABASE_URL")  # both engines: Postgres in CI, SQLite locally
    if url:
        from sleeve_fund.store import make_engine, metadata

        engine = make_engine(url)
        metadata.drop_all(engine)
        store = Store(engine=engine, portfolio_gate=True)
    else:
        store = Store(f"sqlite:///{tmp_path}/ramp.db", portfolio_gate=True)
    store.create_sleeve(name="s1", strategy="buy_and_hold", instrument="BTC/USDT", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=0, risk_profile="balanced")
    return store


def _round_trips(store, n, start, tag):
    for k in range(n):
        ts = start + timedelta(hours=k)
        store.record_fill("s1", side="BUY", qty=0.001, price=60_000, fee=0.0, order_id=f"{tag}b{k}",
                          trade_id=f"{tag}tb{k}", ts=ts)
        store.record_fill("s1", side="SELL", qty=0.001, price=60_100, fee=0.0, order_id=f"{tag}s{k}",
                          trade_id=f"{tag}ts{k}", ts=ts + timedelta(minutes=30))


def _seen(monkeypatch):
    seen = {}

    def spy(trades, track, cost):
        seen["trades"] = trades
        return ramp.Gate(False, "x")

    monkeypatch.setattr(ramp, "gate", spy)
    monkeypatch.setattr(ramp, "daily_track", lambda *a, **k: pd.Series(dtype=float))
    return seen


def test_trades_before_the_current_books_re_base_never_count(tmp_path, monkeypatch):
    store = _store(tmp_path)
    _round_trips(store, 40, T0, "old")  # the earlier book: 40 trades
    rebase = T0 + timedelta(days=3)
    store.settle_flow(store.fund_flow("deposit", 100_000, actor="pm", reason="QA", rebase=True, now=rebase), rebase)
    _round_trips(store, 5, rebase + timedelta(hours=1), "new")
    seen = _seen(monkeypatch)
    ramp.read_gate(store, "s1", rebase + timedelta(days=2))
    assert seen["trades"] == 5


def test_a_paper_to_live_move_restarts_the_track_so_paper_weeks_never_pass_it(tmp_path,
                                                                                                    monkeypatch):
    from sqlalchemy import insert

    from sleeve_fund.store import sleeve_accounts_t

    store = _store(tmp_path)
    _round_trips(store, 40, T0, "paper")  # its paper weeks
    moved = T0 + timedelta(days=5)
    store.create_account("live-1", "live", venue="binance", note="QA: a live account")
    with store.engine.begin() as c:
        c.execute(insert(sleeve_accounts_t).values(sleeve="s1", account="live-1", assigned_at=moved))
    _round_trips(store, 3, moved + timedelta(hours=1), "live")
    seen = _seen(monkeypatch)
    ramp.read_gate(store, "s1", moved + timedelta(days=1))
    assert seen["trades"] == 3


def test_no_trades_yet_in_this_book_stays_at_minimum_size(tmp_path):
    g = ramp.read_gate(_store(tmp_path), "s1", T0)
    assert not g.passed and "no trades yet" in g.words


def test_an_unreadable_track_fails_closed_at_minimum_size():
    broken = SimpleNamespace(engine=None)
    g = ramp.read_gate(broken, "s1", T0)
    assert not g.passed and "couldn't be read" in g.words


def test_read_from_the_journal_a_live_gate_never_passes_while_the_cost_band_has_no_reader(tmp_path, monkeypatch):
    store = _store(tmp_path)
    _round_trips(store, 40, T0, "t")
    monkeypatch.setattr(store, "book_equity_at_or_before", lambda ts: D(100_000))
    g = ramp.read_gate(store, "s1", T0 + timedelta(days=3))
    assert not g.passed and "40 of the 30 closed trades" in g.words and "cost band" in g.words


# ---- the engine: OFF is unchanged, ON caps entries only, once-a-day gate ---------------------------------------------
def _strategy(in_fund=True, on=True, now=T0, read=None):
    events = []
    s = SimpleNamespace(_in_fund=lambda: in_fund, _min_qty=lambda: D("0.001"), _lot=lambda: D("0.001"),
                        instrument=None, _ramp_on=on, _ramp_day=None, _noted=set(), events=events,
                        runtime=SimpleNamespace(store=None, name="s1", now=lambda: now[0] if isinstance(now, list)
                                                else now))
    s._note = lambda kind, msg, level="warning": events.append((kind, msg))
    return s


def test_off_the_limits_are_untouched_the_switch_isnt_reread_and_the_position_is_never_read(monkeypatch):
    monkeypatch.setattr(ramp, "enabled", lambda: 1 / 0)  # read once at start, never per entry
    monkeypatch.setattr(ramp, "read_gate", lambda *a: 1 / 0)
    limits = {"risk per trade": D(5_000)}
    assert LongFlatStrategy._ramp_limit(_strategy(on=False), limits, 1, lambda: 1 / 0, D(60_000)) is None
    assert limits == {"risk per trade": D(5_000)}


def test_the_switch_is_read_once_when_the_strategy_starts():
    src = inspect.getsource(LongFlatStrategy.__init__)
    assert "self._ramp_on = ramp.enabled()" in src
    callers = {name for name, fn in inspect.getmembers(LongFlatStrategy, inspect.isfunction)
               if re.search(r"\bramp\.enabled\(", inspect.getsource(fn))}
    assert callers == {"__init__"}


def test_on_outside_the_fund_or_in_a_backtest_nothing_ramps():
    limits = {}
    assert LongFlatStrategy._ramp_limit(_strategy(in_fund=False), limits, 1, lambda: D(0), D(60_000)) is None
    assert limits == {}


@pytest.mark.parametrize("passed, held, want", [(False, D(0), D(60)), (False, D("0.001"), D(0)), (True, D(0), None)])
def test_on_inside_the_fund_short_of_its_gate_the_entry_is_capped_at_the_minimum(monkeypatch, passed, held, want):
    monkeypatch.setattr(ramp, "read_gate", lambda store, name, now: ramp.Gate(passed, "w"))
    limits = {"risk per trade": D(5_000)}
    g = LongFlatStrategy._ramp_limit(_strategy(), limits, 1, lambda: held, D(60_000))
    assert (g is None) is passed and limits.get(ramp.LIMIT) == want


def test_the_gate_is_read_once_a_utc_day_and_a_change_is_journaled_in_words(monkeypatch):
    now, reads, verdicts = [T0 + timedelta(hours=1)], [], iter([False, True, False])
    monkeypatch.setattr(ramp, "read_gate",
                        lambda store, name, at: reads.append(at) or ramp.Gate(next(verdicts), "the words"))
    s = _strategy(now=now)
    s._ramp_gate = lambda: LongFlatStrategy._ramp_gate(s)
    assert not LongFlatStrategy._ramp_gate(s).passed
    now[0] += timedelta(hours=20)  # the same UTC day: no new read
    assert not LongFlatStrategy._ramp_gate(s).passed and len(reads) == 1 and s.events == []
    now[0] += timedelta(hours=4)  # the next day: read again, passed
    assert LongFlatStrategy._ramp_gate(s).passed and len(reads) == 2
    now[0] += timedelta(days=1)  # dropped back
    assert not LongFlatStrategy._ramp_gate(s).passed
    assert [k for k, _ in s.events] == ["ramp_gate", "ramp_gate"]
    assert s.events[0][1].startswith("Minimum-size ramp passed") and "the words" in s.events[0][1]
    assert "open positions are kept" in s.events[1][1]


def test_only_the_two_entry_sizers_read_the_ramp_so_no_exit_is_ever_capped():
    """Exits, stops, take-profits and reductions go through their own paths: the ramp is a limit only where a new
    entry or add is sized (_open on a perpetual, _buy_all on spot), so a store error on the gate read can't touch an
    exit, and a drop back never cuts an open position (ramp.cap is never below 0 and only caps the new order)."""
    callers = {name for name, fn in inspect.getmembers(LongFlatStrategy, inspect.isfunction)
               if name != "_ramp_limit" and re.search(r"\b_ramp_limit\(", inspect.getsource(fn))}
    assert callers == {"_open", "_buy_all"}


def test_an_accounts_mode_is_fixed_when_it_is_made_never_edited():
    """Advisor A9 (mode rule): the track trusts an account's kind (paper or live) from its creation, so nothing in the
    store ever updates it."""
    import sleeve_fund.store as store_mod

    src = inspect.getsource(store_mod)
    edits = re.findall(r"update\(accounts_t\)[^\n]*\.values\(([^)]*)\)", src)
    assert edits and not [e for e in edits if "kind" in e], edits
