"""CUT-A (Advisor; HoE Drop A) with SIZE-FUND Q4/Q5: inside the fund no decision reads a Strategy's own equity.

- Q4: the Strategy's own drawdown halt and daily-loss pause are gone in the fund (the fund's brakes, T5, act). Its
  liquidation still halts it. A backtest's single Strategy is the whole fund and keeps its own.
- Q5: an entry sizes off the fund's last portfolio mark; a stale or unreadable mark is refused, never sized off.
- End to end (test_cut_a_e2e): a Strategy with a starting balance of 0 sizes off the fund through the paper engine.

Red first on 3a272149 (the per-Strategy halt in force; paper sized off its own equity)."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from sleeve_fund.gate_ledger import DbLedger
from sleeve_fund.paper.runtime import SleeveRuntime
from sleeve_fund.portfolio.gate import PortfolioState
from sleeve_fund.store import Store
from sleeve_fund.strategies.base import LongFlatStrategy

T0 = datetime(2025, 10, 3, 9, 0, tzinfo=UTC)
HELD = {"cash": 0.0, "qty": 1.0}


def _store(tmp_path, gate: bool) -> Store:
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        from sleeve_fund.store import make_engine, metadata

        engine = make_engine(url)
        metadata.drop_all(engine)
        return Store(engine=engine, portfolio_gate=gate)
    return Store(f"sqlite:///{tmp_path}/c.db", portfolio_gate=gate)


def _bridge(store) -> None:
    """The PM's re-base deposit lands: inside the fund from here (B34-CAPITAL-PREDICATE: the bridge decides; TEC-B34,
    before it a Strategy still carries its own capital and keeps its own halts). A Strategy made before it keeps its
    starting balance, as the live fund's were before the PM's re-base."""
    before = T0 - timedelta(hours=1)
    store.settle_flow(store.fund_flow("deposit", 100_000, actor="pm", reason="the fund's capital", rebase=True,
                                      now=before), before)


def _runtime(store, t, balance=10_000):
    if "s1" not in {s.name for s in store.sleeves()}:
        store.create_sleeve(name="s1", strategy="buy_and_hold", instrument="BTC/USDT", bar_spec="1-DAY-LAST-EXTERNAL",
                            starting_balance=balance, risk_profile="balanced")
    rt = SleeveRuntime(store, "s1", now=lambda: t[0])
    rt.on_start(0.0005)
    return rt


# ---- Q4: no per-Strategy drawdown halt or daily pause inside the fund ------------------------------------------------
@pytest.mark.parametrize("gate", [True, False])
def test_a_strategy_down_20_percent_keeps_trading_inside_the_fund_and_halts_outside_it(tmp_path, gate):
    t = [T0]
    store = _store(tmp_path, gate)
    rt = _runtime(store, t)
    if gate:
        _bridge(store)
    rt.tick(equity=10_000, price=10_000, **HELD)
    t[0] += timedelta(days=1, hours=1)  # a new day: the daily pause is not what acts
    rt.tick(equity=10_000, price=10_000, **HELD)
    t[0] += timedelta(minutes=1)
    sent = rt.tick(equity=7_900, price=7_900, **HELD)  # its contribution down 21%
    if gate:
        assert sent is None and store.sleeve("s1").status == "running" and rt.can_open()
    else:  # a backtest's or a tool's journal: the Strategy is the whole fund and keeps its own halt
        assert sent == "flatten" and store.sleeve("s1").status == "halted"


def test_a_days_loss_past_the_profiles_limit_never_pauses_a_strategy_inside_the_fund(tmp_path):
    t = [T0]
    store = _store(tmp_path, True)
    rt = _runtime(store, t)
    _bridge(store)
    rt.tick(equity=10_000, price=10_000, **HELD)
    t[0] += timedelta(hours=10)
    assert rt.tick(equity=9_450, price=9_450, **HELD) is None  # down 5.5% on the day
    assert store.sleeve("s1").status == "running"


def test_a_strategy_marked_at_zero_inside_the_fund_is_never_halted_as_invalid_equity(tmp_path):
    """Its own equity is its P&L from a starting balance of 0: zero is a fact, not bad data."""
    t = [T0]
    store = _store(tmp_path, True)
    rt = _runtime(store, t, balance=0)
    _bridge(store)
    assert rt.tick(equity=0.0, price=60_000, cash=0.0, qty=0.0) is None
    assert store.sleeve("s1").status == "running"


def test_its_liquidation_still_halts_it_inside_the_fund(tmp_path):
    t = [T0]
    store = _store(tmp_path, True)
    rt = _runtime(store, t)
    rt.tick(equity=10_000, price=10_000, **HELD)
    t[0] += timedelta(minutes=1)
    assert rt.tick(equity=0.0, price=5_000, ruined="liquidated at 5,000", **HELD) == "flatten"
    assert store.sleeve("s1").status == "halted"


# ---- Q5: sized off the fund's last mark; stale is refused ------------------------------------------------------------
def _reader(tmp_path, marked_ago: timedelta | None, equity=D("19887.78")):
    store = _store(tmp_path, True)
    now = T0
    if marked_ago is not None:
        DbLedger(store, lambda: None).set_state(PortfolioState(mark_ts=now - marked_ago, equity=equity, hwm=equity,
                                                               day=now.date(), day_start_equity=equity))
    s = SimpleNamespace(runtime=SimpleNamespace(store=store, now=lambda: now), log=SimpleNamespace(warning=print))
    return LongFlatStrategy._paper_fund_equity(s)


@pytest.mark.parametrize("ago, base", [(timedelta(seconds=5), 19_887.78), (timedelta(seconds=59), 19_887.78),
                                       (timedelta(seconds=61), None), (None, None)])
def test_the_fund_figure_sizes_only_while_its_mark_is_fresh(tmp_path, ago, base):
    got = _reader(tmp_path, ago)
    assert got == (pytest.approx(base) if base is not None else None)


def test_a_fund_held_strategy_never_falls_back_to_its_own_equity():
    """Paper inside the fund with a stale book: refused (None), not its own equity."""
    s = SimpleNamespace(fund_equity=None, _fund_held=lambda: True, _paper_fund_equity=lambda: None,
                        _entry_px=None, _entry_qty=0.0)
    assert LongFlatStrategy._sizing_base(s, 10_000.0) is None
    s._paper_fund_equity = lambda: 19_887.78
    assert LongFlatStrategy._sizing_base(s, 0.0) == 19_887.78


def test_a_gain_not_yet_taken_anywhere_in_the_fund_never_sizes_an_entry_up(tmp_path):
    """TEC-QD-CUTA (HoE): base = deposits net of withdrawals + realised + min(0, unrealised). A Strategy holding 10 from
    1,000, marked at 1,200, adds 2,000 of unrealised gain to the fund's mark; the next entry anywhere sizes off the
    fund without it (20,000 less the entry's fee), not 22,000."""
    from sleeve_fund.supervisor import fund_equity

    store = _store(tmp_path, True)
    store.create_sleeve(name="a", strategy="buy_and_hold", instrument="BTC/USDT", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=20_000, risk_profile="balanced")
    store.record_fill("a", side="BUY", qty=10, price=1_000, fee=5.0, order_id="o1", trade_id="t1", ts=T0)
    store.record_equity("a", equity=21_995, cash=9_995, qty=10, price=1_200, benchmark=20_000, ts=T0)
    marked = fund_equity(store)
    DbLedger(store, lambda: None).set_state(PortfolioState(mark_ts=T0, equity=marked, hwm=marked, day=T0.date(),
                                                           day_start_equity=marked))
    s = SimpleNamespace(runtime=SimpleNamespace(store=store, now=lambda: T0), log=SimpleNamespace(warning=print))
    assert LongFlatStrategy._paper_fund_equity(s) == pytest.approx(19_995.0)


def test_inside_the_fund_the_strategys_own_open_risk_check_never_refuses_an_entry():
    """CUT-A (d): the per-Strategy interim open-risk refusal read the Strategy's own book; inside the fund the gate's
    whole-fund open-risk limit (T3) judges the entry, under its own name."""
    s = SimpleNamespace(_stop_frac=0.02, pair=None, _backtest=False, _fund_held=lambda: True)
    assert LongFlatStrategy._open_risk_refusal(s, 1, 10.0, 60_000.0, 0.0, 0) is None  # equity 0: nothing of its own


def test_the_fund_figure_is_pe2s_reader_once_it_is_there(tmp_path, monkeypatch):
    """PE2's supervisor.sizing_equity (the book less every position's gain not yet taken, None when stale) replaces the
    raw mark once merged; None from it is refused."""
    import sleeve_fund.supervisor as sup

    s = SimpleNamespace(runtime=SimpleNamespace(store=None, now=lambda: T0), log=SimpleNamespace(warning=print))
    monkeypatch.setattr(sup, "sizing_equity", lambda store, now=None: D(18000), raising=False)
    assert LongFlatStrategy._paper_fund_equity(s) == 18_000.0
    monkeypatch.setattr(sup, "sizing_equity", lambda store, now=None: None, raising=False)
    assert LongFlatStrategy._paper_fund_equity(s) is None


# ---- MUST-FUNDCASH-LOSS: who keeps the shortfall cover (CR F-FC-1, F-FC-2) ------------------------------------------
@pytest.mark.parametrize("held, rebased, bridged", [(True, True, True), (True, False, False), (False, True, False),
                                                    (False, False, False)])
def test_only_a_fund_held_strategy_after_the_re_base_loses_its_own_shortfall_cover(tmp_path, held, rebased, bridged):
    """Before the PM's re-base deposit a Strategy still carries its own capital (F-FC-1), and one outside the fund's
    book keeps its own account after it (F-FC-2): both keep the cover. Only a fund-held one after the bridge doesn't."""
    store = _store(tmp_path, True)
    if rebased:
        store.settle_flow(store.fund_flow("deposit", 20_000, actor="pm", reason="QA", rebase=True, now=T0), T0)
    s = SimpleNamespace(runtime=SimpleNamespace(fund_held=held, store=store))
    assert LongFlatStrategy._fund_bridged(s) is bridged


@pytest.mark.parametrize("bridged, credit", [(True, 0.0), (False, 250.0)])
def test_a_book_below_zero_once_flat_is_covered_only_outside_the_bridged_fund(monkeypatch, bridged, credit):
    rows = []
    monkeypatch.setattr(LongFlatStrategy, "_fund_bridged", lambda self: bridged)
    s = SimpleNamespace(_mark=lambda: (-250.0, -250.0, 0.0, 60_000.0), _cash_adj=0.0,
                        insurance_log=[], clock=SimpleNamespace(utc_now=lambda: T0),
                        runtime=SimpleNamespace(name="s1", now=lambda: T0, store=SimpleNamespace(
                            record_insurance=lambda *a, **k: rows.append(k) or "new", event=lambda *a, **k: None)))
    LongFlatStrategy._cover_shortfall(s, 60_000.0, ("o1", "t1"))
    assert s._cash_adj == credit and len(rows) == (1 if credit else 0)


# ---- MARK-ON-STOP inside the fund: the stop marks a Strategy whose own equity is below zero --------------------------
@pytest.mark.parametrize("funded, marked", [(200_000.0, True), (0.0, False)])
def test_the_stop_marks_a_fund_held_strategy_below_zero_and_skips_a_wiped_out_one(funded, marked):
    """A fund-held Strategy's own equity starts at 0, so after a loss it is below zero: its P&L, not a book that can't
    be valued. Its simulated account (with the fund's virtual funding back in) is what has to be above zero, so its
    stop still leaves a last mark holding every fill (no tie for the start check). Outside the fund an account at or
    below zero is wiped out and is not marked, as before."""
    got = []
    s = SimpleNamespace(_restore=None, _mark=lambda: (-1_716.15, -1_716.15, 0.0, 58_000.0),
                        log=SimpleNamespace(warning=print),
                        runtime=SimpleNamespace(backtest=False, funded=funded, final_mark=lambda **k: got.append(k)))
    LongFlatStrategy._final_mark(s)
    assert got == ([{"equity": -1_716.15, "cash": -1_716.15, "qty": 0.0, "price": 58_000.0}] if marked else [])


@pytest.mark.parametrize("held, funded, breached", [(True, 200_000.0, False), (False, 0.0, True)])
def test_a_trade_calls_the_guards_off_the_account_and_inside_the_fund_never_off_its_own_equity(held, funded, breached):
    """Paper on a perp, on every trade (_guard_breached): a fund-held Strategy holding a position is below zero after
    its entry's fee alone, and its own halt and pause don't act inside the fund, so neither may call a tick on every
    trade; its account (the fund's funding back in) and its liquidation distance still do. Outside the fund an
    account at or below zero still acts at once."""
    s = SimpleNamespace(_entry_px=60_000.0, _mark=lambda: (-30.0, -6_030.0, 0.1, 60_000.0), pair=None,
                        _guard_book=lambda e, c, q, p: (e, c, q), _liq_now=lambda c, q: None,
                        runtime=SimpleNamespace(status="running", fund_held=held, funded=funded, _day_open=0.0, peak=0.0,
                                                halts=True, profile=SimpleNamespace(min_liquidation_distance=0.1)))
    assert LongFlatStrategy._guard_breached(s, 60_000.0) is breached


def test_inside_the_fund_a_perp_nearing_liquidation_still_calls_the_guards_on_the_trade():
    """F-GUARD-OWN-EQ (a), HoE: the per-trade call still acts inside the fund when the position is within the profile's
    distance of its liquidation price (the cut), whatever the Strategy's own equity reads."""
    s = SimpleNamespace(_entry_px=60_000.0, _mark=lambda: (-30.0, -6_030.0, 0.1, 60_000.0), pair=None,
                        _guard_book=lambda e, c, q, p: (e, c, q), _liq_now=lambda c, q: 57_000.0,
                        runtime=SimpleNamespace(status="running", fund_held=True, funded=200_000.0, _day_open=0.0,
                                                peak=0.0, halts=True, profile=SimpleNamespace(min_liquidation_distance=0.1)))
    assert LongFlatStrategy._guard_breached(s, 60_000.0) is True  # 5% from liquidation, inside the 10% distance
    s._liq_now = lambda c, q: 40_000.0
    assert LongFlatStrategy._guard_breached(s, 60_000.0) is False  # 33% away: nothing to act on


def test_inside_the_fund_an_account_at_or_below_zero_calls_the_guards_on_the_trade_on_its_own():
    """F-GE-1 (CR): the per-trade call's account check stands alone. A fund-held perp whose simulated account (its own
    equity with the fund's funding back in) is at or below zero acts on the trade even with no liquidation price near
    and the Strategy's own halt and pause not asked inside the fund."""
    s = SimpleNamespace(_entry_px=60_000.0, _mark=lambda: (-30.0, -6_030.0, 0.1, 60_000.0), pair=None,
                        _guard_book=lambda e, c, q, p: (e, c, q), _liq_now=lambda c, q: None,
                        runtime=SimpleNamespace(status="running", fund_held=True, funded=30.0, _day_open=0.0, peak=0.0,
                                                halts=True, profile=SimpleNamespace(min_liquidation_distance=0.1)))
    assert LongFlatStrategy._guard_breached(s, 60_000.0) is True
    s.runtime.funded = 31.0
    assert LongFlatStrategy._guard_breached(s, 60_000.0) is False


@pytest.mark.parametrize("fund_held", [True, False])
def test_f_ge_2_a_two_leg_perp_leg_past_half_its_entry_distance_acts_on_the_trade(fund_held):
    """F-GE-2 (CR 01:36 UK 15 Oct): the per-trade call acts when a two-leg Strategy's perp leg comes nearer its
    liquidation price than half the distance it was entered at (Pair.distance_breached, Advisor B), in or out of the
    fund, with the account well above zero and the venue's minimum liquidation distance still clear."""
    from sleeve_fund.strategies.pair import CLAMP, Pair

    pair = SimpleNamespace(state="hedged", _entry_dist={})
    pair.distance_breached = lambda leg, price, liq: Pair.distance_breached(pair, leg, price, liq)
    s = SimpleNamespace(_entry_px=60_000.0, _mark=lambda: (10_000.0, 4_000.0, 0.1, 60_000.0), pair=pair,
                        _guard_book=lambda e, c, q, p: (e, c, q), _liq_now=lambda c, q: 30_000.0,
                        runtime=SimpleNamespace(status="running", fund_held=fund_held, funded=0.0, _day_open=10_000.0,
                                                peak=10_000.0, halts=True, profile=SimpleNamespace(
                                                    min_liquidation_distance=0.01, max_drawdown=0.99,
                                                    daily_loss=0.99)))
    # Entered 50% from liquidation, so it acts nearer than 25%: 0.4 at 50,000 leaves it; 0.1667 at 36,000 acts.
    assert CLAMP == 0.5
    assert LongFlatStrategy._guard_breached(s, 50_000.0) is False
    assert LongFlatStrategy._guard_breached(s, 36_000.0) is True
    pair.state = "flat"  # a pair not legging or hedged has no entry distance to keep
    assert LongFlatStrategy._guard_breached(s, 36_000.0) is False


# ---- Own capital (PM, 30 Oct 2026): a leg inside the fund sized as a research leg's own account ---------------------
def _own(own_pnl, fund=99_831.66, entry=None, price=100.0, own_capital=17_271.0):
    return SimpleNamespace(fund_equity=None, _fund_held=lambda: True, _paper_fund_equity=lambda: fund,
                           _cfg=SimpleNamespace(own_capital=own_capital), _mark=lambda: (own_pnl, 0.0, 0.0, price),
                           _entry_px=entry, _entry_qty=1.0 if entry else 0.0, _entry_side=1, _price=lambda: price)


def test_own_capital_sizes_off_itself_plus_its_own_pnl_not_the_fund():
    """A leg at 0.173 of a 99,831.66 fund sizes off 17,271 and compounds on its own P&L, gains or losses, as the
    basket's own-account leg does; the fund's figure stays the one journaled for a liquidation's share."""
    s = _own(0.0)
    assert LongFlatStrategy._sizing_base(s, 0.0) == pytest.approx(17_271.0)
    assert LongFlatStrategy._fund_at_entry(SimpleNamespace(**vars(s), _in_fund=lambda: True), 17_271.0) == {
        "fund_equity": 99_831.66, "own_capital_base": 17_271.0}
    assert LongFlatStrategy._sizing_base(_own(1_200.0), 0.0) == pytest.approx(18_471.0)
    assert LongFlatStrategy._sizing_base(_own(-2_000.0), 0.0) == pytest.approx(15_271.0)


def test_own_capital_leaves_out_a_gain_not_yet_taken_and_never_passes_the_fund():
    held = _own(10.0, entry=100.0, price=110.0)  # 1 unit from 100 marked at 110: the 10 gain is left out
    assert LongFlatStrategy._sizing_base(held, 0.0) == pytest.approx(17_271.0)
    assert LongFlatStrategy._sizing_base(_own(0.0, fund=12_000.0), 0.0) == pytest.approx(12_000.0)


def test_own_capital_still_fails_closed_on_a_stale_fund_mark():
    assert LongFlatStrategy._sizing_base(_own(0.0, fund=None), 0.0) is None


def test_without_own_capital_a_leg_sizes_off_the_fund_as_before():
    s = _own(500.0, own_capital=None)
    assert LongFlatStrategy._sizing_base(s, 500.0) == pytest.approx(99_831.66)
    assert LongFlatStrategy._fund_at_entry(SimpleNamespace(**vars(s), _in_fund=lambda: True), 99_831.66) == {
        "fund_equity": 99_831.66}


def test_own_capital_must_be_a_positive_amount():
    from nautilus_trader.model import BarType, InstrumentId

    from sleeve_fund.strategies.base import LongFlatConfig

    kw = dict(instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"), assumed_taker_fee=0.0005,
              bar_type=BarType.from_str("BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL"))
    assert LongFlatConfig(**kw, own_capital=17_271.0).own_capital == 17_271.0
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="own_capital"):
            LongFlatConfig(**kw, own_capital=bad)
