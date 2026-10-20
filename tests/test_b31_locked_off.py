"""B-31 locked half (PM, 20 Oct 2026): T3 (whole fund loss at stops) and both T5 brakes (daily pause, drawdown halt) can
be switched off for testing, like T1, T2 and T4 (B-32), with the same confirm step and reason. Switched off, each is off
everywhere: the save keeps it off, the gate gives it no cap and the brakes never fire. Its saved limit is kept, so
switching it back on restores it. Order rules (a stop or 1x, the stop inside half the distance to liquidation, the
leverage cap) are the Strategy's, not compliance tests, and are untouched.

The two None guards CR asked for after B-32 (T5 off reaching research/portfolio_run._intrabar and the daily pause
below drawdown check in portfolio/compliance.validate_tests) are pinned here: unreachable until T5 could be switched off.

Runs on Postgres too with TEST_DATABASE_URL, on an empty schema."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal as D

import pandas as pd
import pytest
from test_compliance_tests import _edit, engine, store  # noqa: F401
from test_exposure_gate_xfails import _offline  # noqa: F401

from sleeve_fund import compliance_rules as rules
from sleeve_fund.portfolio import compliance
from sleeve_fund.portfolio.compliance import SEEDED, from_compliance
from sleeve_fund.portfolio.limits import Book, Intent, book_breach, decide

LOCKED_BEFORE = ("fund_stop_loss", "fund_day_loss", "fund_drawdown")  # T3, T5 pause, T5 halt


@pytest.mark.parametrize("key", LOCKED_BEFORE)
def test_a_once_locked_test_asked_off_is_valid_on_the_form_and_the_server(key):
    assert rules.validate({**rules.defaults(), key: {"limit": rules.BY_KEY[key].default, "enabled": False}}) == []
    assert rules.LOCKED == frozenset() and compliance.LOCKED == ()


@pytest.mark.parametrize("key", LOCKED_BEFORE)
def test_a_once_locked_test_saved_off_reads_back_off_and_on_again_restores_its_limit(store, key):  # noqa: F811
    c1 = store.compliance()
    limit = next(t["limit"] for t in c1.tests if t["test"] == key)
    store.add_compliance_version(_edit(c1, **{key: False}), c1.asset_types, expected_version=1, note="testing",
                                 environ={})
    c2 = store.compliance()
    assert not next(t["enabled"] for t in c2.tests if t["test"] == key)
    field = compliance.FIELD[key]
    assert getattr(from_compliance(c2), field) is None
    store.add_compliance_version(_edit(c2, **{key: True}), c2.asset_types, expected_version=2, note="back on")
    c3 = store.compliance()
    assert next(t["limit"] for t in c3.tests if t["test"] == key) == limit
    assert getattr(from_compliance(c3), field) == getattr(from_compliance(c1), field)


def test_every_test_off_lets_the_whole_entry_through_and_no_brake_fires():
    """book 100,000; 3 BTC at 50,000 is 1.5x gross and net, 37,500 margin at 4x, 1,500 at the stops: over T1-T4's
    seeded limits it would be trimmed; with every test off it passes whole. A book 40% down fires neither brake."""
    off = replace(SEEDED, gross=None, net_instrument=None, margin=None, open_risk=None, asset_open_risk=None,
                  group_open_risk=None, drawdown=None, daily_loss=None)
    step = D("0.001")
    whole = Intent("BTC", 1, D(3), D(50_000), D(12_500), D(500), step, step)
    tight = replace(SEEDED, gross=0.25)
    assert decide(Book(D(100_000)), whole, tight).approved_qty < D(3)  # the same order is trimmed with a test on
    d = decide(Book(D(100_000)), whole, off)
    assert (d.outcome, d.approved_qty, d.figures) == ("approved", D(3), {}), d
    assert book_breach(off, D(60), D(100), D(100)) is None


@pytest.mark.parametrize("off", [{"drawdown": None}, {"daily_loss": None}, {"drawdown": None, "daily_loss": None}])
def test_compliance_validate_takes_a_t5_brake_off_and_still_orders_the_two_left_on(off):
    """CR (B-32 follow-up 2): portfolio/compliance.py's "daily pause below the drawdown halt" check with one or both
    brakes off. Off is never compared; both on still are."""
    compliance.validate_tests(replace(SEEDED, **off))
    with pytest.raises(ValueError, match="below the drawdown halt"):
        compliance.validate_tests(replace(SEEDED, daily_loss=0.05, drawdown=0.05))


def test_the_research_replay_counts_no_brake_crossing_with_t5_off():
    """CR (B-32 follow-up 1): portfolio_run._intrabar with both T5 brakes off. One long BTC from 100,000 whose next
    hour's low takes the book 40% down: with T5 on that counts a halt and a pause day; off, none, and no crash."""
    from sleeve_fund.research.portfolio_run import _intrabar

    t0, t1 = pd.Timestamp("2026-10-20 01:00", tz="UTC"), pd.Timestamp("2026-10-20 02:00", tz="UTC")
    marks = {t0.tz_localize(None).to_pydatetime(): (100_000.0, [("BTC", 2.0, 50_000.0)]),
             t1.tz_localize(None).to_pydatetime(): (100_000.0, [])}
    hours = {"BTC": pd.DataFrame({"low": [30_000.0], "high": [50_000.0]}, index=[t1])}
    book = pd.Series([100_000.0, 100_000.0], index=[t0, t1])
    worst_on, on = _intrabar(book, marks, hours, SEEDED, None, [])
    assert on == {"halt": 1, "pause": 1} and worst_on == pytest.approx(0.4)
    worst, off = _intrabar(book, marks, hours, replace(SEEDED, drawdown=None, daily_loss=None), None, [])
    assert off == {"halt": 0, "pause": 0} and worst == pytest.approx(0.4)  # the depth is still reported


# ----------------------------------------------------------------------------------------------------------------
# Advisor B31-BRAKE-SWITCH (20 Oct): a brake switched back on judges from the book at switch-on (the Resume re-base,
# halt_reference), the true peak kept; switching one off never lifts a halt or pause in force; refused while the
# exchange order client is on; every change journaled with who, when and why; all of it held in the database.
# ----------------------------------------------------------------------------------------------------------------
from datetime import UTC, datetime, timedelta  # noqa: E402

from sleeve_fund.gate_ledger import DbLedger  # noqa: E402
from sleeve_fund.portfolio.gate import PortfolioState  # noqa: E402
from sleeve_fund.risk import trading_day  # noqa: E402
from sleeve_fund.store import Store  # noqa: E402

NOW = datetime(2026, 10, 20, 13, 0, tzinfo=UTC)  # 14:00 UK


def _state(store, **kw) -> PortfolioState:
    led = DbLedger(store, lambda: ())
    if kw:
        base = dict(equity=D(100_000), mark_ts=NOW, hwm=D(100_000), day=trading_day(NOW), day_start_equity=D(100_000))
        led.set_state(PortfolioState(**(base | kw)))
    return led.state()


def _switch(store, key, on, note="testing the fund", **kw):
    kw.setdefault("environ", {})  # the order client OFF, as these cells' own; the deploy-format path is cell g2's
    c = store.compliance()
    return store.add_compliance_version(_edit(c, **{key: on}), c.asset_types, expected_version=c.version, note=note,
                                        now=NOW, **kw)


def test_a_drawdown_halt_switched_back_on_20_percent_down_rebases_and_keeps_the_true_peak(store):  # noqa: F811
    """Advisor cell a: book 80,000 against a peak of 100,000, the halt switched back on: it measures from 80,000, so
    nothing halts now; a further 15% fall (68,000) does. The peak and the true drawdown are journaled."""
    _switch(store, "fund_drawdown", False)
    _state(store, equity=D(80_000), day_start_equity=D(80_000))
    _switch(store, "fund_drawdown", True)
    st = _state(store)
    assert (st.halt_reference, st.hwm, st.halted) == (D(80_000), D(100_000), None)
    profile = from_compliance(store.compliance())
    assert book_breach(profile, D(80_000), st.reference(), st.day_start_equity) is None
    halt_only = replace(profile, daily_loss=None)  # a 15% fall inside one day would pause first: judge the halt alone
    assert book_breach(halt_only, D("68000.01"), st.reference(), st.day_start_equity) is None
    assert book_breach(halt_only, D(68_000), st.reference(), st.day_start_equity).action == "halt"
    [e] = [e for e in store.events(limit=20) if e["kind"] == "portfolio_brake_rebased"]
    assert "80,000.00" in e["message"] and "100,000.00" in e["message"] and "20.0%" in e["message"], e


def test_a_daily_pause_switched_back_on_mid_day_counts_the_rest_of_the_day_from_then(store):  # noqa: F811
    """Advisor cell b: switched on at 14:00 UK with the day down 4%: today's loss counts from 96,000, so a further 3%
    (93,120) pauses and less doesn't. The next day starts afresh (gate.mark_book), unchanged."""
    _switch(store, "fund_day_loss", False)
    _state(store, equity=D(96_000))
    _switch(store, "fund_day_loss", True)
    st = _state(store)
    assert (st.day_start_equity, st.halt_reference, st.paused) == (D(96_000), None, None)
    profile = from_compliance(store.compliance())
    assert book_breach(profile, D("93120.01"), st.reference(), st.day_start_equity) is None
    assert book_breach(profile, D(93_120), st.reference(), st.day_start_equity).action == "pause"
    [e] = [e for e in store.events(limit=20) if e["kind"] == "portfolio_brake_rebased"]
    assert "96,000.00" in e["message"] and "4.0%" in e["message"], e


@pytest.mark.parametrize("key, held", [("fund_drawdown", {"halted": "the book fell 15%"}),
                                       ("fund_day_loss", {"paused": "the day lost 3%",
                                                          "paused_until": NOW + timedelta(hours=11)})])
def test_switching_a_brake_off_while_it_holds_never_lifts_it(store, key, held):  # noqa: F811
    """Advisor cells c and d: only the PM's Resume (a halt) or the next day (a pause) lifts it."""
    before = _state(store, equity=D(84_000), **held)
    _switch(store, key, False)
    after = _state(store)
    assert (after.halted, after.paused, after.paused_until) == (before.halted, before.paused, before.paused_until)


def test_switching_a_brake_on_with_nothing_marked_yet_rebases_nothing(store):  # noqa: F811
    _switch(store, "fund_drawdown", False)
    _switch(store, "fund_drawdown", True)
    assert _state(store) == PortfolioState() and not [e for e in store.events(limit=20)
                                                       if e["kind"] == "portfolio_brake_rebased"]


@pytest.mark.parametrize("key", LOCKED_BEFORE)
def test_switching_off_is_refused_while_the_exchange_order_client_is_on(store, key):  # noqa: F811
    """Advisor MUST 2 / cell g: paper only. With the order client ON the brakes and T3 stay on; switching back on, or
    a limit edit, is never refused."""
    on = {"EXEC_MODE": "demo", "EXEC_HOST": "api-demo.bybit.com", "BYBIT_DEMO_API_KEY": "k",
          "BYBIT_DEMO_API_SECRET": "s"}
    with pytest.raises(ValueError, match="order client is on"):
        _switch(store, key, False, environ=on)
    assert store.compliance().version == 1
    _switch(store, key, False, environ={})
    _switch(store, key, True, environ=on)
    assert store.compliance().version == 3


def test_an_off_with_no_reason_is_refused_and_each_change_is_journaled_with_who_when_and_why(store):  # noqa: F811
    """Advisor cell i and MUST 3: the reason is required; the version and its decision row carry the PM, the time
    and the reason, and what changed."""
    with pytest.raises(ValueError, match="say why"):
        _switch(store, "fund_drawdown", False, note="  ")
    _switch(store, "fund_drawdown", False, note="testing the brakes")
    [d] = [d for d in store.decisions(action="compliance_version") if "v2" in d["reason"]]
    assert d["actor"] == "pm" and d["ts"].replace(tzinfo=UTC) == NOW and "testing the brakes" in d["reason"], d


def test_a_restart_keeps_a_brake_off_and_a_rebased_reference(store, engine):  # noqa: F811
    """Advisor cell h: both live in the database, never in a process: a fresh store on it reads them back."""
    _switch(store, "fund_drawdown", False)
    _state(store, equity=D(80_000))
    again = Store(engine=engine)
    assert from_compliance(again.compliance()).drawdown is None
    _switch(again, "fund_drawdown", True)
    assert _state(Store(engine=engine)).halt_reference == D(80_000)


def test_the_paper_gate_marks_a_book_40_percent_down_with_t5_off_and_neither_halts_nor_pauses(store):  # noqa: F811
    """The paper engine's side of CR's None guards (HoE: both engines): gate.mark_book on the fund's own ledger, T5 off,
    a book 40% below its peak and its day's start: no halt, no pause, the true peak kept. A brake on again without the
    switch's re-base fires on the very next mark, which is why switching on re-bases (cells above)."""
    from sleeve_fund.portfolio.gate import mark_book

    led = DbLedger(store, lambda: ())
    _state(store, equity=D(100_000))
    off = replace(from_compliance(store.compliance()), drawdown=None, daily_loss=None)
    assert mark_book(led, D(60_000), NOW + timedelta(minutes=1), off) is None
    st = led.state()
    assert (st.halted, st.paused, st.hwm) == (None, None, D(100_000))
    assert mark_book(led, D(60_000), NOW + timedelta(minutes=2), SEEDED) == "halt"  # on, not re-based: it fires


ON = {"EXEC_MODE": "demo", "EXEC_HOST": "api-demo.bybit.com", "BYBIT_DEMO_API_KEY": "k", "BYBIT_DEMO_API_SECRET": "s"}


@pytest.mark.parametrize("key", LOCKED_BEFORE)
def test_in_deploy_format_the_switch_reads_the_supervisors_client_not_the_dashboards_own(store, key, monkeypatch):  # noqa: F811
    """CR MUST (20 Oct): the dashboard saves with no client settings of its own (docker-compose gives them to the
    supervisor only). The save reads what the supervisor published at its start: ON refuses switching off, so does no
    word from a supervisor at all; OFF lets it through."""
    from sleeve_fund.paper.exec_client import publish

    for var in ON:
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ValueError, match="until the supervisor has said the exchange order client is off"):
        _switch(store, key, False, environ=None)  # no supervisor has said: unknown, so refused
    publish(store, ON)
    with pytest.raises(ValueError, match="order client is on"):
        _switch(store, key, False, environ=None)
    assert store.compliance().version == 1
    publish(store, {})  # a supervisor restarted with the client OFF
    _switch(store, key, False, environ=None)
    assert store.compliance().version == 2
    publish(store, ON)
    _switch(store, key, True, environ=None)  # switching back on is never refused
    assert store.compliance().version == 3


@pytest.mark.parametrize("key, held", [("fund_drawdown", {"halted": "the book fell 15%"}),
                                       ("fund_day_loss", {"paused": "the day lost 3%",
                                                          "paused_until": NOW + timedelta(hours=11)})])
def test_switching_a_brake_back_on_while_it_holds_never_lifts_it_and_only_moves_its_reference(store, key, held):  # noqa: F811
    """CR MUST (20 Oct): a halt or a pause in force stays through off and back on, unlike a Resume
    (gate.resume_after_halt); only the brake's reference moves to the book."""
    before = _state(store, equity=D(84_000), **held)
    _switch(store, key, False)
    _switch(store, key, True)
    after = _state(store)
    assert (after.halted, after.paused, after.paused_until) == (before.halted, before.paused, before.paused_until)
    moved = after.halt_reference if key == "fund_drawdown" else after.day_start_equity
    assert moved == D(84_000) and after.hwm == before.hwm


def test_a_stale_form_never_rebases_a_brake_another_form_switched_on(store):  # noqa: F811
    """CR follow-up: the re-base now comes before the version is written, so a form opened before another saved is
    refused before it, and the reference the other form set stays."""
    _switch(store, "fund_drawdown", False)
    stale = store.compliance()
    _state(store, equity=D(90_000))
    _switch(store, "fund_drawdown", True)  # re-based to 90,000
    _state(store, equity=D(85_000), halt_reference=D(90_000))
    with pytest.raises(ValueError, match="changed since this page was opened"):
        store.add_compliance_version(_edit(stale, fund_drawdown=True), stale.asset_types,
                                     expected_version=stale.version, note="testing the fund", now=NOW, environ={})
    assert _state(store).halt_reference == D(90_000)


def test_the_one_client_reader_says_on_off_or_unknown_for_the_store_and_the_page(store, monkeypatch):  # noqa: F811
    """CR MUST (20 Oct): one reader (exec_client.client_state) that the store's save and FE's page both call."""
    from sleeve_fund.paper.exec_client import client_state, publish

    for var in ON:
        monkeypatch.delenv(var, raising=False)
    assert client_state(store) == "unknown"
    publish(store, ON)
    assert client_state(store) == "on"
    publish(store, {})
    assert client_state(store) == "off"
    monkeypatch.setenv("EXEC_MODE", "demo")  # this process's own settings on: on, whatever the supervisor said
    assert client_state(store) == "on"
    assert client_state(store, {}) == "off" and client_state(store, ON) == "on"  # a given environ alone decides
