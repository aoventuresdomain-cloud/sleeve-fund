"""Drop B (PE2): NIGHTLY-RECON runs the per-poll book check (RECON-B) on the night's journal, read-only; and the engine
readers the Portfolio Settings page shows: the fund state banner (C7, fund_state) and "In use now" per test, asset and
group (C2, compliance_use), on the book and the tests version the gate checks against. SQLite, and Postgres with
TEST_DATABASE_URL."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal as D

from test_drop_a_fund_book import MIN, _broken, _entry, _rebased
from test_halt_ref import T0, _make, _mark, _trade, store  # noqa: F401

from sleeve_fund import nightly_recon, supervisor
from sleeve_fund.gate_ledger import DbLedger
from sleeve_fund.portfolio import compliance
from sleeve_fund.portfolio.gate import MARK_STALE, next_utc_midnight
from sleeve_fund.store import Store

NIGHT = T0 + timedelta(days=1)


def _url(store) -> str:
    return store.engine.url.render_as_string(hide_password=False)


def _night(store, tmp_path):
    store.engine.dispose()  # the night opens its own read-only connection
    ok, path, fails = nightly_recon.run(tmp_path / "recon", _url(store), NIGHT)
    return ok, path.read_text(), fails


def _fund(store):
    """A fund-held book of 100,000 with one Strategy that made 100 and is flat, marked."""
    _rebased(store)
    _make(store, "x", 0)
    _trade(store, "x", 100, T0 + MIN)
    _mark(store, T0 + MIN)


# --- NIGHTLY-RECON: the identity check ---------------------------------------------------------------------------------


def test_the_night_runs_the_book_check_and_a_whole_book_passes(store, tmp_path):  # noqa: F811
    _fund(store)
    ok, page, fails = _night(store, tmp_path)
    assert ok and not fails
    assert "The fund's book check:\n  PASS  the fund's cash and Strategy P&L match the journal (gap 0.00" in page


def test_pin_a_book_that_doesnt_add_up_fails_the_night(store, tmp_path, monkeypatch):  # noqa: F811
    _fund(store)
    real = Store.strategy_cash_pnl
    monkeypatch.setattr(Store, "strategy_cash_pnl", lambda self, name, since=None, at=None:
                        real(self, name, since, at) + D("0.02"))
    ok, page, fails = _night(store, tmp_path)
    assert not ok and fails == ["the fund's cash and Strategy P&L differ from the journal by -0.02"]
    assert page.startswith("FAIL\n")


def test_pin_a_position_the_fills_dont_come_to_fails_the_night(store, tmp_path):  # noqa: F811
    _fund(store)
    store.record_equity("x", equity=100.0, cash=100.0, qty=1.0, price=1_000.0, benchmark=100.0, ts=T0 + 2 * MIN)
    ok, _, fails = _night(store, tmp_path)
    assert not ok
    assert "x: the journal's fills come to 0 but the live position is 1" in fails


def test_pin_an_insurance_credit_inside_the_fund_fails_the_night(store, tmp_path):  # noqa: F811
    from test_drop_a_fund_book import _covered

    _fund(store)
    _covered(store, "x", "liquidation", T0 + 2 * MIN, "liq1")
    ok, _, fails = _night(store, tmp_path)
    assert not ok and any("an insurance credit of" in f for f in fails)


def test_a_hold_the_poll_latched_is_known_and_a_book_that_adds_up_again_passes(store, tmp_path, monkeypatch):  # noqa: F811
    _fund(store)
    fix = _broken(store, monkeypatch)
    _mark(store, T0 + 2 * MIN)
    fix()
    ok, page, _ = _night(store, tmp_path)
    assert ok
    assert "  NOTE  known, new entries are already held by the book check until the PM resumes: " in page


def test_before_the_fund_holds_the_cash_there_is_no_book_check_and_no_fail(store, tmp_path):  # noqa: F811
    _make(store, "a", 10_000)
    _trade(store, "a", 50, T0)
    ok, page, _ = _night(store, tmp_path)
    assert ok and "  NOTE  the fund doesn't hold the cash yet" in page


def test_the_book_check_at_night_writes_nothing_to_the_journal(store, tmp_path, monkeypatch):  # noqa: F811
    _fund(store)
    _broken(store, monkeypatch)
    url = _url(store)
    if not url.startswith("sqlite"):
        return  # Postgres: the read-only engine refuses any write (start_check's own cell)
    path = url[len("sqlite:///"):]

    def digest():
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    store.engine.dispose()  # the test's own connection closed first: its close may write
    before = digest()
    _night(store, tmp_path)
    assert digest() == before


# --- C7: the fund state banner -------------------------------------------------------------------------------------


def test_a_fresh_mark_with_nothing_wrong_is_normal_no_banner(store):  # noqa: F811
    _fund(store)
    assert supervisor.fund_state(store, T0 + MIN) == {"state": "normal", "reason": None, "since": None, "until": None}


def test_a_mark_older_than_the_gate_allows_is_stale_since_the_last_mark(store):  # noqa: F811
    _fund(store)
    st = supervisor.fund_state(store, T0 + MIN + MARK_STALE + timedelta(seconds=1))
    assert st["state"] == "stale" and st["since"] == T0 + MIN and "last marked 61 s ago" in st["reason"]
    assert supervisor.fund_state(store, T0 + MIN + MARK_STALE)["state"] == "normal"  # the gate's own edge


def test_the_daily_brake_says_until_the_next_day_begins_and_since_the_pause(store):  # noqa: F811
    _fund(store)
    _trade(store, "x", -3_200, T0 + 2 * MIN)  # 3.2% of the day's start, past T5a's 3%
    _mark(store, T0 + 3 * MIN)
    st = supervisor.fund_state(store, T0 + 3 * MIN)
    assert (st["state"], st["since"], st["until"]) == ("brake", T0 + 3 * MIN, next_utc_midnight(T0))
    assert "daily limit" in st["reason"]


def test_the_drawdown_halt_says_why_and_since_the_halt(store):  # noqa: F811
    _fund(store)
    _trade(store, "x", -16_000, T0 + 2 * MIN)  # past T5b's 15%
    _mark(store, T0 + 3 * MIN)
    st = supervisor.fund_state(store, T0 + 3 * MIN)
    assert (st["state"], st["since"], st["until"]) == ("halt", T0 + 3 * MIN, None)
    assert "drawdown limit" in st["reason"]


def test_a_failed_book_check_shows_until_the_pms_resume_and_a_halt_outranks_it(store, monkeypatch):  # noqa: F811
    _fund(store)
    _broken(store, monkeypatch)
    _mark(store, T0 + 2 * MIN)
    st = supervisor.fund_state(store, T0 + 2 * MIN)
    assert (st["state"], st["since"]) == ("book_check", T0 + 2 * MIN) and "0.02" in st["reason"]
    led = DbLedger(store, lambda: ())
    led.set_state(replace(led.state(), halted="the test's halt"))
    assert supervisor.fund_state(store, T0 + 2 * MIN)["state"] == "halt"


# --- C2: in use now ------------------------------------------------------------------------------------------------


def test_in_use_now_counts_a_live_reservation_against_every_test_by_hand(store):  # noqa: F811
    _fund(store)  # book 100,100; day start 100,000; peak 100,100
    assert _entry(store, T0 + MIN).outcome == "approved"  # 0.001 BTC at 60,000: margin 60, risk to its stop 1.2
    use = supervisor.compliance_use(store)
    t = use["tests"]
    assert use["equity"] == D(100_100) and use["marked_at"] == T0 + MIN
    assert use["version"] == compliance.load(store).version
    assert t["asset_stop_loss"] == {"test": "T1", "limit": D("4004"), "used": {"BTC": D("1.2")}}
    assert t["type_stop_loss"] == {"test": "T2", "limit": D("6006"), "used": {"large": D("1.2")}}
    assert t["fund_stop_loss"] == {"test": "T3", "limit": D("10010"), "used": D("1.2")}
    assert t["fund_gross"] == {"test": "T4", "limit": D("150150"), "used": D(60)}
    assert t["asset_net"] == {"test": "T4", "limit": D("50050"), "used": {"BTC": D(60)}}
    assert t["fund_margin"] == {"test": "T4", "limit": D("50050"), "used": D(60)}
    assert t["fund_day_loss"] == {"test": "T5", "limit": D("3000"), "used": D(0)}
    assert t["fund_drawdown"] == {"test": "T5", "limit": D("15015"), "used": D(0)}


def test_in_use_now_is_the_gates_own_before_figure_for_the_next_idea(store):  # noqa: F811
    _fund(store)
    _entry(store, T0 + MIN)
    use = supervisor.compliance_use(store)
    figs = _entry(store, T0 + MIN + timedelta(seconds=1)).figures
    e = use["equity"]
    pairs = {"asset_open_risk": use["tests"]["asset_stop_loss"]["used"]["BTC"],
             "group_open_risk": use["tests"]["type_stop_loss"]["used"]["large"],
             "open_risk": use["tests"]["fund_stop_loss"]["used"], "gross": use["tests"]["fund_gross"]["used"],
             "net_instrument": use["tests"]["asset_net"]["used"]["BTC"], "margin": use["tests"]["fund_margin"]["used"]}
    for field, used in pairs.items():
        assert figs[field]["before"] == float(used / e), field
    assert figs["asset_open_risk"]["cap"] == float(use["tests"]["asset_stop_loss"]["limit"] / e)


def test_a_test_switched_off_has_no_limit_and_still_shows_its_use(store):  # noqa: F811
    _fund(store)
    _entry(store, T0 + MIN)
    p = compliance.load(store)
    compliance.save(store, replace(p, asset_open_risk=None), "pm")
    t = supervisor.compliance_use(store)["tests"]["asset_stop_loss"]
    assert t["limit"] is None and t["used"] == {"BTC": D("1.2")}


def test_a_drawdown_and_a_days_loss_are_used_from_the_peak_and_the_days_start(store):  # noqa: F811
    _fund(store)
    _trade(store, "x", -1_100, T0 + 2 * MIN)
    _mark(store, T0 + 3 * MIN)  # book 99,000: 1,000 under the day's start, 1,100 under the peak
    t = supervisor.compliance_use(store)["tests"]
    assert t["fund_day_loss"]["used"] == D(1_000) and t["fund_drawdown"]["used"] == D(1_100)
