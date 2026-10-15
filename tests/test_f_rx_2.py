"""F-RX-2 (Advisor ruling, 15 Oct; HoE plan-1300 Drop 3a): repeated rejected exits. Three closing orders in a row
rejected, denied, or confirmed unfilled after no reply ("unfilled_unknown", DA 15 Oct 09:03 UK), on one Strategy row,
with no time window, is one error per episode, one re-alert 30 minutes on, and one info line when a closing order on
that row goes through again. While an episode is open no Strategy in the fund opens or adds on that instrument; exits
and stops still go out. Alert only: nothing switches order type. The closing orders are read through QD's F-RX-1
reader (Store.closing_orders), so the closing intents are never hard-coded here."""

from datetime import UTC, datetime, timedelta

import pytest

from sleeve_fund.paper import runtime
from sleeve_fund.store import Store

T0 = datetime(2026, 10, 19, 9, 0, tzinfo=UTC)
MIN = timedelta(minutes=1)


@pytest.fixture
def store(tmp_path):
    s = Store(f"sqlite:///{tmp_path}/j.db")
    for name, inst in (("s1", "BTC/USDT"), ("s2", "BTC/USDT"), ("s3", "ETH/USDT")):
        s.create_sleeve(name=name, strategy="ping_pong", instrument=inst, bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, venue="BINANCE", params={"market": "perp"})
        if name != "s3":  # a closing reject only matters on a row that holds (a flat row's is benign, Advisor A1)
            s.record_fill(name, side="BUY", qty=0.1, price=60_000.0, fee=0.0, order_id=f"{name}-in",
                          trade_id=f"{name}-in", ts=T0 - MIN)
    return s


_n = [0]


def _order(store, sleeve, status, at, intent="exit"):
    _n[0] += 1
    oid = f"o{_n[0]}"
    store.record_order(sleeve, order_id=oid, side="SELL", qty=0.1, intent=intent, reason="r", ts=at)
    if status == "filled":
        store.update_order(oid, status="filled", fill_qty=0.1, fill_px=60_000.0)
    elif status != "submitted":
        store.update_order(oid, status=status, message="venue said no")
    return oid


def _kinds(store, *kinds):
    return [e for e in store.events_after(0, min_level="info", limit=1000) if e["kind"] in kinds]


OPEN, STILL, CLEARED = "exit_rejected_repeatedly", "exit_rejected_still", "exit_rejects_cleared"


def _watch():
    from sleeve_fund.ops_health import ExitRejectWatch

    return ExitRejectWatch()


def test_a_three_closing_rejects_in_a_row_is_one_error_naming_the_instrument_and_no_venue(store):
    w = _watch()
    _order(store, "s1", "rejected", T0)
    _order(store, "s1", "rejected", T0 + MIN)
    w.check(store, T0 + 2 * MIN)
    assert _kinds(store, OPEN) == []  # two is not an episode
    _order(store, "s1", "rejected", T0 + 2 * MIN)
    w.check(store, T0 + 3 * MIN)
    w.check(store, T0 + 4 * MIN)
    [e] = _kinds(store, OPEN)
    assert e["level"] == "error" and e["sleeve"] == "s1" and "BTC/USDT" in e["message"]
    assert "BINANCE" not in e["message"].upper()


@pytest.mark.parametrize("status", ["denied", "unfilled_unknown"])
def test_b_a_denial_and_a_confirmed_unfilled_order_each_count(store, status):
    w = _watch()
    _order(store, "s1", "rejected", T0)
    _order(store, "s1", status, T0 + MIN)
    _order(store, "s1", "rejected", T0 + 2 * MIN)
    w.check(store, T0 + 3 * MIN)
    assert len(_kinds(store, OPEN)) == 1


def test_c_a_closing_order_that_goes_through_ends_the_streak_and_there_is_no_time_window(store):
    w = _watch()
    _order(store, "s1", "rejected", T0)
    _order(store, "s1", "rejected", T0 + MIN)
    _order(store, "s1", "filled", T0 + 2 * MIN)
    _order(store, "s1", "rejected", T0 + 3 * MIN)
    _order(store, "s1", "rejected", T0 + 4 * MIN)
    w.check(store, T0 + 5 * MIN)
    assert _kinds(store, OPEN) == []
    _order(store, "s1", "rejected", T0 + timedelta(days=3))  # days later: still the third in a row
    w.check(store, T0 + timedelta(days=3, minutes=1))
    assert len(_kinds(store, OPEN)) == 1


def test_d_a_rejected_entry_never_counts_and_each_row_is_judged_on_its_own(store):
    w = _watch()
    for i in range(3):
        _order(store, "s1", "rejected", T0 + i * MIN, intent="entry")
    _order(store, "s1", "rejected", T0 + 3 * MIN)
    _order(store, "s2", "rejected", T0 + 4 * MIN)
    _order(store, "s2", "rejected", T0 + 5 * MIN)
    w.check(store, T0 + 6 * MIN)
    assert _kinds(store, OPEN) == []


def _episode(store, w, at=T0):
    for i in range(3):
        _order(store, "s1", "rejected", at + i * MIN)
    w.check(store, at + 3 * MIN)
    assert len(_kinds(store, OPEN)) == 1


def test_e_one_re_alert_30_minutes_after_the_episode_opened_and_never_again(store):
    w = _watch()
    _episode(store, w)
    w.check(store, T0 + 32 * MIN)
    assert _kinds(store, STILL) == []
    w.check(store, T0 + 33 * MIN)
    w.check(store, T0 + 90 * MIN)
    [e] = _kinds(store, STILL)
    assert e["level"] == "error" and "BTC/USDT" in e["message"]
    assert len(_kinds(store, OPEN)) == 1


def test_f_a_closing_order_that_goes_through_clears_the_episode_once_and_a_new_streak_opens_a_new_one(store):
    w = _watch()
    _episode(store, w)
    _order(store, "s1", "filled", T0 + 10 * MIN)
    w.check(store, T0 + 11 * MIN)
    w.check(store, T0 + 12 * MIN)
    [c] = _kinds(store, CLEARED)
    assert c["level"] == "info" and c["sleeve"] == "s1"
    for i in range(3):
        _order(store, "s1", "rejected", T0 + (20 + i) * MIN)
    w.check(store, T0 + 23 * MIN)
    assert len(_kinds(store, OPEN)) == 2


def test_g_a_late_fill_on_an_unfilled_unknown_order_ends_the_streak(store):
    """DA condition 3: a fill after "unfilled_unknown" still lands, and the streak clears on it."""
    w = _watch()
    _order(store, "s1", "rejected", T0)
    _order(store, "s1", "rejected", T0 + MIN)
    late = _order(store, "s1", "unfilled_unknown", T0 + 2 * MIN)
    w.check(store, T0 + 3 * MIN)
    assert len(_kinds(store, OPEN)) == 1
    store.update_order(late, fill_qty=0.1, fill_px=60_000.0)
    w.check(store, T0 + 4 * MIN)
    assert len(_kinds(store, CLEARED)) == 1


def test_h_a_restart_neither_opens_nor_re_alerts_an_episode_the_journal_already_has(store):
    _episode(store, _watch())
    w = _watch()
    w.check(store, T0 + 4 * MIN)
    assert len(_kinds(store, OPEN)) == 1
    w.check(store, T0 + 40 * MIN)
    w2 = _watch()
    w2.check(store, T0 + 41 * MIN)
    assert len(_kinds(store, STILL)) == 1


def test_i_while_an_episode_is_open_no_strategy_opens_on_that_instrument_and_it_lifts_when_cleared(store):
    w = _watch()
    _episode(store, w)
    now = T0 + 4 * MIN
    why = runtime.exit_reject_block(store, "BTC/USDT", now)
    assert why is not None and why.code == "exit_rejects"
    assert why.startswith(f"{runtime.LABELS['exit_rejects']}: No new entries on BTC/USDT: closing orders were "
                          "rejected 3 times in a row on s1;")
    for name in ("s1", "s2"):  # fund-wide on the instrument, not only the row that was rejected
        blocked, said = runtime.entry_blocked(store, name, now)
        assert blocked and "exit_rejects" in said.codes
    assert "exit_rejects" not in (runtime.entry_blocked(store, "s3", now)[1] or runtime.Why("", "x")).codes
    _order(store, "s1", "filled", T0 + 5 * MIN)
    w.check(store, T0 + 6 * MIN)
    assert runtime.exit_reject_block(store, "BTC/USDT", T0 + 6 * MIN) is None


def test_j_a_journal_that_cant_be_read_refuses_entries_on_the_instrument(store, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("journal busy")

    monkeypatch.setattr(store, "last_event", broken)
    why = runtime.exit_reject_block(store, "BTC/USDT", T0)
    assert why is not None and why.code == "exit_rejects" and "couldn't be read" in why


def test_k_the_forwarder_runs_the_watch_and_sends_the_episode_in_the_same_minute(store):
    from sleeve_fund.alerts import Forwarder

    sent = []
    fwd = Forwarder(store, environ={"ALERT_WEBHOOK_URL": "https://hooks.example.com/x"},
                    send=lambda url, text: sent.append(text), now=lambda: T0 + 3 * MIN)
    for i in range(3):
        _order(store, "s1", "rejected", T0 + i * MIN)
    fwd.step()
    assert any("[error] s1:" in m and "BTC/USDT" in m for m in sent)


def test_l_the_episode_also_clears_when_the_row_is_flat_however_it_got_there(store):
    """Advisor F-RX-2: "cleared by closing fill or flat" (a liquidation's fill, which is not a closing order we send)."""
    w = _watch()
    _episode(store, w, at=T0 + MIN)
    w.check(store, T0 + 5 * MIN)
    assert _kinds(store, CLEARED) == []  # still holding
    store.record_fill("s1", side="SELL", qty=0.1, price=50_000.0, fee=0.0, order_id="liq1", trade_id="tl1",
                      ts=T0 + 6 * MIN)  # liquidated: no closing order of ours went through
    w.check(store, T0 + 7 * MIN)
    [c] = _kinds(store, CLEARED)
    assert "flat" in c["message"]


def test_m_closing_rejects_on_a_flat_row_never_open_an_episode(store):
    """Advisor (A1 ruling): a reduce-only reject on a flat row is benign and never counts to F-RX-2."""
    w = _watch()
    for i in range(3):
        _order(store, "s3", "rejected", T0 + i * MIN)
    w.check(store, T0 + 3 * MIN)
    assert _kinds(store, OPEN) == []
