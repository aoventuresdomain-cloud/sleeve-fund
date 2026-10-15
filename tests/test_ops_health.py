"""A15 ops-health minimum (G2, Drop 3; Done when in hoe/bgl-plan.md under "### A15"). Red first: each cell names one
Done-when line. What main already has (the forwarder, the uptime ping, the per-Strategy stale-price hold and the
heartbeat_stale restart) is pinned here as it stands, so A15 builds on it and never round it."""

from datetime import UTC, datetime, timedelta

import pytest

from sleeve_fund import start_check
from sleeve_fund.alerts import Forwarder
from sleeve_fund.store import Store

T0 = datetime(2026, 10, 19, 6, 0, tzinfo=UTC)  # 07:00 UK (BST)
HOOK = "https://hooks.example.com/services/T0/B0/token"
PING = "https://hc-ping.example.com/uuid-token"


def _kinds(store, kind):
    return [e for e in store.events_after(0, min_level="info", limit=1000) if e["kind"] == kind]


# ---- 1. Engine heartbeat ---------------------------------------------------------------------------------------

def test_1a_the_start_check_fails_with_no_alert_destination_or_no_uptime_ping():
    env = {"HEALTHCHECK_PING_URL": PING}
    sec = start_check.check_alerts(env)
    assert sec.failed and any("ALERT_WEBHOOK_URL" in w for s, w in sec.lines if s == start_check.FAIL)
    sec = start_check.check_alerts({"ALERT_WEBHOOK_URL": HOOK})
    assert sec.failed and any("HEALTHCHECK_PING_URL" in w for s, w in sec.lines if s == start_check.FAIL)


def test_1b_with_both_set_it_passes_and_names_the_hosts_never_the_token():
    sec = start_check.check_alerts({"ALERT_WEBHOOK_URL": HOOK, "HEALTHCHECK_PING_URL": PING})
    assert not sec.failed
    page = "\n".join(w for _, w in sec.lines)
    assert "hooks.example.com" in page and "hc-ping.example.com" in page
    assert "token" not in page and "uuid" not in page


def test_1c_a_stale_heartbeat_restart_reaches_the_pm_chat_as_an_error():
    """Pinned as main has it: the supervisor journals heartbeat_stale at error, and the forwarder sends it."""
    store, sent = Store.in_memory(), []
    fwd = Forwarder(store, environ={"ALERT_WEBHOOK_URL": HOOK}, send=lambda url, text: sent.append(text))
    store.event("s1", "error", "heartbeat_stale", "no heartbeat for 3 minutes; restarting")
    fwd.step()
    assert len(sent) == 1 and "[error] s1: no heartbeat for 3 minutes; restarting" in sent[0]


def test_1d_the_uptime_ping_goes_every_step_even_when_the_alert_send_fails():
    store, pings = Store.in_memory(), []

    def down(url, text):
        raise OSError("no route")

    fwd = Forwarder(store, environ={"ALERT_WEBHOOK_URL": HOOK, "HEALTHCHECK_PING_URL": PING}, send=down,
                    pinger=pings.append)
    store.event(None, "error", "x", "boom")
    fwd.step()
    fwd.step()
    assert pings == [PING, PING]


# ---- 2. Data staleness per hub ---------------------------------------------------------------------------------

def test_2a_a_quiet_instrument_on_a_hub_warns_once_per_episode_and_says_when_it_is_back():
    from sleeve_fund.ops_health import HubStaleness

    store = Store.in_memory()
    watch = HubStaleness("hub-1")
    watch.seen("BTC/USDT", T0)
    watch.seen("ETH/USDT", T0)
    watch.check(store, T0 + timedelta(minutes=4, seconds=59))
    assert _kinds(store, "hub_data_stale") == []
    for _ in range(3):  # every check while it stays quiet: still one warning
        watch.seen("ETH/USDT", T0 + timedelta(minutes=5))
        watch.check(store, T0 + timedelta(minutes=5, seconds=1))
    [w] = _kinds(store, "hub_data_stale")
    assert w["level"] == "warning" and "BTC/USDT" in w["message"] and "ETH" not in w["message"]
    assert "5 minutes" in w["message"]
    watch.seen("BTC/USDT", T0 + timedelta(minutes=6))
    watch.check(store, T0 + timedelta(minutes=6))
    [b] = _kinds(store, "hub_data_back")
    assert b["level"] == "info" and "BTC/USDT" in b["message"]
    watch.check(store, T0 + timedelta(minutes=12))  # a second episode warns again
    assert len(_kinds(store, "hub_data_stale")) == 3  # BTC again, and ETH now quiet too


def test_2b_an_instrument_the_hub_has_never_seen_is_not_stale_until_it_has_been_asked_for_5_minutes():
    from sleeve_fund.ops_health import HubStaleness

    store = Store.in_memory()
    watch = HubStaleness("hub-1")
    watch.want("SOL/USDT", T0)
    watch.check(store, T0 + timedelta(minutes=4))
    assert _kinds(store, "hub_data_stale") == []
    watch.check(store, T0 + timedelta(minutes=5))
    assert "SOL/USDT" in _kinds(store, "hub_data_stale")[0]["message"]


def test_2c_the_hub_staleness_message_names_no_venue():
    from sleeve_fund.ops_health import HubStaleness
    from sleeve_fund.wording import no_venues

    store = Store.in_memory()
    watch = HubStaleness("BINANCE")
    watch.seen("BTC/USDT", T0)
    watch.check(store, T0 + timedelta(minutes=5))
    [w] = _kinds(store, "hub_data_stale")
    assert w["message"] == no_venues(w["message"]) and "BINANCE" not in w["message"].upper()


# ---- 3. Open position without a stop ---------------------------------------------------------------------------

def _holder(tmp_path, *, params, stop_frac=0.01, side="BUY", qty=0.1, mark=60_000.0, risk_profile="balanced"):
    store = Store(f"sqlite:///{tmp_path}/j.db")
    store.create_sleeve(name="s1", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, venue="BINANCE", params={"market": "perp", **params},
                        risk_profile=risk_profile)
    store.record_order("s1", order_id="e1", side=side, qty=qty, intent="entry", reason="r",
                       signal={} if stop_frac is None else {"stop_frac": stop_frac}, ts=T0)
    store.record_fill("s1", side=side, qty=qty, price=60_000.0, fee=0.0, order_id="e1", trade_id="t1", ts=T0)
    q = qty if side == "BUY" else -qty
    store.record_equity("s1", equity=10_000.0 + q * (mark - 60_000.0), cash=10_000.0 - q * 60_000.0, qty=q,
                        price=mark, benchmark=10_000.0, ts=T0 + timedelta(minutes=1))
    return store


def _no_stop_errors(store):
    return _kinds(store, "position_without_stop")


@pytest.mark.parametrize("case", ["missing", "gapped"])
def test_3a_a_position_with_no_live_stop_errors_once_after_60s(tmp_path, case):
    from sleeve_fund.ops_health import NoStopWatch

    store = (_holder(tmp_path, params={"stop_loss": 0.01}, stop_frac=None) if case == "missing"
             else _holder(tmp_path, params={"stop_loss": 0.01}, mark=59_000.0))  # 1% stop at 59,400: gone through
    watch = NoStopWatch()
    watch.check(store, T0 + timedelta(minutes=1))
    watch.check(store, T0 + timedelta(minutes=1, seconds=59))  # Advisor A15-OPS: 60 s, absolute
    assert _no_stop_errors(store) == []
    for s in (0, 30, 90):
        watch.check(store, T0 + timedelta(minutes=2, seconds=s))
    [e] = _no_stop_errors(store)
    assert e["level"] == "error" and e["sleeve"] == "s1"


def test_3b_a_position_with_its_stop_in_place_never_alerts(tmp_path):
    from sleeve_fund.ops_health import NoStopWatch

    store = _holder(tmp_path, params={"stop_loss": 0.01})
    watch = NoStopWatch()
    for m in range(1, 10):
        watch.check(store, T0 + timedelta(minutes=m))
    assert _no_stop_errors(store) == []


def test_3c_a_model_that_places_no_stop_is_exempt_at_1x(tmp_path):
    from sleeve_fund.ops_health import NoStopWatch

    store = _holder(tmp_path, params={}, stop_frac=None, risk_profile="conservative")  # places no stop, held to 1x
    watch = NoStopWatch()
    for m in range(1, 10):
        watch.check(store, T0 + timedelta(minutes=m))
    assert _no_stop_errors(store) == []


def test_3c2_the_same_model_above_1x_alerts(tmp_path):
    """HoQA: the stopless exemption is for 1x only. Paper refuses to start such a Model (strategies.check_perp_stop),
    so this is a position that got there another way (a profile edited under it)."""
    from sleeve_fund.ops_health import NoStopWatch

    store = _holder(tmp_path, params={}, stop_frac=None, risk_profile="balanced")  # 2x
    watch = NoStopWatch()
    watch.check(store, T0 + timedelta(minutes=1))
    watch.check(store, T0 + timedelta(minutes=2))
    assert len(_no_stop_errors(store)) == 1


def test_3d_the_episode_ends_when_the_stop_is_back_or_the_position_is_flat_and_a_new_one_alerts_again(tmp_path):
    from sleeve_fund.ops_health import NoStopWatch

    store = _holder(tmp_path, params={"stop_loss": 0.01}, stop_frac=None)
    watch = NoStopWatch()
    watch.check(store, T0 + timedelta(minutes=1))
    watch.check(store, T0 + timedelta(minutes=3))
    store.set_exit_plan("s1", "e1", kind="restart", stop_frac=0.01, tp_frac=None, ts=T0 + timedelta(minutes=4))
    watch.check(store, T0 + timedelta(minutes=4))
    [back] = _kinds(store, "position_stop_back")
    assert back["sleeve"] == "s1" and back["level"] == "info"
    store.set_exit_plan("s1", "e1", kind="restart", stop_frac=None, tp_frac=None, ts=T0 + timedelta(minutes=5))
    watch.check(store, T0 + timedelta(minutes=5))
    watch.check(store, T0 + timedelta(minutes=7))
    assert len(_no_stop_errors(store)) == 2


def test_3e_a_stopped_strategy_still_holding_is_watched(tmp_path):
    from sleeve_fund.ops_health import NoStopWatch

    store = _holder(tmp_path, params={"stop_loss": 0.01}, stop_frac=None)
    store.set_desired_state("s1", "stopped")
    watch = NoStopWatch()
    watch.check(store, T0 + timedelta(minutes=1))
    watch.check(store, T0 + timedelta(minutes=5))
    # Stopped but still in the book and still holding: that is the case the alert exists for.
    assert len(_no_stop_errors(store)) == 1


# ---- 4. Daily digest -------------------------------------------------------------------------------------------

def _two_strategies():
    store = Store.in_memory()
    for name, out in (("s1", 62_500.0), ("s2", 59_000.0)):  # +250 and -100 on a round trip of 0.1, no fees
        store.create_sleeve(name=name, strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                            starting_balance=10_000, venue="BINANCE", params={"market": "perp"})
        store.record_fill(name, side="BUY", qty=0.1, price=60_000.0, fee=0.0, order_id=f"{name}e", trade_id=f"{name}a",
                          ts=T0 - timedelta(hours=2))
        store.record_fill(name, side="SELL", qty=0.1, price=out, fee=0.0, order_id=f"{name}x", trade_id=f"{name}b",
                          ts=T0 - timedelta(hours=1))
        eq = 10_000.0 + 0.1 * (out - 60_000.0)
        store.record_equity(name, equity=eq, cash=eq, qty=0.0, price=out, benchmark=10_000.0,
                            ts=T0 - timedelta(hours=1))
    store.event("s1", "warning", "stale_price", "No trade or quote for 5 minutes", ts=T0 - timedelta(hours=3))
    store.event(None, "error", "supervisor_error", "boom", ts=T0 - timedelta(hours=2))
    store.event(None, "error", "old", "two days ago", ts=T0 - timedelta(days=2))
    return store


def test_4a_the_digest_shows_equity_each_strategy_open_positions_alert_counts_and_checks_and_no_venue():
    from sleeve_fund.ops_health import digest
    from sleeve_fund.wording import no_venues

    text = digest(_two_strategies(), T0 + timedelta(hours=1))
    assert text.startswith("Multi-Strategy Fund daily digest")
    assert "20,150.00" in text  # fund equity
    assert "s1: 10,250.00, 250.00" in text and "s2: 9,900.00, -100.00" in text
    assert "1 warning" in text and "1 error" in text  # since the last digest (none: 24 hours)
    assert "Open positions: none" in text
    for word in ("Backup", "Recon", "Clock"):
        assert word in text
    assert text == no_venues(text) and "BINANCE" not in text.upper()


def test_4b_it_goes_once_a_day_at_08_00_uk_even_on_a_quiet_day_and_a_failed_send_is_retried():
    from sleeve_fund.ops_health import DailyDigest

    store, sent, fail = Store.in_memory(), [], [True]

    def send(url, text):
        if fail[0]:
            raise OSError("down")
        sent.append(text)

    d = DailyDigest(store, environ={"ALERT_WEBHOOK_URL": HOOK}, send=send)
    d.step(T0 + timedelta(minutes=59))  # 07:59 UK
    assert sent == [] and _kinds(store, "digest_send_failed") == []
    d.step(T0 + timedelta(hours=1))  # 08:00 UK: due, and the send fails
    d.step(T0 + timedelta(hours=1, seconds=30))
    d.step(T0 + timedelta(hours=1, seconds=59))
    assert sent == [] and len(_kinds(store, "digest_send_failed")) == 1  # HoQA A15-F1: once per outage, not per step
    fail[0] = False
    d.step(T0 + timedelta(hours=1, minutes=1))
    d.step(T0 + timedelta(hours=5))
    assert len(sent) == 1 and len(_kinds(store, "digest_sent")) == 1
    d.step(T0 + timedelta(days=1, hours=1))  # the next morning
    assert len(sent) == 2


def test_4c_a_restart_after_the_digest_went_does_not_send_it_again_that_day():
    from sleeve_fund.ops_health import DailyDigest

    store, sent = Store.in_memory(), []
    DailyDigest(store, environ={"ALERT_WEBHOOK_URL": HOOK}, send=lambda u, t: sent.append(t)).step(
        T0 + timedelta(hours=1))
    DailyDigest(store, environ={"ALERT_WEBHOOK_URL": HOOK}, send=lambda u, t: sent.append(t)).step(
        T0 + timedelta(hours=2))
    assert len(sent) == 1


def test_4d_08_00_uk_follows_the_clock_change():
    from sleeve_fund.ops_health import DailyDigest

    store, sent = Store.in_memory(), []
    d = DailyDigest(store, environ={"ALERT_WEBHOOK_URL": HOOK}, send=lambda u, t: sent.append(t))
    winter = datetime(2026, 10, 26, 7, 59, tzinfo=UTC)  # GMT after 25 Oct: 07:59 UK
    d.step(winter)
    assert sent == []
    d.step(winter + timedelta(minutes=1))
    assert len(sent) == 1


def _book_marks(store, rows):
    from decimal import Decimal as D

    from sqlalchemy import insert

    from sleeve_fund.store import book_marks_t

    with store.engine.begin() as conn:
        conn.execute(insert(book_marks_t), [{
            "ts": ts, "equity": D(eq), "gross": D(0), "net_max": D(0), "margin_used": D(0), "open_risk": D(0),
            "hwm": D(eq), "reference_equity": D(eq), "day_start_equity": D(eq), "status": "ok", "profile_version": 1}
            for ts, eq in rows])


def test_4e_the_last_completed_trading_day_is_its_own_block_measured_as_the_brake_measures_it():
    """Advisor A15-OPS (b): the T5 day (UTC) as a labelled block, from the book's last mark before each midnight."""
    from sleeve_fund.ops_health import digest

    store = Store.in_memory()
    _book_marks(store, [(datetime(2026, 10, 17, 23, 59, tzinfo=UTC), "100000"),
                        (datetime(2026, 10, 18, 0, 0, 30, tzinfo=UTC), "99000"),  # after midnight: not the start
                        (datetime(2026, 10, 18, 23, 59, tzinfo=UTC), "98500"),
                        (datetime(2026, 10, 19, 0, 1, tzinfo=UTC), "97000")])  # today: not in the block
    text = digest(store, T0 + timedelta(hours=1))
    line = next(x for x in text.splitlines() if x.startswith("Last completed trading day"))
    assert "Sun 18 Oct 01:00 UK to Mon 19 Oct 01:00 UK" in line
    assert "100,000.00 to 98,500.00, -1,500.00 (-1.50%)" in line


def test_4f_the_digest_covers_everything_since_the_last_one():
    from sleeve_fund.ops_health import DailyDigest

    store, sent = Store.in_memory(), []
    d = DailyDigest(store, environ={"ALERT_WEBHOOK_URL": HOOK}, send=lambda u, t: sent.append(t))
    d.step(T0 + timedelta(hours=1))  # Mon 08:00 UK
    store.event(None, "error", "x", "after Monday's digest", ts=T0 + timedelta(hours=2))
    late = T0 + timedelta(days=2, hours=3)  # the server was down on Tuesday morning: Wednesday 10:00 UK
    store.event(None, "warning", "y", "Wednesday", ts=late - timedelta(minutes=5))
    d.step(late)
    assert "covering since Mon 19 Oct 08:00 UK" in sent[1]
    assert "1 warning, 1 error" in sent[1]


@pytest.mark.parametrize("utc, sends", [
    (datetime(2026, 10, 24, 6, 59, tzinfo=UTC), False),  # Sat, BST: 07:59 UK
    (datetime(2026, 10, 24, 7, 0, tzinfo=UTC), True),  # Sat 08:00 UK
    (datetime(2026, 10, 25, 7, 59, tzinfo=UTC), False),  # Sun, clocks went back at 02:00: 07:59 GMT
    (datetime(2026, 10, 25, 8, 0, tzinfo=UTC), True),  # Sun 08:00 UK
], ids=["sat-0759", "sat-0800", "sun-0759", "sun-0800"])
def test_4g_08_00_uk_on_the_day_before_and_the_day_of_the_clock_change(utc, sends):
    from sleeve_fund.ops_health import DailyDigest

    store, sent = Store.in_memory(), []
    DailyDigest(store, environ={"ALERT_WEBHOOK_URL": HOOK}, send=lambda u, t: sent.append(t)).step(utc)
    assert bool(sent) is sends


def test_4h_the_digest_names_no_venue_from_the_venue_list():
    """HoQA: a regex over the whole body against every venue name the wording knows."""
    import re

    from sleeve_fund.ops_health import digest
    from sleeve_fund.wording import _NAMES

    store = _two_strategies()
    store.event("s1", "warning", "w", "BTCUSDT-PERP.BINANCE at api.binance.com stalled", ts=T0)
    assert not re.search(rf"(?i)\b(?:{_NAMES})", digest(store, T0 + timedelta(hours=1)))


def test_4i_the_fund_and_strategy_figures_tie_to_the_journal_to_the_cent():
    """HoQA: the same readers as the start check and the dashboard (supervisor.fund_equity, strategy_cash_pnl)."""
    from sleeve_fund import supervisor
    from sleeve_fund.ops_health import digest

    store = _two_strategies()
    text = digest(store, T0 + timedelta(hours=1))
    assert f"Fund equity: {float(supervisor.fund_equity(store)):,.2f}" in text
    for s in ("s1", "s2"):
        pnl = store.strategy_cash_pnl(s, since=T0 + timedelta(hours=1) - timedelta(days=1), at=T0 + timedelta(hours=1))
        assert f"{float(pnl):,.2f}" in next(x for x in text.splitlines() if x.strip().startswith(f"{s}:"))


def test_2d_a_strategy_on_another_hub_is_unaffected():
    from sleeve_fund.ops_health import HubStaleness

    store = Store.in_memory()
    quiet, live = HubStaleness("hub-1"), HubStaleness("hub-2")
    quiet.seen("BTC/USDT", T0)
    live.seen("BTC/USDT", T0)
    live.seen("BTC/USDT", T0 + timedelta(minutes=5))
    for w in (quiet, live):
        w.check(store, T0 + timedelta(minutes=5, seconds=1))
    [w] = _kinds(store, "hub_data_stale")
    assert "hub-1" in w["message"] and "hub-2" not in w["message"]


def test_2e_the_hub_relay_feeds_the_watch_from_its_ticks_and_journals_on_the_store_thread(monkeypatch):
    """The relay's own wiring: the last trade or quote per instrument, by pair, checked on the minute's build."""
    import time as _time

    from test_hub import BTC, _relay

    from sleeve_fund.hub import relay as relay_mod
    from sleeve_fund.ops_health import HubStaleness

    r, _ = _relay()
    events = []
    r.stale, r.event = HubStaleness("hub-1"), lambda *a, **k: events.append((a, k))
    now = _time.time_ns()
    monkeypatch.setattr(relay_mod.time, "time_ns", lambda: now)
    r._since[BTC] = now - 10 * 60 * 1_000_000_000
    r._seen_ns[BTC] = now - 6 * 60 * 1_000_000_000
    r._check_stale()
    [(args, _kw)] = events
    assert args[:3] == (None, "warning", "hub_data_stale") and "BTC/USDT" in args[3]
    r._seen_ns[BTC] = now
    r._check_stale()
    assert [a[2] for a, _ in events] == ["hub_data_stale", "hub_data_back"]


def test_4j_alerts_are_counted_however_many_the_journal_holds():
    """HoQA A15-M1: a capped read of the oldest events reported 0 errors past 100,000 old warnings. Counted in SQL."""
    from sqlalchemy import insert

    from sleeve_fund.ops_health import alert_counts, digest
    from sleeve_fund.store import events_t

    store = Store.in_memory()
    old = T0 - timedelta(days=3)
    with store.engine.begin() as c:
        c.execute(insert(events_t), [{"sleeve": None, "ts": old, "level": "warning", "kind": "w", "message": "old"}
                                     for _ in range(100_001)])
    store.event(None, "error", "x", "an hour ago", ts=T0)
    store.event("bt:1", "error", "x", "a saved backtest's", ts=T0)
    assert alert_counts(store, T0 - timedelta(days=1), T0 + timedelta(hours=1)) == (0, 1)
    assert "0 warnings, 1 error" in digest(store, T0 + timedelta(hours=1))


# ---- 5. Wired into the forwarder (Drop 3a): what the supervisor runs every minute ------------------------------

def test_5a_the_forwarder_sends_a_position_without_a_stop_in_the_minute_it_is_found(tmp_path):
    store, sent = _holder(tmp_path, params={"stop_loss": 0.01}, stop_frac=None), []
    t = [T0 + timedelta(minutes=1)]
    fwd = Forwarder(store, environ={"ALERT_WEBHOOK_URL": HOOK}, send=lambda url, text: sent.append(text),
                    now=lambda: t[0])
    fwd.step()
    assert not any("without a live stop" in m for m in sent)
    t[0] += timedelta(seconds=61)
    fwd.step()
    assert len(_no_stop_errors(store)) == 1
    assert any("[error] s1: Open position without a live stop" in m for m in sent)  # same step: found, then sent
    t[0] += timedelta(minutes=1)
    fwd.step()
    assert len(_no_stop_errors(store)) == 1  # one per episode


def test_5b_the_forwarder_sends_the_daily_digest_at_08_00_uk_once():
    store, sent = Store.in_memory(), []
    t = [T0 + timedelta(minutes=59)]  # 07:59 UK
    fwd = Forwarder(store, environ={"ALERT_WEBHOOK_URL": HOOK}, send=lambda url, text: sent.append(text),
                    now=lambda: t[0])
    fwd.step()
    assert not any(m.startswith("Multi-Strategy Fund daily digest") for m in sent)
    t[0] = T0 + timedelta(hours=1)  # 08:00 UK
    fwd.step()
    fwd.step()
    assert sum(m.startswith("Multi-Strategy Fund daily digest") for m in sent) == 1


def test_5c_a_failing_ops_check_is_one_error_and_never_stops_the_alerts_or_the_uptime_ping(monkeypatch):
    from sleeve_fund import ops_health

    store, sent, pings = Store.in_memory(), [], []

    def broken(self, store, now):
        raise RuntimeError("journal busy")

    monkeypatch.setattr(ops_health.NoStopWatch, "check", broken)
    fwd = Forwarder(store, environ={"ALERT_WEBHOOK_URL": HOOK, "HEALTHCHECK_PING_URL": PING},
                    send=lambda url, text: sent.append(text), pinger=pings.append)
    for _ in range(3):
        fwd.step()
    store.event("s1", "warning", "stale_price", "No trade or quote for 5 minutes")
    fwd.step()
    assert pings == [PING] * 4
    [e] = _kinds(store, "ops_check_failed")
    assert e["level"] == "error" and "no-stop" in e["message"] and "RuntimeError" in e["message"]
    assert any("No trade or quote for 5 minutes" in m for m in sent)
    monkeypatch.undo()
    fwd.step()
    [back] = _kinds(store, "ops_check_back")
    assert back["level"] == "info"


def test_5d_a_first_deploy_after_08_00_sends_no_digest_then_and_the_first_goes_the_next_morning():
    store, sent = Store.in_memory(), []
    t = [T0 + timedelta(hours=6, minutes=10)]  # 13:10 UK: the deploy
    fwd = Forwarder(store, environ={"ALERT_WEBHOOK_URL": HOOK}, send=lambda url, text: sent.append(text),
                    now=lambda: t[0])
    fwd.step()
    t[0] += timedelta(hours=3)
    fwd.step()
    assert sent == []
    t[0] = T0 + timedelta(days=1, hours=1)  # 08:00 UK the next day
    fwd.step()
    assert len(sent) == 1 and sent[0].startswith("Multi-Strategy Fund daily digest")
