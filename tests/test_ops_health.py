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

def _holder(tmp_path, *, params, stop_frac=0.01, side="BUY", qty=0.1, mark=60_000.0):
    store = Store(f"sqlite:///{tmp_path}/j.db")
    store.create_sleeve(name="s1", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, venue="BINANCE", params={"market": "perp", **params})
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
def test_3a_a_position_with_no_live_stop_errors_once_after_one_bar_plus_60s(tmp_path, case):
    from sleeve_fund.ops_health import NoStopWatch

    store = (_holder(tmp_path, params={"stop_loss": 0.01}, stop_frac=None) if case == "missing"
             else _holder(tmp_path, params={"stop_loss": 0.01}, mark=59_000.0))  # 1% stop at 59,400: gone through
    watch = NoStopWatch()
    watch.check(store, T0 + timedelta(minutes=1))
    watch.check(store, T0 + timedelta(minutes=2))  # one 1-minute bar plus 60 s since first seen: not yet over it
    assert _no_stop_errors(store) == []
    for s in (1, 30, 90):
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


def test_3c_a_model_built_stopless_is_exempt(tmp_path):
    from sleeve_fund.ops_health import NoStopWatch

    store = _holder(tmp_path, params={}, stop_frac=None)  # no stop_loss, stop_atr or stop_swing_bars: held to 1x
    watch = NoStopWatch()
    for m in range(1, 10):
        watch.check(store, T0 + timedelta(minutes=m))
    assert _no_stop_errors(store) == []


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
    for name, eq in (("s1", 10_250.0), ("s2", 9_900.0)):
        store.create_sleeve(name=name, strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                            starting_balance=10_000, venue="BINANCE", params={"market": "perp"})
        store.record_equity(name, equity=eq, cash=eq, qty=0.0, price=60_000.0, benchmark=10_000.0,
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
    assert "s1" in text and "s2" in text
    assert "1 warning" in text and "1 error" in text  # the last 24 hours only
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
    assert sent == [] and len(_kinds(store, "digest_send_failed")) == 1
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
