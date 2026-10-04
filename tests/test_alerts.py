"""Review rounds 2 to 5: no outbound alerts and no dead man's switch. The supervisor forwards the
journal's warnings and errors to a webhook and pings an outside uptime monitor."""

from sleeve_fund.alerts import Forwarder, message
from sleeve_fund.store import Store

URL = "https://hooks.example.com/services/T0/B0/token"


def _store():
    store = Store.in_memory()
    store.event(None, "error", "old", "before the supervisor started")  # history isn't resent
    return store


def test_new_warnings_and_errors_go_out_in_one_message():
    store, sent = _store(), []
    fwd = Forwarder(store, environ={"ALERT_WEBHOOK_URL": URL}, send=lambda url, text: sent.append((url, text)))
    fwd.step()
    assert sent == []
    store.event("btc", "warning", "stale_price", "No trade or quote for 5 minutes")
    store.event("btc", "info", "fill", "BUY 0.1")  # info stays in the dashboard
    store.event("bt:123", "error", "risk_halt", "a saved backtest's halt")  # backtests never alert
    store.event(None, "error", "supervisor_error", "boom")
    fwd.step()
    assert len(sent) == 1 and sent[0][0] == URL
    assert sent[0][1] == ("Multi-Strategy Fund: 2 alerts\n[warning] btc: No trade or quote for 5 minutes\n"
                          "[error] system: boom")
    fwd.step()
    assert len(sent) == 1  # nothing new


def test_a_failed_send_is_retried_and_said_once():
    store, sent, down = _store(), [], [True]

    def send(url, text):
        if down[0]:
            raise OSError("unreachable")
        sent.append(text)

    fwd = Forwarder(store, environ={"ALERT_WEBHOOK_URL": URL}, send=send)
    store.event("btc", "error", "risk_halt", "drawdown 20%")
    fwd.step()
    fwd.step()
    fails = [e for e in store.events(limit=50) if e["kind"] == "alert_send_failed"]
    assert len(fails) == 1 and fails[0]["level"] == "info" and "token" not in fails[0]["message"]
    down[0] = False
    fwd.step()
    assert len(sent) == 1 and "drawdown 20%" in sent[0]


def test_the_uptime_ping_goes_every_step_and_nothing_breaks_without_settings():
    store, pings = _store(), []
    fwd = Forwarder(store, environ={"HEALTHCHECK_PING_URL": "https://hc-ping.com/uuid"}, pinger=pings.append,
                    send=lambda *a: (_ for _ in ()).throw(AssertionError("no webhook set")))
    store.event("btc", "error", "x", "y")
    fwd.step()
    fwd.step()
    assert pings == ["https://hc-ping.com/uuid"] * 2
    assert "Alerts are not set up" in fwd.describe() and "hc-ping.com" in fwd.describe()


def test_a_long_burst_is_listed_then_counted():
    events = [{"level": "error", "sleeve": "s", "message": f"m{i}"} for i in range(13)]
    text = message(events)
    assert text.count("\n") == 11 and text.endswith("and 3 more on the Alerts page")


def test_alerts_raised_while_the_supervisor_was_down_still_go_out():
    """Review round 6: the cursor started at the newest event, so anything journaled while the
    supervisor was down was never sent. Each send journals how far it got; a restart picks up there."""
    store, sent = _store(), []
    env = {"ALERT_WEBHOOK_URL": URL}
    fwd = Forwarder(store, environ=env, send=lambda url, text: sent.append(text))
    store.event("btc", "error", "risk_halt", "drawdown 20%")
    fwd.step()
    store.event("eth", "error", "feed_dead", "no market data for 15 minutes")  # supervisor down
    Forwarder(store, environ=env, send=lambda url, text: sent.append(text)).step()
    assert len(sent) == 2 and "feed_dead" not in sent[0] and "no market data" in sent[1]
    assert "drawdown 20%" not in sent[1]  # nothing sent twice


def test_a_late_or_failed_backup_raises_an_alert_once(tmp_path):
    import json
    import os
    import time
    from datetime import timedelta

    from sleeve_fund.store import utcnow

    store, clock = _store(), [utcnow()]
    fwd = Forwarder(store, environ={"BACKUP_DIR": str(tmp_path)}, now=lambda: clock[0])
    fwd.step()
    (warn,) = [e for e in store.events(limit=50) if e["kind"] == "backup_problem"]
    assert warn["level"] == "warning" and "No database backup has been written yet" in warn["message"]
    (tmp_path / "sleeve_fund-a.dump").write_bytes(b"x")
    (tmp_path / "status.json").write_text(json.dumps({"ok": True, "message": "dumped and restored"}))
    clock[0] += timedelta(hours=2)
    fwd.step()
    clock[0] += timedelta(minutes=30)
    (tmp_path / "status.json").write_text(json.dumps({"ok": False, "message": "dump didn't restore: boom"}))
    fwd.step()  # checked hourly: not yet
    clock[0] += timedelta(hours=1)
    fwd.step()
    fwd.step()
    clock[0] += timedelta(hours=1)
    fwd.step()  # the same problem isn't said again
    kinds = [e["message"] for e in store.events(limit=50) if e["kind"] == "backup_problem"]
    assert kinds == ["The last database backup failed: dump didn't restore: boom",
                     "No database backup has been written yet"]  # newest first
    (tmp_path / "status.json").write_text(json.dumps({"ok": True, "message": "ok"}))
    os.utime(tmp_path / "sleeve_fund-a.dump", (time.time() - 30 * 3600,) * 2)
    clock[0] += timedelta(hours=1)
    fwd.step()
    assert store.events(limit=1)[0]["message"].endswith("hours old")  # 30 hours plus the clock moved on


def test_a_stale_backup_warns_once_however_old_it_grows(tmp_path):
    """Review round 7: the stale warning's text changed every hour (its age), so it was said hourly."""
    import json
    import os
    import time
    from datetime import timedelta

    from sleeve_fund.store import utcnow

    (tmp_path / "sleeve_fund-a.dump").write_bytes(b"x")
    (tmp_path / "status.json").write_text(json.dumps({"ok": True, "message": "ok"}))
    os.utime(tmp_path / "sleeve_fund-a.dump", (time.time() - 30 * 3600,) * 2)
    store, clock = _store(), [utcnow()]
    fwd = Forwarder(store, environ={"BACKUP_DIR": str(tmp_path)}, now=lambda: clock[0])
    for _ in range(5):
        fwd.step()
        clock[0] += timedelta(hours=1)
    assert len([e for e in store.events(limit=50) if e["kind"] == "backup_problem"]) == 1


def test_an_unreadable_backup_status_is_a_failure_not_a_pass(tmp_path):
    """Review round 7, R7-B: a status file the backup wrote badly read as no news, so Ops showed the
    backup as fine and nothing alerted."""
    from sleeve_fund import backups
    from sleeve_fund.store import utcnow

    (tmp_path / "sleeve_fund-a.dump").write_bytes(b"x")
    (tmp_path / "status.json").write_bytes(b'{"ok": false, "message": "pg_dump failed: \tbad\xff"')
    kind, text = backups.problem(tmp_path, utcnow())
    assert kind == "failed" and "isn't valid" in text
    assert backups.latest(tmp_path, utcnow())["check"]["ok"] is False


def test_a_send_that_trickles_is_cut_off_and_reported(monkeypatch):
    """Review round 7: the 8 s timeout is per socket read, so a far end sending a byte every few
    seconds held one send for 115 s, with no failure said."""
    import socket
    import threading
    import time

    from sleeve_fund import alerts

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    stop = threading.Event()

    def trickle():
        conn, _ = srv.accept()
        conn.recv(65536)
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n")
        while not stop.is_set():
            try:
                conn.sendall(b"x")
            except OSError:
                break
            time.sleep(0.2)
        conn.close()

    threading.Thread(target=trickle, daemon=True).start()
    monkeypatch.setattr(alerts, "TOTAL", 1.0)
    store = _store()
    fwd = Forwarder(store, environ={"ALERT_WEBHOOK_URL": f"http://127.0.0.1:{srv.getsockname()[1]}/hook"})
    store.event(None, "error", "supervisor_error", "boom")
    t0 = time.monotonic()
    fwd.step()
    took = time.monotonic() - t0
    stop.set()
    srv.close()
    assert took < 3
    (failed,) = [e for e in store.events(limit=10) if e["kind"] == "alert_send_failed"]
    assert "TimeoutError" in failed["message"]
