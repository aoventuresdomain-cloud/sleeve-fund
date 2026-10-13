"""NIGHTLY-RECON: the start check's dashboard-against-the-journal comparison every night, into a dated PASS/FAIL
record, a FAIL alerted at the level stale data uses, and as read-only as the start check."""

import hashlib
import socket
from datetime import timedelta
from pathlib import Path

from sqlalchemy import select

from sleeve_fund import nightly_recon
from sleeve_fund.alerts import Forwarder
from sleeve_fund.store import Store, events_t
from tests.test_start_check import T0, _journal

NIGHT = T0.replace(hour=nightly_recon.RECON_HOUR, minute=5) + timedelta(days=1)  # 14 Oct, after RECON_HOUR


def _record(folder: Path) -> str:
    return (folder / f"recon-{NIGHT:%Y-%m-%d}.txt").read_text()


def _events(url, kind):
    store = Store(url)
    with store.engine.connect() as c:
        rows = c.execute(select(events_t).where(events_t.c.kind == kind)).all()
    store.engine.dispose()
    return rows


def test_a_matching_journal_writes_a_dated_pass_record(tmp_path):
    url, _ = _journal(tmp_path)
    ok, path, fails = nightly_recon.run(tmp_path / "recon", url, NIGHT)
    assert ok and not fails and path.name == "recon-2026-10-14.txt"
    page = _record(tmp_path / "recon")
    assert page.startswith("PASS\nNightly recon 2026-10-14 01:05 UTC")
    assert "  PASS  s1: position 0.1 vs journal 0.1; cash 3,997.00 vs journal 3,997.00" in page


def test_pin_a_dashboard_that_disagrees_with_the_journal_fails(tmp_path):
    url, _ = _journal(tmp_path, mark_qty=0.2)
    ok, _, fails = nightly_recon.run(tmp_path / "recon", url, NIGHT)
    assert not ok and len(fails) == 1 and fails[0].startswith("s1: position 0.2 vs journal 0.1")
    assert _record(tmp_path / "recon").startswith("FAIL\n")


def test_pin_a_fill_newer_than_the_last_mark_is_a_note_not_a_fail(tmp_path):
    url, store = _journal(tmp_path)
    store.create_sleeve(name="s0", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, venue="BINANCE")  # one compared: not every Strategy was skipped
    store.record_equity("s0", equity=10_000.0, cash=10_000.0, qty=0.0, price=60_000.0, benchmark=10_000.0,
                        ts=T0 + timedelta(minutes=1))
    store.record_fill("s1", side="SELL", qty=0.1, price=61_000.0, fee=3.0, order_id="o2", trade_id="t2",
                      ts=T0 + timedelta(minutes=2))
    store.engine.dispose()
    ok, _, fails = nightly_recon.run(tmp_path / "recon", url, NIGHT)
    assert ok and not fails
    assert "  NOTE  s1: a fill landed after its latest mark" in _record(tmp_path / "recon")


def test_pin_a_strategy_with_fills_and_no_mark_fails(tmp_path):
    url, store = _journal(tmp_path)
    store.create_sleeve(name="s2", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, venue="BINANCE")
    store.record_fill("s2", side="BUY", qty=0.1, price=60_000.0, fee=3.0, order_id="o3", trade_id="t3", ts=T0)
    store.engine.dispose()
    ok, _, fails = nightly_recon.run(tmp_path / "recon", url, NIGHT)
    assert not ok and fails == ["s2: the journal has 1 fills and holds 0.1, and the dashboard has no mark to show it"]


def test_a_comparison_that_cannot_run_is_recorded_as_a_fail(tmp_path):
    ok, _, fails = nightly_recon.run(tmp_path / "recon", f"sqlite:///{tmp_path}/missing.db", NIGHT)
    assert not ok and fails == ["the comparison couldn't run: FileNotFoundError"]
    assert _record(tmp_path / "recon").startswith("FAIL\n")


def test_it_is_strictly_read_only_no_writes_and_no_network(tmp_path, monkeypatch):
    """As the start check's proof: the journal's bytes are unchanged and no socket is opened."""
    url, _ = _journal(tmp_path, mark_qty=0.2)  # a FAIL too: the failing path writes nothing either
    path = url[len("sqlite:///"):]

    def digest():
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    before = digest()

    def no_network(*a, **k):
        raise AssertionError("the nightly recon opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    ok, _, _ = nightly_recon.run(tmp_path / "recon", url, NIGHT)
    assert not ok
    assert digest() == before


def _forwarder(url, folder, now, sent):
    return Forwarder(Store(url), environ={"RECON_DIR": str(folder), "ALERT_WEBHOOK_URL": "https://hooks.example/x"},
                     send=lambda u, text: sent.append(text), now=lambda: now[0])


def test_a_fail_raises_a_warning_on_the_alert_path_stale_data_uses(tmp_path):
    url, _ = _journal(tmp_path, mark_cash=3_990.0)
    sent, now = [], [NIGHT]
    fwd = _forwarder(url, tmp_path / "recon", now, sent)
    fwd.step()
    [ev] = _events(url, "recon_failed")
    assert ev.level == "warning"  # stale_price's level: Forwarder sends warnings and errors
    assert "nightly recon FAILED (recon-2026-10-14.txt): s1: position 0.1" in ev.message
    fwd.step()  # the event went out on the next send
    assert any("nightly recon FAILED" in t for t in sent)


def test_it_runs_once_a_night_after_recon_hour_and_a_pass_is_info(tmp_path):
    url, _ = _journal(tmp_path)
    sent, now = [], [NIGHT.replace(hour=0)]
    fwd = _forwarder(url, tmp_path / "recon", now, sent)
    fwd.check_recon()
    assert not (tmp_path / "recon").exists()  # before RECON_HOUR
    now[0] = NIGHT
    fwd.check_recon()
    now[0] = NIGHT + timedelta(hours=5)
    fwd.check_recon()
    Forwarder(Store(url), environ={"RECON_DIR": str(tmp_path / "recon")}, now=lambda: now[0]).check_recon()  # restart
    [ev] = _events(url, "recon_passed")
    assert ev.level == "info" and not _events(url, "recon_failed")
    now[0] = NIGHT + timedelta(days=1)
    fwd.check_recon()
    assert len(_events(url, "recon_passed")) == 2
    assert sorted(p.name for p in (tmp_path / "recon").iterdir()) == ["recon-2026-10-14.txt", "recon-2026-10-15.txt"]


def test_no_recon_dir_means_no_recon(tmp_path):
    url, _ = _journal(tmp_path, mark_qty=0.2)
    Forwarder(Store(url), environ={}, now=lambda: NIGHT).check_recon()
    assert not _events(url, "recon_failed")


def test_a_record_that_cannot_be_written_warns_once_a_night(tmp_path):
    url, _ = _journal(tmp_path)
    (tmp_path / "recon").write_text("not a folder")
    now = [NIGHT]
    fwd = Forwarder(Store(url), environ={"RECON_DIR": str(tmp_path / "recon")}, now=lambda: now[0])
    fwd.check_recon()
    now[0] = NIGHT + timedelta(minutes=1)
    fwd.check_recon()
    [ev] = _events(url, "recon_failed")
    assert ev.level == "warning" and "couldn't write its record" in ev.message
