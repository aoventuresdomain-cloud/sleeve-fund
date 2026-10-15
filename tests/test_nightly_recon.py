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
    store.record_fill("s1", side="SELL", qty=0.1, price=61_000.0, fee=3.0, order_id="o2", trade_id="t2",
                      ts=T0 + timedelta(minutes=2))
    # Another Strategy compared, so the night isn't "nothing was compared" (start_check.check_books: a FAIL)
    store.create_sleeve(name="s2", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, venue="BINANCE")
    store.record_equity("s2", equity=10_000.0, cash=10_000.0, qty=0.0, price=60_000.0, benchmark=10_000.0,
                        ts=T0 + timedelta(minutes=1))
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


def _seeded(tmp_path, gap: float = 0.0, seed: int = 7):
    """A seeded book: three Strategies, each with 20 random fills and a mark of the journal's own figures after them;
    `gap` is added to s2's marked cash and equity."""
    import random

    rnd = random.Random(seed)
    url = f"sqlite:///{tmp_path}/seeded.db"
    store = Store(url)
    for n in ("s1", "s2", "s3"):
        store.create_sleeve(name=n, strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                            starting_balance=10_000, venue="BINANCE")
        held = 0.0
        for i in range(20):
            qty = round(rnd.uniform(0.001, 0.05), 3)
            side = "BUY" if held < qty or rnd.random() < 0.5 else "SELL"
            held += qty if side == "BUY" else -qty
            price = round(rnd.uniform(55_000, 65_000), 1)
            store.record_fill(n, side=side, qty=qty, price=price, fee=round(qty * price * 0.0005, 6), order_id=f"{n}o{i}",
                              trade_id=f"{n}t{i}", ts=T0 + timedelta(seconds=i))
        book = store.journal_book(n, 10_000)
        price = 60_000.0
        cash = float(book["cash"]) + (gap if n == "s2" else 0.0)
        store.record_equity(n, equity=cash + float(book["qty"]) * price, cash=cash, qty=float(book["qty"]), price=price,
                            benchmark=10_000.0, ts=T0 + timedelta(minutes=1))
    store.engine.dispose()
    return url


def test_a_seeded_book_matches_the_journal_to_the_cent(tmp_path):
    ok, _, fails = nightly_recon.run(tmp_path / "recon", _seeded(tmp_path), NIGHT)
    assert ok and not fails
    dashboard = _record(tmp_path / "recon").split("Dashboard against the journal:\n")[1].split("\n\n")[0]
    assert dashboard.count("  PASS  ") == 3


def test_a_one_cent_gap_on_a_seeded_book_fails_and_alerts_though_the_start_check_allows_it(tmp_path):
    """The start check allows a cent per fill (41 cents on 20 fills); the nightly check, a gap under one cent."""
    from sleeve_fund import start_check

    url = _seeded(tmp_path, gap=0.01)
    ok, _, fails = nightly_recon.run(tmp_path / "recon", url, NIGHT)
    assert not ok and len(fails) == 1 and fails[0].startswith("s2: ")
    assert "  PASS  s2: " in start_check.run(url, environ={})[1]
    sent, now = [], [NIGHT]
    fwd = _forwarder(url, tmp_path / "recon2", now, sent)
    fwd.step()
    [ev] = _events(url, "recon_failed")
    assert ev.level == "warning" and "s2: " in ev.message


def test_a_gap_under_a_cent_passes(tmp_path):
    ok, _, fails = nightly_recon.run(tmp_path / "recon", _seeded(tmp_path, gap=0.009), NIGHT)
    assert ok and not fails


def test_a_strategy_withdrawn_with_dust_is_listed_as_known_not_failed(tmp_path):
    from sleeve_fund.store import sleeve_archive_t

    url, store = _journal(tmp_path)
    store.record_fill("s1", side="SELL", qty=0.0999999, price=60_000.0, fee=3.0, order_id="o2", trade_id="t2",
                      ts=T0 + timedelta(minutes=2))
    book = store.journal_book("s1", 10_000)
    store.record_equity("s1", equity=float(book["cash"]) + float(book["qty"]) * 60_000.0, cash=float(book["cash"]),
                        qty=float(book["qty"]), price=60_000.0, benchmark=10_000.0, ts=T0 + timedelta(minutes=3))
    with store.engine.begin() as c:  # put away flat but for the dust, its equity withdrawn (F-C3B-A)
        c.execute(sleeve_archive_t.insert().values(sleeve="s1", archived_at=T0 + timedelta(minutes=4),
                                                   withdrawn=book["cash"]))
    store.engine.dispose()
    ok, _, fails = nightly_recon.run(tmp_path / "recon", url, NIGHT)
    assert ok and not fails
    assert "  NOTE  s1: known, it left the fund's book holding dust of 1e-07 no order can close" in _record(
        tmp_path / "recon")
