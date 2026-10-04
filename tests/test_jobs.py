"""Review round 4: each backtest runs in a process of its own, so an engine abort ends the run and
not the dashboard, and the run's memory goes back to the server."""

import os

import pytest
from fastapi.testclient import TestClient

from sleeve_fund.dashboard.jobs import Jobs
from sleeve_fund.store import Store

AUTH = ("pm", "test-pw")


# Run in the child process, so they live at module level where it can import them.
def _counts(progress, job_id, n):
    for i in range(n):
        progress((i + 1) / n)
    return f"{job_id}:{os.getpid()}"


def _refuses(progress, job_id):
    raise ValueError("capital: between 100 and 1,000,000,000")


def _panics(progress, job_id):
    from nautilus_trader.indicators import SimpleMovingAverage

    sma = SimpleMovingAverage(2000)  # the engine aborts the process past 1,024 bars
    for _ in range(3000):
        sma.update_raw(1.0)


def _finish(jobs, *args):
    job = jobs.submit(*args)
    assert job.done_event.wait(120)
    return job


def test_a_run_reports_progress_and_its_answer_from_its_own_process():
    job = _finish(Jobs(isolate=True), "k1", "count", _counts, 5)
    assert job.status == "done" and job.progress == 1.0
    run_id, pid = job.run_id.split(":")
    assert run_id == job.id and int(pid) != os.getpid()


def test_a_refused_run_says_why():
    job = _finish(Jobs(isolate=True), "k2", "refuse", _refuses)
    assert job.status == "error" and job.error == "capital: between 100 and 1,000,000,000"


def test_an_engine_abort_ends_the_run_not_the_dashboard():
    jobs = Jobs(isolate=True)
    job = _finish(jobs, "k3", "panic", _panics)
    assert job.status == "error" and "engine stopped unexpectedly" in job.error and "unaffected" in job.error
    after = _finish(jobs, "k4", "count", _counts, 2)  # the queue carries on
    assert after.status == "done"


def test_a_backtest_runs_and_saves_from_its_own_process(tmp_path, monkeypatch):
    """End to end through the page, isolated as on the server: the child opens the journal by its
    address and reads minutes from the history store the dashboard points it at."""
    from sleeve_fund import history
    from sleeve_fund.dashboard import app as app_mod
    from test_dashboard import _wavy_minutes

    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    monkeypatch.setenv("BACKTEST_ISOLATE", "1")
    monkeypatch.setenv("HISTORY_DIR", str(tmp_path / "hist"))  # the child reads it when it starts
    monkeypatch.setattr(history, "DEFAULT_ROOT", tmp_path / "hist")
    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    monkeypatch.setattr(app_mod, "BACKTEST_WAIT", 120.0)
    history.HistoryStore(tmp_path / "hist").append("KRAKEN", "ETH/USD", _wavy_minutes(20), cursor="x")
    store = Store(f"sqlite:///{tmp_path}/t.db")
    c = TestClient(app_mod.create_app(store))
    assert c.app.state.jobs.isolate
    r = c.get("/backtest?run=1&instrument=ETH/USD&strategy=trend_filter&p_trend_filter__fast=5"
              "&p_trend_filter__slow=20&bar_spec=1-HOUR-LAST-INTERNAL&risk_profile=aggressive",
              auth=AUTH, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/backtest/"), r.text[-500:]
    run_id = r.headers["location"].rsplit("/", 1)[1]
    assert store.backtest(run_id)["title"].startswith("Trend filter on ETH/USD")
    assert store.orders(f"bt:{run_id}", limit=1000)


@pytest.mark.parametrize("hours", [3, 200 * 24])
def test_thinning_marks_as_they_come_saves_exactly_what_it_did(hours):
    """The run's journal keeps one mark an hour once past 5,000; what it saves is unchanged."""
    from datetime import datetime, timedelta, timezone

    from sleeve_fund.paper.journal import MemoryJournal

    def marks(thin):
        j = MemoryJournal()
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for i in range(hours * 60):
            ts = t0 + timedelta(minutes=i)
            if thin:
                j.record_equity("bt", equity=100 + i, cash=0, qty=1, price=100 + i, benchmark=100, ts=ts)
            else:
                j.equity.append({"ts": ts, "equity": 100 + i, "cash": 0, "qty": 1, "price": 100 + i,
                                 "benchmark": 100})
        return j

    thin, full = marks(True), marks(False)
    assert thin.marks_to_keep() == full.marks_to_keep()
    assert thin.first_equity("bt") == full.first_equity("bt") and thin.last_equity("bt") == full.last_equity("bt")
    if hours > 100:
        assert len(thin.equity) < 5000 + hours + 1 < len(full.equity)


def test_a_saved_long_run_keeps_its_deepest_drawdown():
    """Review round 5, R5-M2: a long intraday run saves one mark a day, so a fall and recovery within a
    day vanished: the page showed -19.27% beside "halted: drawdown 20.0%". The peak and the trough of
    the deepest drawdown are now saved too, so the drawdown on the page is the one the guard saw."""
    from datetime import datetime, timedelta, timezone

    from sleeve_fund.paper.journal import MemoryJournal

    j, t0 = MemoryJournal(), datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i in range(200 * 1440):
        ts = t0 + timedelta(minutes=i)
        eq = 100.0 + i / 1000
        if 150 * 1440 + 600 <= i < 150 * 1440 + 660:  # an hour mid-day: 30% down, then all back
            eq *= 0.7
        j.record_equity("bt", equity=eq, cash=0, qty=1, price=eq, benchmark=100, ts=ts)
    kept = j.marks_to_keep()
    assert len(kept) < 300 and j.max_drawdown() == pytest.approx(0.3, abs=1e-4)
    peak, worst = 0.0, 0.0
    for m in kept:
        peak = max(peak, m["equity"])
        worst = max(worst, 1 - m["equity"] / peak)
    assert worst == pytest.approx(j.max_drawdown(), abs=1e-12)
    assert [m["ts"] for m in kept] == sorted(m["ts"] for m in kept)
