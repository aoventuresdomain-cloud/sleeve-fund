"""QA regression cells: #151 (DA-10) open interest and funding collection, read point-in-time.

Ported from /mnt/project-files/sleeve-fund/quant-review/v2-p1/degraded-155-scripts/test_funding_155_qa.py (QA's copy
of open-interest-151-scripts/test_open_interest_151_qa.py for the #155 round). The findings these cells pinned as
strict xfails (P1-O3 to P1-O8) are fixed, so the marks are gone; each cell is now a plain regression test. Assertions,
expected values and set-up are QA's, unchanged. Synthetic data only, no venue is called.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pytest

from sleeve_fund import funding, history
try:
    from sleeve_fund import open_interest as oi  # #151 only (QA's copy kept the import optional for older trees)
except ImportError:
    oi = None
from sleeve_fund.history import HistoryStore

V, P = "BINANCE", "BTC/USDT"
MIN = 60_000  # ms


def ms(s: str) -> int:
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)


def utc(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def put(root, rows, series="open_interest"):
    """Write a series file as the collector does: [stamp ms, *values, first_seen ms, backfill]."""
    p = oi._path(V, P, root, series)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"snapshots": rows}))


def live_rows(start: str, n: int, lag_min: float = 3.0, **kw):
    t0 = ms(start)
    return [[t0 + i * 5 * MIN, 100.0 + i, 1e6 + i, t0 + i * 5 * MIN + int(lag_min * MIN), 0] for i in range(n)]


T0 = ms("2026-10-01")


# =============================================================================================================
# 1. Point in time
# =============================================================================================================

def test_the_lag_is_the_publication_lag_not_an_outage(tmp_path):
    """30 days kept as published (3 min lag) except a 48-hour outage, whose snapshots were fetched when the
    collector came back (backfill=0, first_seen = the catch-up). Publication lag is still 3 minutes.

    Source: /mnt/project-files/sleeve-fund/quant-review/v2-p1/degraded-155-scripts/test_funding_155_qa.py; finding
    P1-O4."""
    t0, rows = ms("2026-09-01"), []
    o0, o1 = 20 * 288, 20 * 288 + 48 * 12
    for i in range(30 * 288):
        st = t0 + i * 5 * MIN
        rows.append([st, 100.0 + i, 1.0, t0 + o1 * 5 * MIN + 2 * MIN if o0 <= i < o1 else st + 3 * MIN, 0])
    put(tmp_path, rows)
    lag = oi.lag(V, P, tmp_path)
    print(f"\nP1-O4 evidence: p95 lag after a 48 h outage in 30 days = {lag}")
    assert lag <= pd.Timedelta(minutes=5)
    # what it does to reads: at a quiet time well after the outage, the newest known snapshot is 12 h old
    at = utc("2026-09-28 12:00")
    got = oi.as_of(V, P, at, tmp_path)
    newest_stamp = oi.snapshots(V, P, tmp_path).query("contracts == @got.contracts").index[0]
    assert at - newest_stamp <= pd.Timedelta(minutes=10), at - newest_stamp


def test_as_of_never_returns_a_live_snapshot_first_seen_after_t(tmp_path):
    """as_of(t) never returns a live snapshot first seen after t (the p95 tail, or a hole filled after an outage).

    Source: /mnt/project-files/sleeve-fund/quant-review/v2-p1/degraded-155-scripts/test_funding_155_qa.py; finding
    P1-O6."""
    rows = live_rows("2026-10-01", 288, lag_min=2)
    rows[150][3] = rows[150][0] + 60 * MIN  # the venue was an hour late with this one
    put(tmp_path, rows)
    at = utc("2026-10-01") + pd.Timedelta(minutes=750) + oi.lag(V, P, tmp_path)
    got = oi.as_of(V, P, at, tmp_path)
    seen = oi.snapshots(V, P, tmp_path).query("contracts == @got.contracts")["first_seen"].iloc[0]
    print(f"\nP1-O6 evidence: as_of({at}) returned contracts {got.contracts}, first seen {seen}")
    assert seen <= at


def test_a_nan_snapshot_is_not_served(tmp_path):
    """A NaN value from the venue is neither stored nor served by as_of as if it were known.

    Source: /mnt/project-files/sleeve-fund/quant-review/v2-p1/degraded-155-scripts/test_funding_155_qa.py; finding
    P1-O8."""
    t0 = ms("2026-10-01")
    loader = lambda pair, start: [(t0, float("nan"), 1e6), (t0 + 5 * MIN, 101.0, 1e6)] if start <= t0 else []  # noqa: E731
    oi.refresh(V, P, root=tmp_path, loader=loader)
    got = oi.as_of(V, P, utc("2026-10-01 00:24"), tmp_path)  # 20 min fallback lag: only the NaN one known
    print(f"\nP1-O8 evidence: as_of returned {got.tolist() if got is not None else None}")
    assert got is None or not np.isnan(got.values.astype(float)).any()


# =============================================================================================================
# 3. Week-old alert
# =============================================================================================================

def test_the_alert_is_keyed_on_first_seen(tmp_path):
    """Kept 1 day ago, but the newest snapshot it fetched was stamped 8 days ago (a catch-up that stopped short).

    Source: /mnt/project-files/sleeve-fund/quant-review/v2-p1/degraded-155-scripts/test_funding_155_qa.py; finding
    P1-O7."""
    now = utc("2026-10-06 12:00")
    st = ms("2026-09-28 12:00")
    put(tmp_path, [[st, 1.0, 1.0, ms("2026-10-05 12:00"), 0]])
    assert oi.at_risk(V, P, tmp_path, now=now) is None


def test_a_week_old_series_is_raised_even_when_the_price_refresh_fails(tmp_path, monkeypatch):
    """When the instrument's price refresh fails, the run loop still refreshes open interest and runs the week-old check.

    Source: /mnt/project-files/sleeve-fund/quant-review/v2-p1/degraded-155-scripts/test_funding_155_qa.py; finding
    P1-O5."""
    put(tmp_path, live_rows("2026-09-01", 10))  # newest kept a month ago
    raised = []

    class Stop(Exception):
        pass

    monkeypatch.setattr(history, "_pairs_in_use", lambda v, store=None: [(P, None)])
    monkeypatch.setattr(history, "refresh", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("venue 503")))
    monkeypatch.setattr(oi, "refresh", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("venue 503")))
    monkeypatch.setattr(funding, "refresh", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("venue 503")))
    monkeypatch.setattr(history, "_warn_at_risk", lambda problem, store=None: raised.append(problem))
    monkeypatch.setattr(time, "sleep", lambda s: (_ for _ in ()).throw(Stop()))
    with pytest.raises(Stop):
        history.main(["--root", str(tmp_path), "run", "--venue", "binance"])
    print(f"\nP1-O5 evidence: alerts raised in one pass with the price refresh failing: {raised}")
    assert any(p and "open interest last kept" in p for p in raised)


# =============================================================================================================
# 4. Durable, locked writes
# =============================================================================================================

def test_an_open_interest_conflict_does_not_break_the_price_series(tmp_path):
    """An open interest conflict is not written to the price series' provenance.jsonl, so the hub's bar write and the
    parity report don't raise KeyError, and the refill's new minutes are served.

    Source: /mnt/project-files/sleeve-fund/quant-review/v2-p1/degraded-155-scripts/test_funding_155_qa.py; finding
    P1-O3."""
    from sleeve_fund import parity

    oi.refresh(V, P, root=tmp_path, loader=lambda pair, start: [(T0, 100.0, 6e6)] if start <= T0 else [])
    oi.refresh(V, P, root=tmp_path, loader=lambda pair, start: [(T0, 101.0, 6.1e6)])  # the venue revised it
    s = HistoryStore(tmp_path)
    t = utc("2026-10-05 12:00").value
    s.append_bars(V, P, [(t + i * 60 * 10**9, 100.0, 100.5, 99.5, 100.0, 1.0) for i in range(3)], "live")
    refill = [(t + i * 60 * 10**9, 100.0, 100.5, 99.5, 100.2 if i == 1 else 100.0, 1.0) for i in range(5)]
    errors = []
    try:
        s.append_bars(V, P, refill, "refill")  # one differing bar, two new ones
    except KeyError as exc:
        errors.append(f"append_bars: KeyError {exc}")
    one = s.read(V, P, 1)
    try:
        parity.compare(P, one, one, utc("2026-10-05"), utc("2026-10-06"), 0.0, s.provenance(V, P))
    except KeyError as exc:
        errors.append(f"parity.compare: KeyError {exc}")
    kinds = [e.get("kind") for e in s.provenance(V, P)]  # what the dashboard's history badge counts
    print(f"\nP1-O3 evidence: {errors}; newest minute served {one.index[-1]}; provenance kinds {kinds}")
    assert errors == []
    assert one.index[-1] == utc("2026-10-05 12:05")  # the refill's two new minutes are served
