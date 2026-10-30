"""Family A's funding table (pre-registration section 3, Advisor check 7): scripts/funding_counts.py counts the
settlements due over the window, those kept from the venue, those a signal read at a stand-in (none: the funding
blocks read stored rates only) and those missing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import funding_counts  # noqa: E402

H = pd.Timedelta(hours=1)


def _store(root: Path, times: list) -> None:
    path = root / "BINANCE" / "BTC-USDT" / "funding.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"rates": [[int(t.value // 1_000_000), 0.0001] for t in times]}))


def test_a_full_schedule_is_all_venue_sourced(tmp_path):
    start = pd.Timestamp("2024-01-01", tz="UTC")
    _store(tmp_path, [start + 8 * H * k for k in range(3 * 10)])  # ten days, three a day
    c = funding_counts.counts("BINANCE", "BTC/USDT", start, pd.Timestamp("2024-01-10", tz="UTC"), tmp_path)
    assert (c["due"], c["venue_sourced"], c["baseline_filled"], c["missing"]) == (30, 30, 0, 0)


def test_holes_and_uncovered_edges_count_as_missing_and_follow_the_interval(tmp_path):
    start = pd.Timestamp("2024-01-01", tz="UTC")
    eight = [start + 8 * H * k for k in range(3, 3 * 6)]  # the first day not kept; 8-hourly to day 6
    del eight[5:7]  # two settlements missed
    four = [eight[-1] + 4 * H * k for k in range(1, 6 * 3)]  # then 4-hourly for three days
    _store(tmp_path, eight + four)
    c = funding_counts.counts("BINANCE", "BTC/USDT", start, pd.Timestamp("2024-01-10", tz="UTC"), tmp_path)
    kept = len(eight) + len(four)
    after = 8  # the last kept is 9 Jan 12:00; 4-hourly from 16:00 to the window's end, 11 Jan 00:00, is 8 more due
    assert c["venue_sourced"] == kept and c["baseline_filled"] == 0
    assert c["missing"] == 3 + 2 + after and c["due"] == kept + c["missing"]


def test_the_table_prints_and_says_why_baseline_filled_is_nought(tmp_path, capfd):
    start = pd.Timestamp("2024-01-01", tz="UTC")
    _store(tmp_path, [start + 8 * H * k for k in range(6)])
    assert funding_counts.main(["--venue", "BINANCE", "--pairs", "BTC/USDT,ETH/USDT", "--start", "2024-01-01",
                                "--end", "2024-01-02", "--root", str(tmp_path)]) == 2  # ETH has nothing stored
    out = capfd.readouterr().out
    assert "| BTC | 6 | 6 | 0 | 0 |" in out and "| ETH | no rates stored |" in out
    assert "0 by construction" in out
