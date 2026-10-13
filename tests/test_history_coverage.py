"""history coverage (Advisor 13 Oct 19:35 UK, PRE): one line per instrument saying whether a test window's 1-minute
bars and settled funding are all held. Read-only."""

import json

import pandas as pd

from sleeve_fund import funding
from sleeve_fund.history import HistoryStore, coverage_line
from test_history import _minutes

W0, W1 = pd.Timestamp("2026-01-01 00:00", tz="UTC"), pd.Timestamp("2026-01-03 00:00", tz="UTC")


def _rates(root, every_hours, skip=()):
    path = funding._path("X", "ABC/USDT", root)
    path.parent.mkdir(parents=True, exist_ok=True)
    t = [W0 - pd.Timedelta(hours=8) + pd.Timedelta(hours=every_hours * i) for i in range(12)]
    path.write_text(json.dumps({"rates": [[int(x.timestamp() * 1000), 0.0001] for i, x in enumerate(t)
                                          if i not in skip]}))


def test_a_window_held_whole_is_covered_and_one_minute_or_settlement_short_is_not(tmp_path):
    store = HistoryStore(tmp_path)
    store.append("X", "ABC/USDT", _minutes("2025-12-31 23:00", 3 * 1440), cursor="c")
    _rates(tmp_path, 8)
    line = coverage_line(store, "X", "ABC/USDT", W0, W1, perpetual=True)
    assert line.startswith("COVERED X ABC/USDT 2026-01-01 00:00..2026-01-03 00:00:"), line
    assert "missing 0 min" in line and "0 hole(s)" in line

    early = coverage_line(store, "X", "ABC/USDT", W0 - pd.Timedelta(days=1), W1, perpetual=True)
    assert early.startswith("GAPS") and "(1380 before the first stored)" in early, early

    _rates(tmp_path, 8, skip={4})
    holed = coverage_line(store, "X", "ABC/USDT", W0, W1, perpetual=True)
    assert holed.startswith("GAPS") and "1 hole(s)" in holed, holed


def test_nothing_stored_is_said_plainly_and_spot_has_no_funding_part(tmp_path):
    store = HistoryStore(tmp_path)
    assert coverage_line(store, "X", "NEW/USDT", W0, W1, perpetual=True) == \
        "GAPS X NEW/USDT 2026-01-01 00:00..2026-01-03 00:00: 1m none stored funding none stored"
    store.append("X", "ABC/USD", _minutes("2025-12-31 23:00", 3 * 1440), cursor="c")
    assert "funding" not in coverage_line(store, "X", "ABC/USD", W0, W1, perpetual=False)
