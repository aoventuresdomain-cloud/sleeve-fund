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


def test_a_hole_just_before_a_change_counts_on_the_interval_before_it_and_is_listed(tmp_path, capfd):
    start = pd.Timestamp("2024-01-01", tz="UTC")
    eight = [start + 8 * H * k for k in range(15)]
    four = [eight[-1] + 4 * H * k for k in range(1, 31)]
    del eight[-2]  # the last 8-hourly settlement but one is missed, right at the change to 4-hourly
    _store(tmp_path, eight + four)
    end = four[-1].normalize() - pd.Timedelta(days=1)
    c = funding_counts.counts("BINANCE", "BTC/USDT", start, end, tmp_path)
    kept = [t for t in eight + four if t < end + pd.Timedelta(days=1)]
    assert (c["venue_sourced"], c["missing"]) == (len(kept), 1)  # one 8-hourly settlement, not three 4-hourly (CR F1)
    assert c["holes_across_a_change"] == [(eight[-2], eight[-1])]
    funding_counts.main(["--venue", "BINANCE", "--pairs", "BTC/USDT", "--start", "2024-01-01",
                         "--end", f"{end:%Y-%m-%d}", "--root", str(tmp_path)])
    assert "spans an interval change; its missing settlements are counted on the interval before it" in capfd.readouterr().out



def _rates(root: Path, pair: str, rows: list) -> None:
    path = root / "BINANCE" / pair.replace("/", "-") / "funding.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"rates": [[int(t.value // 1_000_000), r] for t, r in rows]}))


def _expected(rows: list, start, until) -> dict:
    """The gate counts re-derived by hand, without the block: the next 4-hour close c with the stamp before c less a
    minute; the window the stamps after c less 90 days and before c less a minute; no value under 80% of its 270
    settlements or when the latest is over 8 hours and a minute back; the percentile the share strictly below."""
    out = {"high": 0, "read": 0, "at_pct": 0, "no_value": 0, "other_read": 0}
    m, day = pd.Timedelta(minutes=1), pd.Timedelta(days=1)
    for t, r in rows:
        if not (start <= t < until) or r < 0.0003:
            continue
        out["high"] += 1
        c = pd.Timestamp(t.date(), tz="UTC")
        while not t < c - m:
            c += 4 * H
        window = [(u, x) for u, x in rows if c - 90 * day < u < c - m]
        if len(window) < 0.8 * 270 or c - window[-1][0] > 8 * H + m:
            out["no_value"] += 1
        elif window[-1][0] != t:
            out["other_read"] += 1
        else:
            out["read"] += 1
            out["at_pct"] += 100.0 * sum(x < r for _, x in window) / len(window) >= 90
    return out


def test_the_short_gate_counts_rates_at_the_line_and_the_blocks_own_reading_at_the_next_close(tmp_path, capfd):
    """A0's short gate (Advisor, 1 Nov): rates at or above 0.0003 per 8 hours, and those the funding_percentile block
    reads at 90 or above at the next 4-hour close. Synthetic, 8-hourly, stamped half a second after the hour."""
    t0 = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(milliseconds=500)
    rates = {t0 + 8 * H * k: 0.0001 + 1e-7 * (k % 7) for k in range(3 * 200)}
    def at(d, n=0):
        return t0 + 8 * H * (3 * d + n)

    rates[at(30)] = 0.0004  # inside the first 90 days, under the window's coverage: no value
    rates[at(120)] = 0.0004  # the window's top: 100
    rates[at(121)] = 0.0003  # exactly at the line: counted, and still above all but day 120's
    rates[at(150)] = -0.0005  # negative: not at the line
    rates[at(160)] = 0.00029  # just under the line
    for k in range(40):  # a run of high rates before day 140, so its 0.0003 and the run's later ones rank low
        rates[at(125) + 8 * H * k] = 0.0006  # days 125 to 138
    rates[at(140)] = 0.0003
    rates[at(170, 1)] = 0.0005  # day 170 08:00, and a stray settlement inside the same 4-hour candle after it,
    rates[at(170, 1) + H] = 0.0001  # so the next close reads the later one
    early = at(190) - pd.Timedelta(seconds=30.5)  # day 190, stamped 30 s early: the 00:00 close can't read it
    del rates[at(190)]
    rates[early] = 0.0005
    rates[at(195)] = 0.06  # beyond the cap: not believed, so neither counted nor in any window
    rows = sorted(rates.items())
    _rates(tmp_path, "BTC/USDT", rows)
    rows = [(u, x) for u, x in rows if abs(x) <= 0.05]  # what the blocks read (funding.believable)
    start, end = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-07-18", tz="UTC")
    g = funding_counts.gate_counts("BINANCE", "BTC/USDT", start, end, tmp_path)
    assert g == _expected(rows, start, end + pd.Timedelta(days=1))
    assert g["high"] == 1 + 1 + 1 + 40 + 1 + 1 + 1 and g["no_value"] == 1 and g["other_read"] == 1
    assert g["read"] == g["high"] - 2 and 2 <= g["at_pct"] < g["read"]
    later = pd.Timestamp("2024-05-02", tz="UTC")  # the window's start is honoured: days 30, 120 and 121 fall out
    late = funding_counts.gate_counts("BINANCE", "BTC/USDT", later, end, tmp_path)
    assert late == _expected(rows, later, end + pd.Timedelta(days=1)) and late["high"] == g["high"] - 3
    assert funding_counts.main(["--venue", "BINANCE", "--pairs", "BTC/USDT,ETH/USDT", "--start", "2024-01-01",
                                "--end", "2024-07-18", "--root", str(tmp_path), "--gate"]) == 2  # ETH: none stored
    out = capfd.readouterr().out
    assert (f"| BTC | {g['high']} | {g['read']} | {g['at_pct']} | {g['no_value']} | {g['other_read']} |") in out
    assert "| ETH | 0 | 0 | 0 | 0 | 0 |" in out
    gate_table = out[out.index("Short gate"):]
    assert "0.0" not in gate_table.split("\n", 3)[-1] and ":" not in gate_table.split("\n", 3)[-1]  # counts only


def test_without_the_gate_flag_the_table_is_unchanged(tmp_path, capfd):
    start = pd.Timestamp("2024-01-01", tz="UTC")
    _store(tmp_path, [start + 8 * H * k for k in range(6)])
    funding_counts.main(["--venue", "BINANCE", "--pairs", "BTC/USDT", "--start", "2024-01-01", "--end", "2024-01-02",
                         "--root", str(tmp_path)])
    assert "Short gate" not in capfd.readouterr().out


def _book(path: Path, entries: list) -> None:
    lines = ["asset,leg,test_day,trade,side,entry_time,entry_signal_values"]
    lines += [f"{a},{a.lower()},yes,{n},short,{t:%Y-%m-%dT%H:%M:%S},fr.pct=95.0; fr.rate={r!r}"
              for n, (a, t, r) in enumerate(entries)]
    path.write_text("\n".join(lines) + "\n")


def test_check_8_passes_when_each_entry_read_the_settlement_before_its_close_and_fails_on_the_closes_own(
        tmp_path, capfd, monkeypatch):
    """Advisor's check 8 (settled funding only): the trade book's fr.rate must be the latest believable settlement
    stamped before the decision close less a minute. A rate stamped at the close itself is the next one's."""
    t0 = pd.Timestamp("2024-01-01", tz="UTC") - pd.Timedelta(seconds=30.5)  # stamped early: still the hour's own
    rows = [(t0 + 8 * H * k, 0.0001 + 1e-6 * k) for k in range(3 * 30)]
    rows[40] = (rows[40][0], 0.06)  # day 13 08:00, beyond the cap: not believed, never a rate a rule reads
    gone = rows.pop(60)  # day 20 00:00 missing: the entry after it can't read day 19 16:00's as the latest due
    _rates(tmp_path / "h", "BTC/USDT", rows)
    rate = dict(rows)
    stamps = [t for t, _ in rows]

    def right(close):
        return [rate[t] for t in stamps if t < close - pd.Timedelta(minutes=1) and abs(rate[t]) <= 0.05][-1]

    closes = [pd.Timestamp("2024-01-05", tz="UTC") + 4 * H * k for k in range(24)]  # half of them 00, 08 or 16
    good = [("BTC", c + pd.Timedelta(minutes=1), right(c)) for c in closes]
    book = tmp_path / "s" / "qra0_x_binance-basket-btc-eth-sol-usdt-store-240m_20261101-165919-basket-trades.csv"
    book.parent.mkdir()
    _book(book, good)
    c = funding_counts.check8("BINANCE", book, tmp_path / "h")
    assert c == {"sampled": 12, "on_settlement": 6, "failed": 0}
    early = [("BTC", c + pd.Timedelta(minutes=1), rate[next(t for t in stamps if t > c - pd.Timedelta(minutes=1))])
             for c in closes if c.hour % 8 == 0]  # each read the settlement at its own close: look-ahead
    off = [g for g in good if (g[1] - pd.Timedelta(minutes=1)).hour % 8]
    _book(book, off + early)
    c = funding_counts.check8("BINANCE", book, tmp_path / "h")
    assert c == {"sampled": 12, "on_settlement": 6, "failed": 6}  # every 00, 08 or 16 one read too early
    _book(book, [("BTC", stamps[40].ceil("4h") + 4 * H + pd.Timedelta(minutes=1), 0.06)])  # read the unbelievable
    assert funding_counts.check8("BINANCE", book, tmp_path / "h") == {"sampled": 1, "on_settlement": 0, "failed": 1}
    _book(book, [("BTC", gone[0].ceil("4h") + 4 * H + pd.Timedelta(minutes=1), rows[59][1])])  # a stale rate read
    assert funding_counts.check8("BINANCE", book, tmp_path / "h") == {"sampled": 1, "on_settlement": 0, "failed": 1}
    _book(book, good)
    monkeypatch.setenv("TEARSHEET_DIR", str(book.parent))
    stem = book.name.removesuffix("-basket-trades.csv")
    args = ["--venue", "BINANCE", "--pairs", "BTC/USDT", "--start", "2024-01-01", "--end", "2024-01-30", "--root",
            str(tmp_path / "h")]
    assert funding_counts.main(args + ["--check8", stem]) == 0
    out = capfd.readouterr().out
    assert "check 8: PASS" in out and "12 entries sampled, 6 on a 00, 08 or 16 UTC close" in out
    tail = out[out.index("check 8 (settled"):]
    assert "0.0" not in tail and "2024" not in tail  # counts only: no rate, no time
    _book(book, off + early)
    assert funding_counts.main(args + ["--check8", stem]) == 1
    assert "check 8: FAIL (6 of 12)" in capfd.readouterr().out
    assert funding_counts.main(args + ["--check8", "../etc/passwd"]) == 2
    assert funding_counts.main(args + ["--check8", stem.replace("240m", "60m")]) == 2  # no such sheet
    assert "isn't in TEARSHEET_DIR" in capfd.readouterr().out
