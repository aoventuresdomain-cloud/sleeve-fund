"""GOLDEN-VENUE: the export writes the same bytes for the same candles, refuses a short window, writes nothing to the
store; and the committed fixtures match their manifest exactly, so editing one candle turns this file red."""

import gzip
import json
import shutil

import numpy as np
import pandas as pd
import pytest

import golden_lib
from sleeve_fund.history import GOLDEN_HEADER, HistoryStore, export_csv, main

T0 = pd.Timestamp("2026-01-01", tz="UTC")


def _store(tmp_path, n=60):
    store = HistoryStore(tmp_path / "store")
    idx = pd.date_range(T0, periods=n, freq="1min", tz="UTC")
    c = 100 + np.arange(n) * 0.1  # not exact in binary, so the repr round trip is tested
    store.append("X", "ABC/USD", pd.DataFrame({"open": c, "high": c + 0.25, "low": c - 0.25, "close": c + 0.05,
                                               "volume": 1.5 + np.arange(n)}, index=idx), cursor="c")
    return store


def _window(a, b):
    return T0 + pd.Timedelta(minutes=a), T0 + pd.Timedelta(minutes=b)


def test_export_is_one_row_per_minute_in_the_window_with_values_as_stored(tmp_path):
    store = _store(tmp_path)
    csv = export_csv(store, "X", "ABC/USD", *_window(10, 40))
    lines = csv.splitlines()
    assert lines[0] == GOLDEN_HEADER and len(lines) == 31 and csv.endswith("\n")
    assert lines[1].startswith("2026-01-01T00:10Z,") and lines[-1].startswith("2026-01-01T00:39Z,")
    stored = store.read("X", "ABC/USD", 1, *_window(10, 40))
    back = pd.read_csv(pd.io.common.StringIO(csv), float_precision="round_trip")
    for col in ("open", "high", "low", "close", "volume"):
        assert back[col].tolist() == stored[col].tolist()  # exact, not approximate
    assert (back["degraded"] == 0).all()


def test_the_same_candles_give_the_same_bytes(tmp_path):
    store = _store(tmp_path)
    one = export_csv(store, "X", "ABC/USD", *_window(0, 30))
    assert one == export_csv(HistoryStore(store.root), "X", "ABC/USD", *_window(0, 30))
    assert golden_lib.digest(one.encode()) == golden_lib.digest(export_csv(store, "X", "ABC/USD", *_window(0, 30)).encode())


def test_a_window_with_a_missing_minute_is_refused_not_exported_short(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="minutes missing"):
        export_csv(store, "X", "ABC/USD", *_window(30, 90))  # runs past the stored minutes
    with pytest.raises(ValueError, match="whole minutes"):
        export_csv(store, "X", "ABC/USD", T0 + pd.Timedelta(seconds=30), T0 + pd.Timedelta(minutes=5))
    with pytest.raises(ValueError, match="whole minutes"):
        export_csv(store, "X", "ABC/USD", *_window(5, 5))
    with pytest.raises(KeyError):
        export_csv(store, "X", "NONE/USD", *_window(0, 5))


def test_the_cli_writes_csv_to_stdout_refuses_to_stderr_and_changes_no_file(tmp_path, capfd, monkeypatch):
    store = _store(tmp_path)
    before = {p: p.read_bytes() for p in store.root.rglob("*") if p.is_file()}
    monkeypatch.setattr("sleeve_fund.venues.venue", lambda name=None: type("P", (), {"name": "X"})())
    assert main(["--root", str(store.root), "export", "ABC/USD", "--start", "2026-01-01T00:00",
                 "--end", "2026-01-01T00:05"]) == 0
    assert capfd.readouterr().out == export_csv(store, "X", "ABC/USD", *_window(0, 5))
    assert main(["--root", str(store.root), "export", "ABC/USD", "--start", "2026-01-01T00:30",
                 "--end", "2026-01-01T02:00"]) == 2
    out = capfd.readouterr()
    assert out.out == "" and out.err.startswith("export refused:")
    assert {p: p.read_bytes() for p in store.root.rglob("*") if p.is_file()} == before


def test_the_committed_fixtures_match_their_manifest():
    assert golden_lib.check() == []


def _fixture_dir(tmp_path, store):
    root = tmp_path / "golden"
    root.mkdir()
    raw = export_csv(store, "X", "ABC/USD", *_window(0, 30)).encode()
    (root / "x-abc-usd.csv.gz").write_bytes(gzip.compress(raw, mtime=0))
    (root / "manifest.json").write_text(json.dumps({"header": GOLDEN_HEADER, "files": {
        "x-abc-usd.csv.gz": golden_lib.entry(raw, "X", "ABC/USD", "2026-01-01T00:00", "2026-01-01T00:30")}}))
    return root


def test_editing_one_candle_turns_the_check_red(tmp_path):
    root = _fixture_dir(tmp_path, _store(tmp_path))
    assert golden_lib.check(root) == []
    assert len(golden_lib.load("x-abc-usd.csv.gz", root)) == 30
    path = root / "x-abc-usd.csv.gz"
    lines = gzip.decompress(path.read_bytes()).decode().splitlines(keepends=True)
    cells = lines[15].split(",")
    cells[4] = repr(float(cells[4]) + 0.01)  # one close, one cent
    lines[15] = ",".join(cells)
    path.write_bytes(gzip.compress("".join(lines).encode(), mtime=0))
    problems = golden_lib.check(root)
    assert len(problems) == 1 and "sha256" in problems[0]
    with pytest.raises(ValueError, match="has changed"):
        golden_lib.load("x-abc-usd.csv.gz", root)


def test_a_fixture_added_or_removed_without_the_manifest_turns_the_check_red(tmp_path):
    root = _fixture_dir(tmp_path, _store(tmp_path))
    shutil.copy(root / "x-abc-usd.csv.gz", root / "stray.csv.gz")
    assert golden_lib.check(root) == ["stray.csv.gz: on disk but not in the manifest"]
    (root / "stray.csv.gz").unlink()
    (root / "x-abc-usd.csv.gz").unlink()
    assert golden_lib.check(root) == ["x-abc-usd.csv.gz: in the manifest but not on disk"]


def test_a_dropped_row_turns_the_check_red(tmp_path):
    root = _fixture_dir(tmp_path, _store(tmp_path))
    path = root / "x-abc-usd.csv.gz"
    lines = gzip.decompress(path.read_bytes()).splitlines(keepends=True)
    path.write_bytes(gzip.compress(b"".join(lines[:-1]), mtime=0))
    problems = golden_lib.check(root)
    assert any("sha256" in p for p in problems) and any("29 rows" in p for p in problems)
