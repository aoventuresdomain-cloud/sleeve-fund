"""GOLDEN-VENUE: the export writes the same bytes for the same candles, refuses a short window, writes nothing to the
store; and the committed fixtures match their manifest exactly, so editing one candle turns this file red."""

import base64
import gzip
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import golden_lib
from sleeve_fund.history import (GOLDEN_CANON_HEADER, GOLDEN_HEADER, HistoryStore, export_canon_replaced, export_csv,
                                 main)

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


def _replaced(store, minute, stored, offered, source="canon"):
    """One provenance record as the canon backfill writes it; "at" counts up so each record is told apart."""
    path = store._dir("X", "ABC/USD") / "provenance.jsonl"
    n = len(path.read_text().splitlines()) if path.exists() else 0
    with open(path, "a") as f:
        f.write(json.dumps({"kind": "replaced", "at": f"2026-01-02T00:00:{n:02d}+00:00", "source": source,
                            "minute": (T0 + pd.Timedelta(minutes=minute)).isoformat(),
                            "stored": [stored] * 5, "offered": [offered] * 5}) + "\n")


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
    assert (back["missing"] == 0).all() and (back["degraded"] == 0).all()


def test_a_window_with_a_degraded_or_missing_bar_is_not_canonical_and_is_refused(tmp_path, monkeypatch):
    store = _store(tmp_path)
    read = store.read
    for flag, value in (("degraded", True), ("missing", 1)):
        def flagged(*a, flag=flag, value=value, **k):
            df = read(*a, **k)
            df.loc[df.index[3], flag] = value
            return df
        monkeypatch.setattr(store, "read", flagged)
        with pytest.raises(ValueError, match=f"1 minutes {flag} in the window, first 2026-01-01T00:03Z"):
            export_csv(store, "X", "ABC/USD", *_window(0, 10))


def test_the_same_candles_give_the_same_bytes(tmp_path):
    store = _store(tmp_path)
    one = export_csv(store, "X", "ABC/USD", *_window(0, 30))
    assert one == export_csv(HistoryStore(store.root), "X", "ABC/USD", *_window(0, 30))
    again = export_csv(store, "X", "ABC/USD", *_window(0, 30))
    assert golden_lib.digest(one.encode()) == golden_lib.digest(again.encode())


def test_a_window_with_a_missing_minute_is_refused_not_exported_short(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="minutes absent"):
        export_csv(store, "X", "ABC/USD", *_window(30, 90))  # runs past the stored minutes
    with pytest.raises(ValueError, match="whole minutes"):
        export_csv(store, "X", "ABC/USD", T0 + pd.Timedelta(seconds=30), T0 + pd.Timedelta(minutes=5))
    with pytest.raises(ValueError, match="whole minutes"):
        export_csv(store, "X", "ABC/USD", *_window(5, 5))
    with pytest.raises(KeyError):
        export_csv(store, "X", "NONE/USD", *_window(0, 5))


def test_the_cli_writes_csv_to_stdout_refuses_to_stderr_and_changes_no_file(tmp_path, capfd, monkeypatch):
    store = _store(tmp_path)
    _replaced(store, 1, 1.0, 2.0)
    before = {p: p.read_bytes() for p in store.root.rglob("*") if p.is_file()}
    monkeypatch.setattr("sleeve_fund.venues.venue", lambda name=None: type("P", (), {"name": "X"})())
    assert main(["--root", str(store.root), "export", "ABC/USD", "--start", "2026-01-01T00:00",
                 "--end", "2026-01-01T00:05"]) == 0
    assert capfd.readouterr().out == export_csv(store, "X", "ABC/USD", *_window(0, 5))
    assert main(["--root", str(store.root), "export", "ABC/USD", "--start", "2026-01-01T00:30",
                 "--end", "2026-01-01T02:00"]) == 2
    out = capfd.readouterr()
    assert out.out == "" and out.err.startswith("export refused:")
    args = ["--root", str(store.root), "export", "ABC/USD", "--start", "2026-01-01T00:00", "--end", "2026-01-01T00:05",
            "--canon-replaced"]
    assert main([*args, "--expect", "1"]) == 0
    assert capfd.readouterr().out == export_canon_replaced(store, "X", "ABC/USD", *_window(0, 5))
    assert main([*args, "--expect", "2"]) == 2
    assert capfd.readouterr().err.startswith("export refused:")
    assert {p: p.read_bytes() for p in store.root.rglob("*") if p.is_file()} == before


def test_the_committed_fixtures_match_their_manifest():
    assert golden_lib.check() == []


def test_canon_replaced_is_the_latest_canon_record_per_minute_in_the_window(tmp_path):
    store = _store(tmp_path)
    _replaced(store, 2, 1.0, 2.0)
    _replaced(store, 2, 1.5, 2.5)  # a later run over the same minute: this one is kept
    _replaced(store, 4, 1.0, 3.0)
    _replaced(store, 5, 1.0, 9.0, source="live")  # not the canon backfill
    _replaced(store, 40, 1.0, 9.0)  # outside the window
    csv = export_canon_replaced(store, "X", "ABC/USD", *_window(0, 30), expect=2)
    lines = csv.splitlines()
    assert lines[0] == GOLDEN_CANON_HEADER and len(lines) == 3
    assert lines[1] == "2026-01-01T00:02Z,2026-01-02T00:00:01+00:00," + ",".join(["1.5"] * 5 + ["2.5"] * 5)
    assert lines[2].startswith("2026-01-01T00:04Z,") and lines[2].endswith(",3.0")
    with pytest.raises(ValueError, match="2 canon replacements in the window, the canon run reported 3"):
        export_canon_replaced(store, "X", "ABC/USD", *_window(0, 30), expect=3)
    assert export_canon_replaced(store, "X", "ABC/USD", *_window(10, 30)) == GOLDEN_CANON_HEADER + "\n"


def _fixture_dir(tmp_path, store):
    root = tmp_path / "golden"
    root.mkdir()
    _replaced(store, 7, 1.0, 2.0)
    files = {}
    for name, kind, raw in (
            ("x-abc-usd.csv.gz", "candles", export_csv(store, "X", "ABC/USD", *_window(0, 30))),
            ("x-abc-usd-canon.csv.gz", "canon_replaced",
             export_canon_replaced(store, "X", "ABC/USD", *_window(0, 30)))):
        (root / name).write_bytes(gzip.compress(raw.encode(), mtime=0))
        files[name] = golden_lib.entry(raw.encode(), kind, "X", "ABC/USD", "2026-01-01T00:00", "2026-01-01T00:30")
    (root / "manifest.json").write_text(json.dumps({"headers": golden_lib.HEADERS, "files": files}))
    return root


def test_editing_one_candle_turns_the_check_red(tmp_path):
    root = _fixture_dir(tmp_path, _store(tmp_path))
    assert golden_lib.check(root) == []
    assert len(golden_lib.load("x-abc-usd.csv.gz", root)) == 30
    assert len(golden_lib.load("x-abc-usd-canon.csv.gz", root)) == 1
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
    (root / "x-abc-usd-canon.csv.gz").unlink()
    assert golden_lib.check(root) == ["x-abc-usd-canon.csv.gz: in the manifest but not on disk"]


def test_a_dropped_row_turns_the_check_red(tmp_path):
    root = _fixture_dir(tmp_path, _store(tmp_path))
    path = root / "x-abc-usd.csv.gz"
    lines = gzip.decompress(path.read_bytes()).splitlines(keepends=True)
    path.write_bytes(gzip.compress(b"".join(lines[:-1]), mtime=0))
    problems = golden_lib.check(root)
    assert any("sha256" in p for p in problems) and any("29 rows" in p for p in problems)


# PR B: the committed fixtures (HoQA's conditions: a non-empty manifest, read only through load(), signed list).
SIGNED = {  # HoQA golden-venue-manifest.md, 8 Oct 2026: sha256 of each uncompressed CSV
    "binance-btc-usdt-2026-10-06T0000-2026-10-07T2224-canon.csv": "2c25d6873d1e9f4a8e30035a99e91ab0b823039a4b353a3d5e49e57565351eaa",
    "binance-btc-usdt-2026-10-06T0000-2026-10-07T2224.csv": "4552d1167e3fe1fcce721aa63521ae3624e9dd65bce21e915a9d60ec17102f1b",
    "binance-eth-usdt-2026-10-06T0000-2026-10-07T2224-canon.csv": "dd56227edd57cb364825a3bec9b5da72089573aef0b37ae30095cf2f04692d61",
    "binance-eth-usdt-2026-10-06T0000-2026-10-07T2224.csv": "7a635a03b99b386af2cbf3276a1ee245e0839b7d04c5c602dbf7d0b808bfbb8f",
    "binance-sol-usdt-2026-10-06T0000-2026-10-07T2224-canon.csv": "16b81f31ed9a401ce4bcc803c229411ae37cdfde9b34cc264491a69a057870b6",
    "binance-sol-usdt-2026-10-06T0000-2026-10-07T2224.csv": "1b0aa0e6ef42548ab4d165b4be349af48d2b4d16bc87b232fac438351ec097bf",
    "binance-sui-usdt-2026-10-06T0000-2026-10-07T2224-canon.csv": "b6bbdbb5693ee44bc25d195457f74a87ac69a5f95b3e298a6b7b04430ca5f1f7",
    "binance-sui-usdt-2026-10-06T0000-2026-10-07T2224.csv": "7cff06a6254707cd2d0f71fef960ff8725eba975bb0b6b33454e42878dbe4dc2",
    "binance-xrp-usdt-2026-10-06T0000-2026-10-07T2224-canon.csv": "e496e766948a7add51a265266c9672a666236b578cc1aa390bbc6cb82c9edc53",
    "binance-xrp-usdt-2026-10-06T0000-2026-10-07T2224.csv": "8b89399ed47d249a28610f78bb764605571345f3351820d6ea8ac2dad9acf9b8",
}
CANON_COUNTS = {"BTC/USDT": 1500, "ETH/USDT": 1582, "SOL/USDT": 1516, "XRP/USDT": 1406, "SUI/USDT": 1123}


def test_the_manifest_lists_exactly_the_signed_files():
    files = golden_lib.manifest()["files"]
    assert files, "the golden manifest is empty"
    assert {name.removesuffix(".gz"): e["sha256"] for name, e in files.items()} == SIGNED


def test_each_candle_fixture_is_the_whole_window_every_minute_whole():
    window = pd.date_range("2026-10-06T00:00", "2026-10-07T22:24", freq="1min", inclusive="left", tz="UTC")
    for name, e in golden_lib.manifest()["files"].items():
        df = golden_lib.load(name)
        if e["kind"] == "candles":
            assert df.index.equals(pd.DatetimeIndex(window, name="open_utc")), name
            assert not df[["missing", "degraded"]].any().any(), name
            assert (df["high"] >= df[["open", "close", "low"]].max(axis=1)).all(), name
            assert (df["low"] <= df[["open", "close"]].min(axis=1)).all(), name
        else:
            assert len(df) == CANON_COUNTS[e["pair"]] and df.index.isin(window).all(), name


def test_nothing_reads_the_golden_fixtures_except_through_golden_lib():
    root = Path(__file__).parent.parent
    allowed = {Path(golden_lib.__file__).resolve(), Path(__file__).resolve()}
    pattern = re.compile(r"""data["'/ ,]+golden|GOLDEN_DIR|[a-z0-9]+-[a-z0-9]+-usdt-\d{4}-\d\d-\d\dT\d{4}""")
    for bypass in ("pd.read_csv(golden_lib.GOLDEN_DIR / name)", "venue2-btc-usdt-2026-10-06T0000-2026-10-07T2224.csv.gz"):
        assert pattern.search(bypass), bypass
    offenders = [str(p.relative_to(root)) for d in ("sleeve_fund", "tests", "scripts") for p in (root / d).rglob("*.py")
                 if p.resolve() not in allowed and pattern.search(p.read_text(errors="ignore"))]
    assert offenders == [], f"read golden fixtures through golden_lib.load(), not directly: {offenders}"


def _printed_log(files: dict[str, bytes]) -> str:
    """A print job's log as the API returns it: SHA256SUMS, then each .gz base64 between BEGIN and END, timestamped."""
    lines = [f"{golden_lib.digest(raw)}  {name}" for name, raw in files.items()]
    for name, raw in files.items():
        gz = gzip.compress(raw, mtime=0)
        b64 = base64.b64encode(gz).decode()
        lines += [f"BEGIN {name}.gz {golden_lib.digest(gz)} {len(gz)}", *(b64[i:i + 1000] for i in range(0, len(b64), 1000)),
                  f"END {name}.gz"]
    return "\n".join(f"2026-10-08T06:53:48.{i:07d}Z {line}" for i, line in enumerate(lines)) + "\n"


def test_decode_log_rebuilds_the_files_and_fails_closed(tmp_path):
    files = {"a.csv": b"h\n" + b"1,2\n" * 5000, "b-canon.csv": b"h\n3\n"}
    log = _printed_log(files)
    assert sorted(golden_lib.decode_log(log, tmp_path / "ok")) == ["a.csv.gz", "b-canon.csv.gz"]
    assert gzip.decompress((tmp_path / "ok" / "a.csv.gz").read_bytes()) == files["a.csv"]
    no_begin = "\n".join(line for line in log.splitlines() if "BEGIN b-canon" not in line)
    with pytest.raises(ValueError, match="b-canon.csv: listed in SHA256SUMS but not in the log"):
        golden_lib.decode_log(no_begin, tmp_path / "x1")
    assert not (tmp_path / "x1").exists()  # a.csv.gz decoded fine, but nothing is written when the log fails
    lines = log.splitlines()
    cut = "\n".join(lines[:3] + lines[4:])  # one base64 line of a.csv.gz lost
    with pytest.raises(ValueError, match="a.csv.gz: does not match its BEGIN line"):
        golden_lib.decode_log(cut, tmp_path / "x2")
    with pytest.raises(ValueError, match="no END line"):
        golden_lib.decode_log("\n".join(lines[:-1]), tmp_path / "x3")
