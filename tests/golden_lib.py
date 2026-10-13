"""GOLDEN-VENUE: the committed venue candle fixtures and the manifest that pins their bytes.

Each fixture is tests/data/golden/<name>.csv.gz, written by `python -m sleeve_fund.history export` (one instrument, one
window) and gzipped: kind "candles" (the stored 1-minute bars) or "canon_replaced" (with --canon-replaced: the canon
backfill's latest replacement per minute, stored and offered). manifest.json lists every fixture with its kind, the
sha256 of its decompressed CSV, its row count and what it holds. check() reports every way the folder and the
manifest disagree; load() refuses a fixture that does not match. Read the fixtures only through load() (HoQA, PR B).

decode_log() rebuilds the .csv.gz files from a golden-export print job's log (#230), for when the run's artifact cannot
be downloaded; it fails unless every SHA256SUMS entry is rebuilt and matches.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import re
from pathlib import Path

import pandas as pd

from sleeve_fund.history import GOLDEN_CANON_HEADER, GOLDEN_HEADER

GOLDEN_DIR = Path(__file__).parent / "data" / "golden"
HEADERS = {"candles": GOLDEN_HEADER, "canon_replaced": GOLDEN_CANON_HEADER}


def manifest(root: Path = GOLDEN_DIR) -> dict:
    return json.loads((root / "manifest.json").read_text())


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def entry(raw: bytes, kind: str, venue: str, pair: str, start: str, end: str) -> dict:
    """The manifest line for one exported CSV (uncompressed bytes)."""
    return {"kind": kind, "sha256": digest(raw), "rows": raw.count(b"\n") - 1, "venue": venue, "pair": pair,
            "start": start, "end": end}


def check(root: Path = GOLDEN_DIR) -> list[str]:
    """Every disagreement between the fixtures on disk and the manifest; empty when they match exactly."""
    m = manifest(root)
    problems = [] if m.get("headers") == HEADERS else [f"manifest headers {m.get('headers')!r} are not the export's"]
    listed = m.get("files", {})
    on_disk = {p.name for p in root.glob("*.csv.gz")}
    problems += [f"{n}: on disk but not in the manifest" for n in sorted(on_disk - listed.keys())]
    problems += [f"{n}: in the manifest but not on disk" for n in sorted(listed.keys() - on_disk)]
    for name in sorted(on_disk & listed.keys()):
        raw = gzip.decompress((root / name).read_bytes())
        want = listed[name]
        if digest(raw) != want["sha256"]:
            problems.append(f"{name}: sha256 {digest(raw)} is not the manifest's {want['sha256']}")
        if want.get("kind") not in HEADERS:
            problems.append(f"{name}: kind {want.get('kind')!r} is not one of {sorted(HEADERS)}")
        elif not raw.startswith(HEADERS[want["kind"]].encode() + b"\n"):
            problems.append(f"{name}: does not start with the {want['kind']} export header")
        if (rows := raw.count(b"\n") - 1) != want["rows"]:
            problems.append(f"{name}: {rows} rows, the manifest says {want['rows']}")
    return problems


def load(name: str, root: Path = GOLDEN_DIR) -> pd.DataFrame:
    """One fixture's candles indexed by open time (UTC), after checking its bytes against the manifest."""
    want = manifest(root)["files"][name]
    raw = gzip.decompress((root / name).read_bytes())
    if digest(raw) != want["sha256"]:
        raise ValueError(f"{name}: sha256 does not match the manifest; the fixture has changed")
    df = pd.read_csv(io.BytesIO(raw), float_precision="round_trip")
    key = "open_utc" if want["kind"] == "candles" else "minute_utc"
    df.index = pd.to_datetime(df.pop(key), utc=True)
    df.index.name = key
    return df


def decode_log(text: str, out: Path) -> list[str]:
    """The .csv.gz files printed by golden-export's print job, written to out from the job's log text. Each file sits
    between "BEGIN <name> <sha256 of the .gz> <bytes>" and "END <name>"; Actions' timestamp prefixes are stripped.
    Raises unless each file matches its BEGIN line and every SHA256SUMS entry (sha256 of the uncompressed CSV) has a
    rebuilt file that matches it, with none left over; nothing is written to out until every check passes. Returns
    the names written."""
    lines = [re.sub(r"^\d{4}-\d\d-\d\dT[\d:.]+Z ", "", line) for line in text.splitlines()]
    sums = {m[2]: m[1] for line in lines if (m := re.fullmatch(r"([0-9a-f]{64})  (\S+\.csv)", line))}
    if not sums:
        raise ValueError("no SHA256SUMS lines in the log")
    rebuilt, name, chunks, want = {}, None, [], None
    for line in lines:
        if m := re.fullmatch(r"BEGIN (\S+\.csv\.gz) ([0-9a-f]{64}) (\d+)", line):
            name, chunks, want = m[1], [], (m[2], int(m[3]))
        elif name and line == f"END {name}":
            data = base64.b64decode("".join(chunks), validate=True)
            if digest(data) != want[0] or len(data) != want[1]:
                raise ValueError(f"{name}: does not match its BEGIN line; the log is cut or garbled")
            rebuilt[name] = data
            name = None
        elif name:
            chunks.append(line)
    if name:
        raise ValueError(f"{name}: no END line; the log is cut")
    for csv, sha in sums.items():
        if f"{csv}.gz" not in rebuilt:
            raise ValueError(f"{csv}: listed in SHA256SUMS but not in the log")
        if digest(gzip.decompress(rebuilt[f"{csv}.gz"])) != sha:
            raise ValueError(f"{csv}: does not match SHA256SUMS")
    if extra := sorted(set(rebuilt) - {f"{c}.gz" for c in sums}):
        raise ValueError(f"not in SHA256SUMS: {extra}")
    out.mkdir(parents=True, exist_ok=True)
    for gz_name, data in rebuilt.items():
        (out / gz_name).write_bytes(data)
    return list(rebuilt)
