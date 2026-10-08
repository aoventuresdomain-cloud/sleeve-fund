"""GOLDEN-VENUE: the committed venue candle fixtures and the manifest that pins their bytes.

Each fixture is tests/data/golden/<name>.csv.gz, written by `python -m sleeve_fund.history export` (one instrument, one
window) and gzipped. manifest.json lists every fixture with the sha256 of its decompressed CSV, its row count and what it
holds. check() reports every way the folder and the manifest disagree; load() refuses a fixture that does not match.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path

import pandas as pd

from sleeve_fund.history import GOLDEN_HEADER

GOLDEN_DIR = Path(__file__).parent / "data" / "golden"


def manifest(root: Path = GOLDEN_DIR) -> dict:
    return json.loads((root / "manifest.json").read_text())


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def entry(raw: bytes, venue: str, pair: str, start: str, end: str) -> dict:
    """The manifest line for one exported CSV (uncompressed bytes)."""
    return {"sha256": digest(raw), "rows": raw.count(b"\n") - 1, "venue": venue, "pair": pair, "start": start,
            "end": end}


def check(root: Path = GOLDEN_DIR) -> list[str]:
    """Every disagreement between the fixtures on disk and the manifest; empty when they match exactly."""
    m = manifest(root)
    problems = [] if m.get("header") == GOLDEN_HEADER else [f"manifest header {m.get('header')!r} is not the export's"]
    listed = m.get("files", {})
    on_disk = {p.name for p in root.glob("*.csv.gz")}
    problems += [f"{n}: on disk but not in the manifest" for n in sorted(on_disk - listed.keys())]
    problems += [f"{n}: in the manifest but not on disk" for n in sorted(listed.keys() - on_disk)]
    for name in sorted(on_disk & listed.keys()):
        raw = gzip.decompress((root / name).read_bytes())
        want = listed[name]
        if digest(raw) != want["sha256"]:
            problems.append(f"{name}: sha256 {digest(raw)} is not the manifest's {want['sha256']}")
        if not raw.startswith(GOLDEN_HEADER.encode() + b"\n"):
            problems.append(f"{name}: does not start with the export header")
        if (rows := raw.count(b"\n") - 1) != want["rows"]:
            problems.append(f"{name}: {rows} rows, the manifest says {want['rows']}")
    return problems


def load(name: str, root: Path = GOLDEN_DIR) -> pd.DataFrame:
    """One fixture's candles indexed by open time (UTC), after checking its bytes against the manifest."""
    want = manifest(root)["files"][name]
    raw = gzip.decompress((root / name).read_bytes())
    if digest(raw) != want["sha256"]:
        raise ValueError(f"{name}: sha256 does not match the manifest; the fixture has changed")
    df = pd.read_csv(io.BytesIO(raw), dtype={"degraded": int}, float_precision="round_trip")
    df.index = pd.to_datetime(df.pop("open_utc"), utc=True)
    df.index.name = "open_utc"
    return df
