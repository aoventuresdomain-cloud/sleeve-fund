"""Re-score a list of research definitions on the server (the study workflow, .github/workflows/study.yml).

    python -m sleeve_fund.research.rescore configs/studies/qr-rescore-list.md [--rows 1-8] [--dry-run]

Each row of the list's studies table (| # | definition | idea family | decision candle |) runs one G1 study of the
`rules` model with params = to_params(definition) plus market "perp" (allow_short on a definition with a short
side), on the venue's stored history, as the Research page runs a rule-builder Strategy's. Rows with first_touch
get the 1-minute loader through the store study. The holdout stays locked: --use-holdout is never passed, and a
request that asks for it is refused. Each study writes only research records: its trials register row (the
journal), its idea counter line and its tear sheet. No Strategy, order, fill or fund row is ever written.

The job log is public, so this prints counts and per-row ran/failed with the error's class only: never a figure,
a verdict or a tear sheet. The results are read on the Research page.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from sleeve_fund.research.pinned import parse as parse_pinned
from sleeve_fund.venues import RESEARCH_VENUE

ROOT = Path(__file__).resolve().parents[2]
DEFINITIONS = ROOT / "configs" / "definitions"
CANDLES = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}
# R-29: on a study whose configs came from main, until R-25 counts tries by idea family (docs/research-definitions.md)
TRIES_BY_HAND = "Tries counted by hand until R-25."
ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*([a-z0-9_]+)\s*\|[^|]*\|\s*(\w+)\s*\|\s*$")
# The list's common settings (research/quant-researcher/qr-rescore-list.md, 20 Oct): walk-forward 730/365, the
# holdout's 365 days locked, risk profile none (1x, uncapped, unguarded).
TRAIN_DAYS, TEST_DAYS, HOLDOUT_DAYS = 730, 365, 365


def rows_of(text: str) -> list[tuple[int, str, int]]:
    """(row number, definition name, decision minutes) for each row of the list's studies table."""
    out = []
    for line in text.splitlines():
        m = ROW.match(line.strip())
        if m is None:
            continue
        n, name, candle = int(m[1]), m[2], m[3]
        if candle not in CANDLES:
            raise ValueError(f"row {n}: decision candle {candle!r} is not one of {', '.join(CANDLES)}")
        out.append((n, name, CANDLES[candle]))
    if not out:
        raise ValueError("the list has no studies table rows")
    if len({n for n, _, _ in out}) != len(out):
        raise ValueError("the list numbers a row twice")
    return out


def pick(rows: list, spec: str | None) -> list:
    """The rows named by '1-8', '9-16', '3' or '1,4,7'; all of them for None or ''."""
    if not spec:
        return rows
    if not re.fullmatch(r"\d+(-\d+)?(,\d+(-\d+)?)*", spec):
        raise ValueError("rows must look like 1-8 or 1,4,7")
    want: set[int] = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        want.update(range(int(lo), int(hi or lo) + 1))
    chosen = [r for r in rows if r[0] in want]
    if missing := sorted(want - {r[0] for r in chosen}):
        raise ValueError(f"no such rows in the list: {', '.join(map(str, missing))}")
    return chosen


def request(name: str, minutes: int, venue: str, pair: str, pinned=None):
    """The store study one row runs: rules, its definition's params, perp, the locked holdout, on the list's pinned
    dates (B-37)."""
    from sleeve_fund.research.run import StudyRequest
    from sleeve_fund.strategies.definitions import load_definition_file, to_params

    path = DEFINITIONS / f"{name}.toml"
    if not path.is_file():
        raise ValueError(f"no definition file {name}.toml")
    defn = load_definition_file(path)
    params = {**to_params(defn, venue.upper()), "market": "perp"}
    if "short" in defn:
        params["allow_short"] = True
    return StudyRequest(strategy="rules", pair=pair, venue=venue, minutes=minutes, risk_profile=None,
                        train_days=TRAIN_DAYS, test_days=TEST_DAYS, holdout_days=HOLDOUT_DAYS, use_holdout=False,
                        params=params, label=name, pinned=pinned)


def main(argv: list[str] | None = None, run=None, journal=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sleeve_fund.research.rescore")
    ap.add_argument("list", help="the study list (markdown with a studies table)")
    ap.add_argument("--rows", default="", help="which rows, e.g. 1-8; all by default")
    ap.add_argument("--venue", default=RESEARCH_VENUE.lower(), help="the venue profile whose stored history is read")
    ap.add_argument("--pair", default="BTC/USDT")
    ap.add_argument("--dry-run", action="store_true", help="check every row's definition and settings, run nothing")
    ap.add_argument("--tries-by-hand", action="store_true", help=f"say on each sheet: {TRIES_BY_HAND}")
    args = ap.parse_args(argv)
    if not re.fullmatch(r"[a-z]{1,32}", args.venue) or not re.fullmatch(r"[A-Z0-9]{1,16}/[A-Z0-9]{1,16}", args.pair):
        print("refused: venue must be lower-case letters, pair like BTC/USDT")
        return 2
    try:
        text = Path(args.list).read_text(encoding="utf-8")
        pinned = parse_pinned(text)  # B-37: a list that pins no dates is refused before anything runs
        if pinned.holdout_days != HOLDOUT_DAYS:
            raise ValueError(f"the list pins a {pinned.holdout_days}-day holdout, not {HOLDOUT_DAYS}")
        rows = pick(rows_of(text), args.rows)
        reqs = [(n, name, request(name, minutes, args.venue, args.pair, pinned)) for n, name, minutes in rows]
    except (OSError, ValueError) as exc:
        print(f"refused: {exc}")
        return 2
    for _, _, req in reqs:
        assert req.use_holdout is False  # never opened from here
        req.validate()
    print(f"{len(reqs)} studies on the list's settings (holdout locked)")
    if args.dry_run:
        for n, name, _ in reqs:
            print(f"row {n:>2} {name}: checked")
        print(f"dry run: {len(reqs)} checked, 0 run")
        return 0
    if run is None:
        from sleeve_fund.research.run import run_store_study as run
    if journal is None:
        from sleeve_fund.__main__ import _journal as journal
    store = journal()
    ok = failed = 0
    extra = {"notes": (TRIES_BY_HAND,)} if args.tries_by_hand else {}
    for n, name, req in reqs:
        try:
            run(req, store=store, **extra)
        except Exception as exc:  # noqa: BLE001 - one row's failure never stops the rest; its class only (public log)
            failed += 1
            print(f"row {n:>2} {name}: FAILED ({type(exc).__name__})")
            continue
        ok += 1
        print(f"row {n:>2} {name}: ran")
    print(f"done: {ok} ran, {failed} failed; results on the Research page")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
