"""Re-score a list of research definitions on the server (the study workflow, .github/workflows/study.yml).

    python -m sleeve_fund.research.rescore configs/studies/qr-rescore-list.md [--rows 1-8] [--dry-run]

Each row of the list's studies table (| # | definition | idea family | decision candle |) runs one G1 study of the
`rules` model with params = to_params(definition) plus market "perp" (allow_short on a definition with a short
side; an optional fifth | market | column of perp or spot-lowfee sets the study's market instead), on the venue's
stored history, as the Research page runs a rule-builder Strategy's. Rows with first_touch
get the 1-minute loader through the store study. The holdout stays locked: --use-holdout is never passed, and a
request that asks for it is refused. Each study writes only research records: its trials register row (the
journal), its idea counter line and its tear sheet. No Strategy, order, fill or fund row is ever written.

The job log is public, so this prints counts and per-row ran/failed with the error's class only: never a figure,
a verdict or a tear sheet. A row that ran prints its sheet's file stem (name, dataset and time: no result), so the
sheet export (.github/workflows/sheet-export.yml) can be run on it. A row that failed prints the stem of a note
holding its message and traceback (refused/ beside the sheets), which the same export reads. The results are read on
the Research page.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from sleeve_fund.research import rounds
from sleeve_fund.research.pinned import parse as parse_pinned
from sleeve_fund.venues import RESEARCH_VENUE

ROOT = Path(__file__).resolve().parents[2]
DEFINITIONS = ROOT / "configs" / "definitions"
CANDLES = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}
# R-29: on a study whose configs came from main, until R-25 counts tries by idea family (docs/research-definitions.md)
TRIES_BY_HAND = "Tries counted by hand until R-25."
# Where a failed row's note goes; None is refused/ beside the tear sheets (tests point it elsewhere)
REFUSED: Path | None = None
ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*([a-z0-9_]+)\s*\|[^|]*\|\s*(\w+)\s*\|(?:\s*([a-z-]*)\s*\|)?\s*$")
# The list's common settings (research/quant-researcher/qr-rescore-list.md, 20 Oct): walk-forward 730/365, the
# holdout's 365 days locked, risk profile none (1x, uncapped, unguarded). A list's pinned dates set its own (fold_days):
# the next round's (PM 30 Oct, "seal May to Oct 2026") is 366/485 with a 183-day holdout.
TRAIN_DAYS, TEST_DAYS, HOLDOUT_DAYS = 730, 365, 365


def fold_days(pinned) -> tuple[int, int, int]:
    """(train, test, holdout) days as the list pins them: fold 1's train start to test start, its test start to test
    end (both included), and the holdout's days. The study then checks every fold against the pinned ones."""
    f = pinned.folds[0]
    return (f.test_start - f.train_start).days, (f.test_end - f.test_start).days + 1, pinned.holdout_days
# R-11/R-12 (Advisor SCORE-WINDOW, 21 Oct): a list whose common settings say "Scoring window: continuous" scores each
# pinned row continuously after its warm-up; without the line it keeps the walk-forward. This round only, the rows
# scored on the old window before (QR-B1, QR-B2, QR-B3) also show that window's verdict and pass only if both pass.
CONTINUOUS_LINE = re.compile(r"^-\s*\*\*Scoring window:\*\*\s*continuous\b", re.M | re.I)
# Advisor 27 Oct (fast lane): an asset list's two runs. "Fund guards: off" runs the basket with every fund limit off
# (its edge verdict, counted once); "Size scale: 0.55" sizes every leg at that share of its size (never a try).
FUND_GUARDS_LINE = re.compile(r"^-\s*\*\*Fund guards:\*\*\s*(on|off)\b", re.M | re.I)
SIZE_SCALE_LINE = re.compile(r"^-\s*\*\*Size scale:\*\*\s*(\d*\.?\d+)\s*$", re.M | re.I)
PAIRS = r"[A-Z0-9]{1,16}/[A-Z0-9]{1,16}(,[A-Z0-9]{1,16}/[A-Z0-9]{1,16}){0,3}"  # one instrument, or a list of up to 4
OLD_WINDOW_TOO = frozenset({
    "qr01_trend_ema_long", "qr01_trend_ema_short", "qr01b_trend_multispeed_long", "qr01b_trend_multispeed_short",
    "qr01c_trend_donchian_long", "qr01c_trend_donchian_short"})


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


def markets_of(text: str) -> dict[int, str]:
    """RESEARCH-SPOT-FEES: row number -> the study's market, for rows whose optional fifth column (| market |) names
    one; a row without it, or with it blank, runs as before (perp). Only perp or spot-lowfee."""
    from sleeve_fund.research.run import StudyRequest, study_market

    out = {}
    for line in text.splitlines():
        m = ROW.match(line.strip())
        if m is None or not m[4]:
            continue
        study_market(StudyRequest(strategy="rules", pair="BTC/USDT", market=m[4]))  # refuses anything else
        out[int(m[1])] = m[4]
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


def confirming_of(text: str) -> dict:
    """The list's confirming-run settings (Advisor 27 Oct) as StudyRequest fields: {} when it names neither."""
    out = {}
    if (m := FUND_GUARDS_LINE.search(text)) is not None:
        out["fund_guards"] = m[1].lower() == "on"
    if (m := SIZE_SCALE_LINE.search(text)) is not None:
        out["size_scale"] = float(m[1])
    return out


def request(name: str, minutes: int, venue: str, pair: str, pinned=None, continuous: bool = False,
            market: str | None = None, confirming: dict | None = None):
    """The store study one row runs: rules, its definition's params, perp (or the row's market), the locked holdout,
    on the list's pinned dates (B-37), scored continuously after a warm-up when the list says so (R-11). pair: one
    instrument, or several joined by commas (STUDY-ASSET-LIST), run as one basket."""
    from sleeve_fund.research.run import StudyRequest
    from sleeve_fund.strategies.definitions import load_definition_file, to_params

    from sleeve_fund.research.variants import PREREGISTERED, definition_path

    variant = PREREGISTERED.get(name.upper())  # a pre-registered variant's row names it: v1 to v4, v3_adx
    path = definition_path(variant["definition"]) if variant else DEFINITIONS / f"{name}.toml"
    if not path.is_file():
        raise ValueError(f"no definition file {path.name}")
    defn = load_definition_file(path)
    params = {**to_params(defn, venue.upper()), "market": "perp", **(variant["params"] if variant else {})}
    if "short" in defn:
        params["allow_short"] = True
    parent = (load_definition_file(definition_path(variant["parent"]))
              if continuous and variant and variant.get("placebo") else None)  # R-24
    pairs = pair.split(",") if "," in pair else None
    train, test, holdout = fold_days(pinned) if pinned is not None else (TRAIN_DAYS, TEST_DAYS, HOLDOUT_DAYS)
    return StudyRequest(strategy="rules", pair=pairs[0] if pairs else pair, pairs=pairs, venue=venue, minutes=minutes,
                        risk_profile=None, train_days=train, test_days=test, holdout_days=holdout, use_holdout=False,
                        params=params, label=name, pinned=pinned, placebo_parent=parent, market=market,
                        **(confirming or {}),
                        **({"score_window": "continuous", "old_window_too": name in OLD_WINDOW_TOO and not pairs}
                           if continuous else {}))


def _minute_check(req, history=None) -> str:
    """For the dry run: whether a row would trade on 1-minute bars over its pinned window, read from the stored history
    without running anything (PM 23 Oct: prove it on the real store first). Dates and bar lengths only (public log)."""
    if req.minutes <= 1 or "," in req.pair:
        return ""
    from dataclasses import replace

    from sleeve_fund.history import HistoryStore
    from sleeve_fund.research.run import execution_bars
    from sleeve_fund.venues import venue as venue_profile

    profile = venue_profile(req.venue)
    history = history or HistoryStore()
    out, each = [], req.pairs is not None and req.minutes != 1440
    for pair in req.pairs if each else (req.pair,):  # a 4-hour asset list: each asset's, as its basket reads them
        label = f" {pair}" if each else ""
        try:
            prices = history.read(profile.name, pair, req.minutes)
            _, oos_exec, oos_step = execution_bars(replace(req, pair=pair, pairs=None), prices, profile, history)
        except (KeyError, ValueError) as exc:
            return f"; execution bars{label}: REFUSED ({exc})"
        if oos_exec is None or not len(oos_exec):
            out.append(f"; execution bars{label}: none")
            continue
        out.append(f"; execution bars{label}: {oos_step}-minute, {oos_exec.index[0]:%d %b %Y} to "
                   f"{oos_exec.index[-1]:%d %b %Y}, {len(oos_exec):,} bars")
    return "".join(out)


def refusal_note(n: int, name: str, args, exc: BaseException, out_dir: Path | None = None) -> str | None:
    """A failed row's message and traceback, kept on the server beside the tear sheets (in refused/), never in the
    public log: the log prints the note's stem only, and the sheet export reads it like a sheet, encrypted. None when
    it can't be written (the row has failed already; the note is a help, never a second failure)."""
    import traceback
    from datetime import datetime, timezone

    try:
        out_dir = out_dir or REFUSED
        if out_dir is None:
            from sleeve_fund.research.run import TEARSHEETS
            out_dir = TEARSHEETS / "refused"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"refused-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-row{n}"
        trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        (out_dir / f"{stem}.md").write_text(
            f"# Row {n} {name} failed\n\n- **List:** {Path(args.list).name}\n- **Venue:** {args.venue}\n"
            f"- **Pair:** {args.pair}\n- **Error:** {type(exc).__name__}: {exc}\n\n```\n{trace}```\n",
            encoding="utf-8")
        return stem
    except Exception:  # noqa: BLE001
        return None


def main(argv: list[str] | None = None, run=None, journal=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sleeve_fund.research.rescore")
    ap.add_argument("list", help="the study list (markdown with a studies table)")
    ap.add_argument("--rows", default="", help="which rows, e.g. 1-8; all by default")
    ap.add_argument("--venue", default=RESEARCH_VENUE.lower(), help="the venue profile whose stored history is read")
    ap.add_argument("--pair", default="BTC/USDT",
                    help="the instrument, or a list run as one basket: BTC/USDT,ETH/USDT,SOL/USDT,XRP/USDT")
    ap.add_argument("--dry-run", action="store_true", help="check every row's definition and settings, run nothing")
    ap.add_argument("--tries-by-hand", action="store_true", help=f"say on each sheet: {TRIES_BY_HAND}")
    args = ap.parse_args(argv)
    if not re.fullmatch(r"[a-z]{1,32}", args.venue) or not re.fullmatch(PAIRS, args.pair):
        print("refused: venue must be lower-case letters, pair like BTC/USDT or BTC/USDT,ETH/USDT")
        return 2
    try:
        text = Path(args.list).read_text(encoding="utf-8")
        pinned = parse_pinned(text)  # B-37: a list that pins no dates is refused before anything runs
        listed = rows_of(text)
        rnd = rounds.round_of(text, [n for n, _, _ in listed])  # M1 items 4-6: None for a list that names no round
        rows = pick(listed, args.rows)
        continuous = CONTINUOUS_LINE.search(text) is not None
        market = markets_of(text)
        confirming = confirming_of(text)
        reqs = [(n, name, request(name, minutes, args.venue, args.pair, pinned, continuous, market.get(n),
                                  confirming)) for n, name, minutes in rows]
        for _, _, req in reqs:
            req.validate()
        ideas = {}  # family -> the ideas its rows count under, every row of the list's, not only those picked
        if rnd is not None:
            from sleeve_fund.research.study import idea_of

            for n, name, minutes in listed:
                params = request(name, minutes, args.venue, args.pair, pinned, continuous, market.get(n)).params
                ideas.setdefault(rnd.family_of(n), set()).add(idea_of(params, None))
    except (OSError, ValueError) as exc:
        print(f"refused: {exc}")
        return 2
    for _, _, req in reqs:
        assert req.use_holdout is False  # never opened from here
        req.validate()
    print(f"{len(reqs)} studies on the list's settings (holdout locked)")
    if confirming:
        print(("one try per basket" if "size_scale" not in confirming else "not new tries") + ": fund guards " + ("on" if confirming.get("fund_guards", True) else
              "OFF (research only)") + (f", size scale {confirming['size_scale']:g}" if "size_scale" in confirming
                                         else ""))
    if rnd is not None:
        print(f"round {rnd.name}, study list SHA-256 {rnd.sha256}: {len(rnd.rows)} registered tries, a budget of "
              f"{rnd.budget} tries per family")
    if args.dry_run:
        for n, name, req in reqs:
            place = f", registered try {rnd.place(n)[0]} of {len(rnd.rows)} in family {rnd.family_of(n)}" if rnd else ""
            print(f"row {n:>2} {name}: checked{_minute_check(req)}{place}")
        print(f"dry run: {len(reqs)} checked, 0 run")
        return 0
    if run is None:
        from sleeve_fund.research.run import run_store_study as run
    if journal is None:
        from sleeve_fund.__main__ import _journal as journal
    store = journal()
    ok = failed = 0
    extra = {"notes": (TRIES_BY_HAND,)} if args.tries_by_hand and rnd is None else {}
    register = None
    if rnd is not None:
        from sleeve_fund.research.trials import TrialsRegister

        register = TrialsRegister(store)
    for n, name, req in reqs:
        if rnd is not None:
            family = rnd.family_of(n)
            so_far = rounds.used(register, ideas[family])
            if so_far >= rnd.budget:  # M1 item 5: refused before it runs, and counted as a failed row
                failed += 1
                print(f"row {n:>2} {name}: REFUSED, family {family} has used {so_far} of its {rnd.budget} tries")
                continue
            extra = {"notes": (rounds.try_line(rnd, n, so_far),)}
        try:
            sheet = run(req, store=store, **extra)
        except Exception as exc:  # noqa: BLE001 - one row's failure never stops the rest; its class only (public log)
            failed += 1
            note = refusal_note(n, name, args, exc)
            print(f"row {n:>2} {name}: FAILED ({type(exc).__name__})" + (f", note {note}" if note else ""))
            continue
        ok += 1
        print(f"row {n:>2} {name}: ran" + (f", sheet {sheet.stem}" if isinstance(sheet, Path) else ""))
    print(f"done: {ok} ran, {failed} failed; results on the Research page")
    if rnd is not None:  # M1 item 6: the round in one line, off the register
        print(rounds.summary(rnd, {f: rounds.used(register, i) for f, i in ideas.items()}))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
