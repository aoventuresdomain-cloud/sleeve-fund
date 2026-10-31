"""TRIAL-CORRECT (Advisor, B0's ruling 31 Oct; the DA's five conditions): one correcting row for a counted basket try
whose register Sharpe was the close-priced comparison's, not its book's (fixed for new tries in PR #341).

The register is write-once (migration 0018), so the wrong row stays; the correction is one more row of the same
variant (definition, code version, dataset) carrying the book's Sharpe. sharpes() reads each variant's latest row, so
the deflated Sharpe's spread then reads the corrected figure, while N (distinct variants), the round's tries and
no_selection don't move; only counts()["evaluations"], a row count shown for awareness, goes up by one.

The Sharpe comes from the sheet, never from a typed number: its results.json (check.basket.sharpe, the daily ledger's)
when it has one, else the book's Sharpe its hold-bar line prints (two decimals; Advisor, 31 Oct: "from the sheet, no
ledger"). A dry run by default: --apply writes. The printout is the report, which holds Sharpes and ids, so the
workflow encrypts it; the public log gets one line.

    python -m sleeve_fund.research.trial_correction --row ID --sheet STEM [--apply]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import secrets
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sleeve_fund.research import trials

# A sheet is written within this long of its study's try row (the row first, as basket_study records it): a later
# re-run of the same basket writes another sheet under the same label and dataset, never another row
SHEET_WITHIN = timedelta(hours=4)
_STAMP = re.compile(r"_(\d{8}-\d{6})(?:-\d+)?$")
_HOLD_LINE = re.compile(r"^\| Hold bar \(long-only baskets\) \| pooled Sharpe (-?\d+\.\d+) \(net ", re.M)
LEDGER, SHEET_ONLY = "the sheet's daily ledger (results.json)", "from the sheet, no ledger"


def target(sheet_md: Path) -> tuple[float, str]:
    """The book's Sharpe for the sheet, and where it came from."""
    results = sheet_md.with_name(f"{sheet_md.stem}.results.json")
    if results.exists():
        sharpe = json.loads(results.read_text(encoding="utf-8"))["check"]["basket"]["sharpe"]
        if not isinstance(sharpe, (int, float)) or not math.isfinite(sharpe):
            raise ValueError(f"refused: {results.name} gives no finite book Sharpe ({sharpe!r})")
        return float(sharpe), LEDGER
    found = _HOLD_LINE.findall(sheet_md.read_text(encoding="utf-8"))
    if len(found) != 1:
        raise ValueError(f"refused: {sheet_md.name} has no results.json and {len(found)} hold-bar Sharpe lines, not one")
    return float(found[0]), SHEET_ONLY


def plan(register: trials.TrialsRegister, row_id: str, sheet_md: Path) -> tuple[dict, dict, str]:
    """(the original row, the correcting row, the Sharpe's source), or ValueError naming why not."""
    rows = register._counted()
    original = next((t for t in rows if t["id"] == row_id), None)
    if original is None:
        raise ValueError(f"refused: no counted row {row_id}")
    if not str(original["definition_name"]).startswith("basket:"):
        raise ValueError(f"refused: {row_id} is not a basket try ({original['definition_name']})")
    if original["status"] != "ok" or original.get("kind", "single") != "single":
        raise ValueError(f"refused: {row_id} is {original['status']}/{original.get('kind')}, not ok/single")
    same = [t for t in rows if t["definition_hash"] == original["definition_hash"]]
    if len(same) != 1:
        raise ValueError(f"refused: {len(same)} counted rows have this basket key, not one (already corrected?)")
    label, dataset = original["definition_name"][len("basket:"):], original["dataset"]
    stamp = _STAMP.search(sheet_md.stem)
    if not sheet_md.stem.startswith(f"{label}_{dataset}_") or stamp is None:
        raise ValueError(f"refused: the sheet {sheet_md.stem} is not {label} on {dataset}")
    written = datetime.strptime(stamp.group(1), "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
    created = trials._aware(original["created_at"])
    if not created <= written <= created + SHEET_WITHIN:
        raise ValueError(f"refused: the sheet was written {written:%d %b %H:%M}, not within {SHEET_WITHIN} after the "
                         f"row ({created:%d %b %H:%M}): it is another run's sheet")
    sharpe, source = target(sheet_md)
    if original["sharpe"] is not None and math.isclose(original["sharpe"], sharpe, abs_tol=1e-9):
        raise ValueError(f"refused: {row_id}'s Sharpe is already {sharpe}")
    fixed = {k: v for k, v in original.items() if k not in ("id", "created_at", "sharpe", "error")}
    fixed.update(id=secrets.token_hex(8), created_at=datetime.now(timezone.utc), sharpe=sharpe, error=None)
    if "by_year" in fixed:
        fixed["by_year"] = None
    if trials._variant(fixed) != trials._variant(original):  # the DA's condition 2: asserted, not assumed
        raise ValueError("refused: the correcting row would be another variant")
    return original, fixed, source


def trail(original: dict, fixed: dict, source: str, stem: str) -> str:
    """The correction's line for the trials-corrections note: the register itself can't mark one."""
    return (f"- {fixed['created_at']:%Y-%m-%d %H:%M} UTC: row {original['id']} corrected by row {fixed['id']}, "
            f"Sharpe {original['sharpe']} -> {fixed['sharpe']} ({source}), sheet {stem}; Advisor B0 ruling, 31 Oct 2026")


def main(argv=None, store=None, sheets: Path | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sleeve_fund.research.trial_correction")
    ap.add_argument("--row", required=True, help="the counted basket try's trials id")
    ap.add_argument("--sheet", required=True, help="its tear sheet's stem, in TEARSHEET_DIR")
    ap.add_argument("--apply", action="store_true", help="write the row (default: a dry run, nothing written)")
    a = ap.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{16}", a.row) or not re.fullmatch(r"[A-Za-z0-9._-]+", a.sheet):
        print("refused: --row is a 16-character trials id and --sheet a plain file stem")
        return 2
    if store is None:
        from sleeve_fund.store import Store

        store = Store()
    from sleeve_fund.research.run import TEARSHEETS

    register = trials.TrialsRegister(store)
    try:
        original, fixed, source = plan(register, a.row, (sheets or TEARSHEETS) / f"{a.sheet}.md")
    except (ValueError, OSError, KeyError) as exc:
        print(exc)
        return 1
    idea = original["idea_hash"]
    before = (register.counts(idea), register.no_selection(idea), sorted(register.sharpes(idea)))
    print("original:", json.dumps(original, default=str, sort_keys=True))
    print("adds:    ", json.dumps(fixed, default=str, sort_keys=True))
    print(f"Sharpe from {source}; the variant's latest Sharpe {original['sharpe']} -> {fixed['sharpe']}")
    if not a.apply:
        print("dry run: nothing written")
        return 0
    if register.store.add_trials([fixed]) != 1:
        print("refused: the row was not added")
        return 1
    after = (register.counts(idea), register.no_selection(idea), sorted(register.sharpes(idea)))
    moved = [k for k in ("ideas", "variants", "n_uncertain") if before[0][k] != after[0][k]]
    print(f"written. N {before[0]['variants']} -> {after[0]['variants']}, evaluations {before[0]['evaluations']} -> "
          f"{after[0]['evaluations']}, no_selection {before[1]!r} -> {after[1]!r}")
    print("trail:", trail(original, fixed, source, a.sheet))
    if moved or before[1] != after[1] or fixed["sharpe"] not in after[2]:
        print(f"CHECK FAILED: {moved or 'no_selection or the spread'} moved")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
