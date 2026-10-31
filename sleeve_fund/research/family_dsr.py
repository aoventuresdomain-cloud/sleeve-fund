"""FAMILY-DSR (Advisor, 31 Oct 2026, on H0): a basket try's deflated Sharpe judged across its whole idea family, not
only its own idea's rows.

A round file pins each family's members and the rows its Sharpe spread may use:

    - **Family members, B:** 66fb76f17c2a5ec3, 0e13cf32e1775235, ...
    - **Valid-fill rows, B:** 5f5556da26db40ba/0123456789abcdef (stand-ins 12%), ...

Members are idea hashes (structural ideas, or a legacy ledger idea's). A valid-fill row is "definition hash / trials
id" for a run whose sheet shows 1-minute execution fired (Advisor's check 1), chosen by hand from the sheets and pinned
here, so the rule is written down and never inferred. The stand-in share is noted beside it: funding stand-ins don't
exclude a row from a family whose signal isn't funding (Advisor, 31 Oct).

- **N:** the larger of the stated count (the round's "Prior tries" line plus the round rows' tries in the register)
  and the register's (every distinct variant under the members), flagged when they differ.
- **Spread:** one Sharpe per variant, its latest valid-fill row's. A pinned row must be its variant's latest counted
  row: a newer one (a correction, a re-run) means the pin is stale and the step refuses, so a correction must be
  applied and pinned before this runs (Advisor: no deflated Sharpe before the B0 and V5 corrections). With fewer than
  3 valid-fill rows, the spread is floored at 0.5 a year.

    python -m sleeve_fund.research.family_dsr --list configs/studies/LIST.md --family B --sheet STEM

reads the sheet's pooled daily returns (STEM-pooled.csv, the test days) and prints the report: ids and Sharpes, so the
workflow encrypts it.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sleeve_fund.research import rounds, trials
from sleeve_fund.research.metrics import PERIODS_PER_YEAR, deflated_sharpe_probability

SPREAD_FLOOR = 0.5  # a year, below MIN_ROWS valid-fill rows (B's clearance, 30 Oct)
MIN_ROWS = 3
HEX16 = r"[0-9a-f]{16}"
MEMBERS_LINE = re.compile(r"^-\s*\*\*Family members, ([A-Za-z0-9]{1,16}):\*\*\s*(.+?)\s*$", re.MULTILINE)
VALID_LINE = re.compile(r"^-\s*\*\*Valid-fill rows, ([A-Za-z0-9]{1,16}):\*\*\s*(.+?)\s*$", re.MULTILINE)
ENTRY = re.compile(rf"({HEX16})/({HEX16})(?:\s*\(stand-ins (\d{{1,3}})%\))?")


@dataclass(frozen=True)
class Family:
    name: str
    members: tuple  # idea hashes
    valid: tuple  # (definition hash, trials id, stand-in share % or None)


def families_of(text: str) -> dict:
    """Each family's pinned members and valid-fill rows, refused in words when a line is malformed."""
    out: dict = {}
    for f, spec in MEMBERS_LINE.findall(text):
        if f in out:
            raise ValueError(f"family {f}'s members are listed twice")
        members = [m.strip() for m in spec.split(",")]
        if not members or any(not re.fullmatch(HEX16, m) for m in members) or len(set(members)) != len(members):
            raise ValueError(f"family {f}'s members are distinct 16-character idea hashes, comma-separated")
        out[f] = Family(f, tuple(members), ())
    seen = set()
    for f, spec in VALID_LINE.findall(text):
        if f not in out:
            raise ValueError(f"valid-fill rows are pinned for family {f}, whose members aren't")
        if f in seen:
            raise ValueError(f"family {f}'s valid-fill rows are listed twice")
        seen.add(f)
        parts = [p.strip() for p in spec.split(",")]
        entries = [ENTRY.fullmatch(p) for p in parts]
        if any(e is None for e in entries):
            raise ValueError(f"family {f}'s valid-fill rows read <definition hash>/<trials id> (stand-ins N%)")
        valid = tuple((e[1], e[2], None if e[3] is None else int(e[3])) for e in entries)
        if len({v[1] for v in valid}) != len(valid):
            raise ValueError(f"family {f} pins a trials id twice")
        out[f] = Family(f, out[f].members, valid)
    if len(re.findall(r"^-\s*\*\*(Family members|Valid-fill rows)", text, re.MULTILINE | re.IGNORECASE)) != (
            len(MEMBERS_LINE.findall(text)) + len(VALID_LINE.findall(text))):
        raise ValueError("a family line reads - **Family members, B:** <hashes> or - **Valid-fill rows, B:** <rows>")
    return out


@dataclass(frozen=True)
class Result:
    family: str
    n_stated: int
    n_register: int
    spread_rows: tuple  # (trials id, definition hash, Sharpe, stand-in %)
    floored: bool
    dsr: float
    binding: tuple = ("", math.nan)  # (which bound set the spread, its value a year): Advisor 31 Oct, shown


    @property
    def n(self) -> int:
        return max(self.n_stated, self.n_register)


def spread_rows(register: trials.TrialsRegister, fam: Family) -> tuple:
    """The pinned valid-fill rows as the register holds them, each its variant's latest counted row, one per variant."""
    counted = register._counted()
    by_id = {t["id"]: t for t in counted}
    latest: dict = {}
    for t in counted:  # trials() is oldest first
        latest[trials._variant(t)] = t
    out, variants = [], set()
    for definition, row_id, standins in fam.valid:
        t = by_id.get(row_id)
        if t is None:
            raise ValueError(f"refused: no counted row {row_id} (a correction not yet applied?)")
        if t["definition_hash"] != definition:
            raise ValueError(f"refused: row {row_id} is definition {t['definition_hash']}, not {definition}")
        if latest[trials._variant(t)]["id"] != row_id:
            raise ValueError(f"refused: row {row_id} is not its variant's latest row ({latest[trials._variant(t)]['id']}"
                             " is newer): re-pin the latest valid-fill row")
        if trials._variant(t) in variants:
            raise ValueError(f"refused: row {row_id}'s variant is pinned twice")
        variants.add(trials._variant(t))
        if t["sharpe"] is None or not math.isfinite(t["sharpe"]):
            raise ValueError(f"refused: row {row_id} has no finite Sharpe")
        out.append((row_id, definition, float(t["sharpe"]), standins))
    return tuple(out)


def compute(register: trials.TrialsRegister, text: str, family: str, returns: pd.Series, row_ideas: set) -> Result:
    """H0's (or any try's) deflated Sharpe across its family. row_ideas: the ideas the round's rows of this family
    count under (rescore derives them as it runs), for the stated count."""
    fam = families_of(text).get(family)
    if fam is None:
        raise ValueError(f"refused: the list pins no members for family {family}")
    if not fam.valid:
        raise ValueError(f"refused: the list pins no valid-fill rows for family {family}")
    rnd = rounds.round_of(text, [n for n in _rows(text)])
    prior = rnd.prior.get(family, 0) if rnd is not None else 0
    n_stated = prior + rounds.used(register, row_ideas)
    n_register = rounds.used(register, set(fam.members))
    rows = spread_rows(register, fam)
    sharpes = [s for _, _, s, _ in rows]
    floored = len(sharpes) < MIN_ROWS
    dsr = deflated_sharpe_probability(returns, max(n_stated, n_register, 1), sharpes,
                                      spread_floor=SPREAD_FLOOR if floored else None)
    return Result(family, n_stated, n_register, rows, floored, dsr, binding(returns, sharpes, floored))


def binding(returns: pd.Series, sharpes: list, floored: bool) -> tuple:
    """Which of the spread's bounds is the largest, and its value a year, as deflated_sharpe_probability takes it:
    the valid-fill rows' measured spread (2 rows or more), the no-skill error 1/sqrt(T-1), and the 0.5 floor (below
    3 rows). The Advisor (31 Oct): the reader sees why the hurdle is where it is."""
    year = math.sqrt(PERIODS_PER_YEAR)
    bounds = [("the no-skill error 1/sqrt(T-1)", year / math.sqrt(max(len(returns.dropna()) - 1, 1)))]
    if len(sharpes) > 1:
        bounds.append(("the valid-fill rows' spread", float(np.std(sharpes, ddof=1))))
    if floored:
        bounds.append((f"the {SPREAD_FLOOR} floor", SPREAD_FLOOR))
    return max(bounds, key=lambda b: b[1])


def _rows(text: str) -> list:
    from sleeve_fund.research.rescore import rows_of

    return [n for n, _, _ in rows_of(text)]


def pooled_returns(path: Path) -> pd.Series:
    """The sheet's pooled daily returns over its test days (basket_run's pooled.csv)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "pooled_return" not in rows[0]:
        raise ValueError(f"refused: {path.name} has no pooled_return column")
    return pd.Series([float(r["pooled_return"]) for r in rows], index=pd.to_datetime([r["date"] for r in rows]))


def report(r: Result) -> list:
    lines = [f"family {r.family}: N {r.n} (stated {r.n_stated}, register {r.n_register})"
             + (" FLAG: the counts differ" if r.n_stated != r.n_register else "")]
    for row_id, definition, sharpe, standins in r.spread_rows:
        lines.append(f"  spread row {row_id} ({definition}): Sharpe {sharpe:.4f}"
                     + (f", funding stand-ins {standins}%" if standins is not None else ""))
    lines.append(f"spread: {len(r.spread_rows)} valid-fill rows" + (f", floored at {SPREAD_FLOOR} a year (fewer than "
                                                                     f"{MIN_ROWS})" if r.floored else ""))
    lines.append(f"spread bound: {r.binding[0]}, {r.binding[1]:.2f} a year")
    lines.append(f"deflated Sharpe probability {r.dsr:.4f}")
    return lines


def main(argv=None, store=None, sheets: Path | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sleeve_fund.research.family_dsr")
    ap.add_argument("--list", required=True, help="the round's study list, under configs/studies/")
    ap.add_argument("--family", required=True, help="the family, e.g. B")
    ap.add_argument("--sheet", required=True, help="the try's tear sheet stem, in TEARSHEET_DIR")
    a = ap.parse_args(argv)
    if (not re.fullmatch(r"[A-Za-z0-9]{1,16}", a.family) or not re.fullmatch(r"[A-Za-z0-9._-]+", a.sheet)
            or ".." in a.sheet or not re.fullmatch(r"configs/studies/[A-Za-z0-9._-]+\.md", a.list) or ".." in a.list):
        print("refused: --list is a configs/studies/ file, --family letters or digits, --sheet a plain file stem")
        return 2
    if store is None:  # the research CLI's journal (allowlisted research, as rescore's): never a default file
        from sleeve_fund.__main__ import _journal

        store = _journal()
        if store is None:
            print("refused: no DATABASE_URL, so no register to read")
            return 2
    from sleeve_fund.research import rescore
    from sleeve_fund.research.run import TEARSHEETS
    from sleeve_fund.research.study import idea_of
    from sleeve_fund.research.trials import legacy_idea_hash

    try:
        text = Path(a.list).read_text(encoding="utf-8")
        pinned = rescore.parse_pinned(text)
        rnd = rounds.round_of(text, _rows(text))
        ideas = set()
        for n, name, minutes in rescore.rows_of(text):
            if rnd is not None and rnd.family_of(n) == a.family:
                req = rescore.request(name, minutes, "binance", "BTC/USDT", pinned)
                ideas.add(idea_of(req.params, None) or legacy_idea_hash(req.strategy))
        returns = pooled_returns((sheets or TEARSHEETS) / f"{a.sheet}-pooled.csv")
        result = compute(trials.TrialsRegister(store), text, a.family, returns, ideas)
    except (ValueError, OSError, KeyError) as exc:
        print(exc if str(exc).startswith("refused:") else f"refused: {exc}")
        return 1
    print("\n".join(report(result)))
    print(f"done: family {a.family}, sheet {a.sheet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
